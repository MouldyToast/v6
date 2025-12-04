"""
Loss Functions for TimeGAN Pipeline
Implements all loss functions for 3-phase training.

Compatibility:
- Works with v2 (feature_dim=3) and v3 (feature_dim=4)
- Handles dynamic batching via 'lengths' argument in GP
- Dimension agnostic

Location: utils/losses.py
"""

import torch
import torch.nn as nn


def reconstruction_loss(x_real, x_recon):
    """
    MSE loss for embedder-recovery training (Phase 1).

    Args:
        x_real: (batch, seq_len, feature_dim) - real trajectories
        x_recon: (batch, seq_len, feature_dim) - reconstructed trajectories

    Returns:
        loss: Scalar tensor
    """
    return nn.MSELoss()(x_recon, x_real)


def supervised_loss(h_real, h_supervised):
    """
    Supervised loss for Phase 2 training (NEW in v2.0).
    Teaches temporal dynamics: h(t) -> h(t+1)

    This is CRITICAL for preventing mode collapse!

    Args:
        h_real: (batch, seq_len, latent_dim) - real embedded sequences
        h_supervised: (batch, seq_len, latent_dim) - supervisor predictions

    Returns:
        loss: Scalar tensor
    """
    # Shift sequences by 1 timestep
    # h_real[:, 1:, :] is the ground truth h(t+1) to h(T)
    # h_supervised[:, :-1, :] is the predicted h(t+1) from h(t)
    h_real_shifted = h_real[:, 1:, :]  # Target: h(t+1) to h(T)
    h_supervised_pred = h_supervised[:, :-1, :]  # Prediction: h(t+1) from h(t)

    # MSE between predicted and actual next-step
    return nn.MSELoss()(h_supervised_pred, h_real_shifted)


def wasserstein_loss(d_real, d_fake):
    """
    Wasserstein distance for WGAN discriminator.

    Discriminator tries to maximize: E[D(real)] - E[D(fake)]
    Which is equivalent to minimizing: E[D(fake)] - E[D(real)]

    Args:
        d_real: (batch, 1) - discriminator scores for real data
        d_fake: (batch, 1) - discriminator scores for fake data

    Returns:
        loss: Scalar tensor (discriminator loss to minimize)
    """
    # Discriminator wants: D(real) > D(fake)
    # So minimize: -D(real) + D(fake) = D(fake) - D(real)
    return d_fake.mean() - d_real.mean()


def gradient_penalty(discriminator, real_data, fake_data, condition, device, lengths=None, lambda_gp=None):
    """
    Compute gradient penalty for WGAN-GP.

    Enforces 1-Lipschitz constraint by penalizing gradients that deviate from norm 1.

    NOTE: lambda_gp parameter is DEPRECATED and ignored. The caller should apply
    lambda_gp scaling externally to avoid double application.

    Args:
        discriminator: Discriminator network
        real_data: (batch, seq_len, feature_dim) - real trajectories
        fake_data: (batch, seq_len, feature_dim) - fake trajectories
        condition: (batch, condition_dim) - conditioning variable
        device: torch device
        lengths: (batch,) - sequence lengths (required for v2 discriminator)
        lambda_gp: DEPRECATED - ignored (caller applies scaling)

    Returns:
        penalty: Scalar tensor (unscaled - caller must apply lambda_gp)
    """
    batch_size = real_data.size(0)

    # Random weight term for interpolation between real and fake
    # Shape: (batch, 1, 1) for broadcasting
    alpha = torch.rand(batch_size, 1, 1, device=device)

    # Interpolate between real and fake data
    # interpolates = alpha * real + (1 - alpha) * fake
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # Check interpolates for NaN/Inf
    if torch.isnan(interpolates).any() or torch.isinf(interpolates).any():
        print(f"\n⚠️  GP: Interpolated samples have NaN/Inf!")
        print(f"   real_data range: [{real_data.min().item():.3f}, {real_data.max().item():.3f}]")
        print(f"   fake_data range: [{fake_data.min().item():.3f}, {fake_data.max().item():.3f}]")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Get discriminator output on interpolated data
    # IMPORTANT: Disable CuDNN for this forward pass because gradient penalty
    # requires double backwards (gradients of gradients), which CuDNN RNNs don't support
    with torch.backends.cudnn.flags(enabled=False):
        if lengths is not None:
            d_interpolates = discriminator(interpolates, condition, lengths)
        else:
            d_interpolates = discriminator(interpolates, condition)

    # Check d_interpolates for NaN/Inf
    if torch.isnan(d_interpolates).any() or torch.isinf(d_interpolates).any():
        print(f"\n⚠️  GP: Discriminator output on interpolates has NaN/Inf!")
        print(f"   interpolates range: [{interpolates.min().item():.3f}, {interpolates.max().item():.3f}]")
        print(f"   d_interpolates range: NaN/Inf detected")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute gradients of discriminator output w.r.t. interpolated input
    try:
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,  # Allow backprop through this computation
            retain_graph=True,
            only_inputs=True
        )[0]
    except RuntimeError as e:
        print(f"\n⚠️  GP: torch.autograd.grad failed!")
        print(f"   Error: {e}")
        print(f"   d_interpolates range: [{d_interpolates.min().item():.3f}, {d_interpolates.max().item():.3f}]")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Check gradients for NaN/Inf
    if torch.isnan(gradients).any() or torch.isinf(gradients).any():
        print(f"\n⚠️  GP: Gradients have NaN/Inf!")
        print(f"   d_interpolates range: [{d_interpolates.min().item():.3f}, {d_interpolates.max().item():.3f}]")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Flatten gradients to compute norm
    gradients = gradients.reshape(batch_size, -1)

    # Compute gradient norm for each sample
    gradient_norm = gradients.norm(2, dim=1)  # L2 norm

    # Check gradient_norm for NaN/Inf
    if torch.isnan(gradient_norm).any() or torch.isinf(gradient_norm).any():
        print(f"\n⚠️  GP: Gradient norms have NaN/Inf!")
        print(f"   gradients range: [{gradients.min().item():.3f}, {gradients.max().item():.3f}]")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Warn if gradient norms are extreme
    if gradient_norm.max() > 1000:
        print(f"\n⚠️  GP: Extreme gradient norms detected!")
        print(f"   gradient_norm: mean={gradient_norm.mean().item():.2f}, max={gradient_norm.max().item():.2f}")

    # Penalty: (||gradient|| - 1)^2
    # Encourages gradient norm to be close to 1
    # NOTE: Caller must apply lambda_gp scaling - we return unscaled penalty
    penalty = ((gradient_norm - 1) ** 2).mean()

    # Final check
    if torch.isnan(penalty) or torch.isinf(penalty):
        print(f"\n⚠️  GP: Final penalty is NaN/Inf!")
        print(f"   gradient_norm range: [{gradient_norm.min().item():.3f}, {gradient_norm.max().item():.3f}]")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return penalty


def generator_loss(d_fake):
    """
    Generator loss for WGAN.

    Generator wants discriminator to output high scores for fake data.
    So it minimizes: -E[D(fake)]

    Args:
        d_fake: (batch, 1) - discriminator scores for generated (fake) data

    Returns:
        loss: Scalar tensor
    """
    # Generator wants to maximize D(fake), so minimize -D(fake)
    return -d_fake.mean()


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing Loss Functions...")

    # Test reconstruction loss
    print("\n=== Test 1: Reconstruction Loss ===")
    x_real = torch.randn(4, 500, 3)
    x_recon = torch.randn(4, 500, 3)
    loss = reconstruction_loss(x_real, x_recon)
    print(f"Reconstruction loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("PASS")

    # Test supervised loss
    print("\n=== Test 2: Supervised Loss (NEW) ===")
    h_real = torch.randn(4, 500, 6)
    h_supervised = torch.randn(4, 500, 6)
    loss = supervised_loss(h_real, h_supervised)
    print(f"Supervised loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("PASS")

    # Test Wasserstein loss
    print("\n=== Test 3: Wasserstein Loss ===")
    d_real = torch.randn(4, 1)
    d_fake = torch.randn(4, 1)
    loss = wasserstein_loss(d_real, d_fake)
    print(f"Wasserstein loss: {loss.item():.4f}")
    print(f"  D(real) mean: {d_real.mean().item():.4f}")
    print(f"  D(fake) mean: {d_fake.mean().item():.4f}")
    print("PASS")

    # Test generator loss
    print("\n=== Test 4: Generator Loss ===")
    d_fake = torch.randn(4, 1)
    loss = generator_loss(d_fake)
    print(f"Generator loss: {loss.item():.4f}")
    print("PASS")

    # Test gradient penalty
    print("\n=== Test 5: Gradient Penalty ===")

    # Simple discriminator for testing
    class DummyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1500, 1)  # 500 * 3 = 1500

        def forward(self, x, condition):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            return self.fc(x_flat)

    device = torch.device('cpu')
    disc = DummyDiscriminator()

    real_data = torch.randn(4, 500, 3)
    fake_data = torch.randn(4, 500, 3)
    condition = torch.randn(4, 1)

    gp = gradient_penalty(disc, real_data, fake_data, condition, device)
    print(f"Gradient penalty (unscaled): {gp.item():.4f}")
    print(f"With lambda_gp=10: {(10 * gp).item():.4f}")
    assert gp.item() >= 0, "Gradient penalty should be non-negative"
    print("PASS")

    print("\n" + "="*70)
    print("ALL LOSS FUNCTION TESTS PASSED")
    print("="*70)
    print("\nv2.0 Features Implemented:")
    print("  - Reconstruction loss (Phase 1)")
    print("  - Supervised loss (Phase 2 - NEW)")
    print("  - Wasserstein loss (Phase 3)")
    print("  - Gradient penalty (Phase 3)")
    print("  - Generator loss (Phase 3)")
