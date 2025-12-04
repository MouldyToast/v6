"""
Utility Functions for TimeGAN V6

Contains helper functions for:
- Mask creation for variable-length sequences
- Masked loss computation
- Gradient penalty for WGAN-GP in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a boolean mask for variable-length sequences.

    Args:
        lengths: (batch,) - actual lengths of each sequence
        max_len: Maximum sequence length (for padding)

    Returns:
        mask: (batch, max_len) - True for valid positions, False for padding

    Example:
        lengths = [3, 5, 2]
        max_len = 5
        mask = [[T, T, T, F, F],
                [T, T, T, T, T],
                [T, T, F, F, F]]
    """
    batch_size = lengths.size(0)
    device = lengths.device

    # Create range tensor [0, 1, 2, ..., max_len-1]
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)

    # Compare with lengths to create mask
    # lengths: (batch, 1), range: (1, max_len) -> broadcasts to (batch, max_len)
    mask = range_tensor < lengths.unsqueeze(1)

    return mask


def masked_mse_loss(target: torch.Tensor, prediction: torch.Tensor,
                    lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss with masking for variable-length sequences.

    Only computes loss on valid (non-padded) positions.

    Args:
        target: (batch, seq_len, dim) - ground truth
        prediction: (batch, seq_len, dim) - model predictions
        lengths: (batch,) - actual lengths of each sequence

    Returns:
        loss: Scalar tensor - mean MSE over valid positions
    """
    batch_size, seq_len, dim = target.shape

    # Create mask: (batch, seq_len)
    mask = create_mask(lengths, seq_len)

    # Expand mask for feature dimension: (batch, seq_len, 1)
    mask_expanded = mask.unsqueeze(-1).float()

    # Compute squared error
    squared_error = (target - prediction) ** 2

    # Apply mask and compute mean
    masked_error = squared_error * mask_expanded

    # Sum over all dimensions, divide by number of valid elements
    total_error = masked_error.sum()
    num_valid = mask_expanded.sum() * dim  # Account for feature dimension

    # Avoid division by zero
    if num_valid == 0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)

    return total_error / num_valid


def masked_l1_loss(target: torch.Tensor, prediction: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss with masking for variable-length sequences.

    Args:
        target: (batch, seq_len, dim) - ground truth
        prediction: (batch, seq_len, dim) - model predictions
        lengths: (batch,) - actual lengths of each sequence

    Returns:
        loss: Scalar tensor - mean L1 over valid positions
    """
    batch_size, seq_len, dim = target.shape

    # Create mask: (batch, seq_len)
    mask = create_mask(lengths, seq_len)

    # Expand mask for feature dimension: (batch, seq_len, 1)
    mask_expanded = mask.unsqueeze(-1).float()

    # Compute absolute error
    abs_error = torch.abs(target - prediction)

    # Apply mask and compute mean
    masked_error = abs_error * mask_expanded

    # Sum over all dimensions, divide by number of valid elements
    total_error = masked_error.sum()
    num_valid = mask_expanded.sum() * dim

    if num_valid == 0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)

    return total_error / num_valid


def compute_gradient_penalty_latent(discriminator: nn.Module,
                                    z_real: torch.Tensor,
                                    z_fake: torch.Tensor,
                                    condition: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP in latent space.

    This is simpler than sequence-level GP because we operate on
    fixed-dimension vectors (latent summaries).

    Args:
        discriminator: LatentDiscriminator network
        z_real: (batch, summary_dim) - real latent summaries
        z_fake: (batch, summary_dim) - generated latent summaries
        condition: (batch, condition_dim) - conditioning variable

    Returns:
        penalty: Scalar tensor - gradient penalty (unscaled)
    """
    batch_size = z_real.size(0)
    device = z_real.device

    # Random interpolation weight
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolate between real and fake
    z_interpolated = (alpha * z_real + (1 - alpha) * z_fake).requires_grad_(True)

    # Check for NaN/Inf
    if torch.isnan(z_interpolated).any() or torch.isinf(z_interpolated).any():
        print("Warning: NaN/Inf in interpolated latents")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Get discriminator score on interpolated samples
    d_interpolated = discriminator(z_interpolated, condition)

    # Check discriminator output
    if torch.isnan(d_interpolated).any() or torch.isinf(d_interpolated).any():
        print("Warning: NaN/Inf in discriminator output")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute gradients
    try:
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=z_interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    except RuntimeError as e:
        print(f"Warning: Gradient computation failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Check gradients
    if torch.isnan(gradients).any() or torch.isinf(gradients).any():
        print("Warning: NaN/Inf in gradients")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


def get_last_valid_output(h_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Extract the last valid hidden state for each sequence.

    Args:
        h_seq: (batch, seq_len, dim) - sequence of hidden states
        lengths: (batch,) - actual lengths of each sequence

    Returns:
        h_last: (batch, dim) - last valid hidden state per sequence
    """
    batch_size = h_seq.size(0)
    device = h_seq.device

    # Get indices of last valid timestep (lengths - 1, clamped to valid range)
    last_idx = (lengths - 1).clamp(min=0).long()

    # Create batch indices
    batch_idx = torch.arange(batch_size, device=device)

    # Index into h_seq to get last valid states
    h_last = h_seq[batch_idx, last_idx]

    return h_last


def initialize_weights(module: nn.Module, init_type: str = 'xavier'):
    """
    Initialize network weights.

    Args:
        module: nn.Module to initialize
        init_type: 'xavier', 'kaiming', or 'orthogonal'
    """
    for name, param in module.named_parameters():
        if 'weight' in name:
            if 'lstm' in name.lower() or 'gru' in name.lower():
                # RNN weights: orthogonal initialization
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
            elif len(param.shape) >= 2:
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(param, nonlinearity='leaky_relu')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing V6 Utility Functions...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test 1: create_mask
    print("\n=== Test 1: create_mask ===")
    lengths = torch.tensor([3, 5, 2])
    mask = create_mask(lengths, max_len=5)
    print(f"Lengths: {lengths.tolist()}")
    print(f"Mask:\n{mask.int()}")

    expected = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
        [True, True, False, False, False]
    ])
    assert torch.equal(mask, expected), "Mask creation failed"
    print("PASS")

    # Test 2: masked_mse_loss
    print("\n=== Test 2: masked_mse_loss ===")
    batch_size, seq_len, dim = 4, 10, 8
    target = torch.randn(batch_size, seq_len, dim)
    prediction = torch.randn(batch_size, seq_len, dim)
    lengths = torch.tensor([10, 7, 5, 8])

    loss = masked_mse_loss(target, prediction, lengths)
    print(f"Masked MSE loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("PASS")

    # Test 3: get_last_valid_output
    print("\n=== Test 3: get_last_valid_output ===")
    h_seq = torch.arange(40).float().view(4, 10, 1).expand(4, 10, 8)
    lengths = torch.tensor([3, 7, 1, 10])
    h_last = get_last_valid_output(h_seq, lengths)
    print(f"h_seq shape: {h_seq.shape}")
    print(f"Lengths: {lengths.tolist()}")
    print(f"h_last shape: {h_last.shape}")
    print(f"Last indices extracted: {(lengths - 1).tolist()}")
    # Verify correct indices were extracted
    for i, (l, h) in enumerate(zip(lengths, h_last)):
        expected_val = (i * 10 + l.item() - 1)  # Index of last valid element
        assert h[0].item() == expected_val, f"Wrong value for batch {i}"
    print("PASS")

    # Test 4: compute_gradient_penalty_latent
    print("\n=== Test 4: compute_gradient_penalty_latent ===")

    # Simple discriminator for testing
    class DummyDiscriminator(nn.Module):
        def __init__(self, summary_dim=96, condition_dim=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(summary_dim + condition_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )

        def forward(self, z, condition):
            x = torch.cat([z, condition], dim=-1)
            return self.net(x)

    disc = DummyDiscriminator()
    z_real = torch.randn(8, 96)
    z_fake = torch.randn(8, 96)
    condition = torch.randn(8, 1)

    gp = compute_gradient_penalty_latent(disc, z_real, z_fake, condition)
    print(f"Gradient penalty: {gp.item():.4f}")
    assert gp.item() >= 0, "GP should be non-negative"
    assert not torch.isnan(gp), "GP should not be NaN"
    print("PASS")

    # Test 5: initialize_weights
    print("\n=== Test 5: initialize_weights ===")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.Linear(128, 64)
    )
    initialize_weights(model, 'xavier')
    print("Xavier initialization applied successfully")
    print("PASS")

    print("\n" + "=" * 60)
    print("ALL V6 UTILITY TESTS PASSED")
    print("=" * 60)
