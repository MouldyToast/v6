"""
Loss Functions for TimeGAN V6

Consolidated loss functions for both training stages:
- Stage 1: Reconstruction and latent consistency losses
- Stage 2: WGAN-GP losses (discriminator, generator, gradient penalty, feature matching)

All losses handle variable-length sequences via masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .utils_v6 import create_mask, masked_mse_loss, masked_l1_loss


# =============================================================================
# Stage 1: Autoencoder Losses
# =============================================================================

def reconstruction_loss(x_real: torch.Tensor, x_recon: torch.Tensor,
                        lengths: torch.Tensor, loss_type: str = 'mse') -> torch.Tensor:
    """
    Compute reconstruction loss with masking.

    Args:
        x_real: (batch, seq_len, feature_dim) - real trajectories
        x_recon: (batch, seq_len, feature_dim) - reconstructed trajectories
        lengths: (batch,) - sequence lengths
        loss_type: 'mse' or 'l1'

    Returns:
        loss: Scalar tensor
    """
    if loss_type == 'mse':
        return masked_mse_loss(x_real, x_recon, lengths)
    elif loss_type == 'l1':
        return masked_l1_loss(x_real, x_recon, lengths)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def latent_consistency_loss(h_seq: torch.Tensor, h_seq_recon: torch.Tensor,
                            lengths: torch.Tensor) -> torch.Tensor:
    """
    Latent consistency loss: encourage expander output to match encoder output.

    This helps the expander learn to reconstruct the full latent sequence
    from just the summary vector.

    Args:
        h_seq: (batch, seq_len, latent_dim) - encoder output
        h_seq_recon: (batch, seq_len, latent_dim) - expander output
        lengths: (batch,) - sequence lengths

    Returns:
        loss: Scalar tensor
    """
    return masked_mse_loss(h_seq.detach(), h_seq_recon, lengths)


def autoencoder_loss(x_real: torch.Tensor, x_recon: torch.Tensor,
                     h_seq: torch.Tensor, h_seq_recon: torch.Tensor,
                     lengths: torch.Tensor,
                     lambda_recon: float = 1.0,
                     lambda_latent: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Combined autoencoder loss for Stage 1.

    Args:
        x_real: Real trajectories
        x_recon: Reconstructed trajectories
        h_seq: Encoder latent sequence
        h_seq_recon: Expander latent sequence
        lengths: Sequence lengths
        lambda_recon: Weight for reconstruction loss
        lambda_latent: Weight for latent consistency loss

    Returns:
        Dict with 'recon', 'latent', 'total' losses
    """
    loss_recon = reconstruction_loss(x_real, x_recon, lengths)
    loss_latent = latent_consistency_loss(h_seq, h_seq_recon, lengths)

    loss_total = lambda_recon * loss_recon + lambda_latent * loss_latent

    return {
        'recon': loss_recon,
        'latent': loss_latent,
        'total': loss_total
    }


# =============================================================================
# Stage 2: WGAN-GP Losses
# =============================================================================

def wasserstein_discriminator_loss(d_real: torch.Tensor,
                                   d_fake: torch.Tensor) -> torch.Tensor:
    """
    WGAN discriminator loss (without gradient penalty).

    D tries to maximize: E[D(real)] - E[D(fake)]
    Which means minimizing: E[D(fake)] - E[D(real)]

    Args:
        d_real: (batch, 1) - discriminator scores for real data
        d_fake: (batch, 1) - discriminator scores for fake data

    Returns:
        loss: Scalar tensor
    """
    return d_fake.mean() - d_real.mean()


def wasserstein_generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    WGAN generator loss.

    G tries to maximize: E[D(fake)]
    Which means minimizing: -E[D(fake)]

    Args:
        d_fake: (batch, 1) - discriminator scores for fake data

    Returns:
        loss: Scalar tensor
    """
    return -d_fake.mean()


def gradient_penalty_loss(discriminator: nn.Module,
                          z_real: torch.Tensor,
                          z_fake: torch.Tensor,
                          condition: torch.Tensor,
                          lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Gradient penalty for WGAN-GP (in latent space).

    Enforces 1-Lipschitz constraint.

    Args:
        discriminator: Discriminator network
        z_real: (batch, summary_dim) - real latent summaries
        z_fake: (batch, summary_dim) - fake latent summaries
        condition: (batch, condition_dim) - conditions
        lambda_gp: Gradient penalty weight

    Returns:
        Scaled gradient penalty
    """
    from .utils_v6 import compute_gradient_penalty_latent
    gp = compute_gradient_penalty_latent(discriminator, z_real, z_fake, condition)
    return lambda_gp * gp


def feature_matching_loss(z_fake: torch.Tensor,
                          z_real: torch.Tensor,
                          match_std: bool = True) -> torch.Tensor:
    """
    Feature matching loss: match moments of latent distributions.

    Helps stabilize GAN training by providing an additional learning signal.

    Args:
        z_fake: (batch, summary_dim) - generated latent summaries
        z_real: (batch, summary_dim) - real latent summaries (detached)
        match_std: Also match standard deviation

    Returns:
        loss: Scalar tensor
    """
    # Match means
    loss = F.mse_loss(z_fake.mean(dim=0), z_real.mean(dim=0))

    # Match standard deviations
    if match_std:
        loss = loss + F.mse_loss(z_fake.std(dim=0), z_real.std(dim=0))

    return loss


def discriminator_loss(d_real: torch.Tensor,
                       d_fake: torch.Tensor,
                       gp: torch.Tensor,
                       lambda_gp: float = 10.0) -> Dict[str, torch.Tensor]:
    """
    Full WGAN-GP discriminator loss.

    Args:
        d_real: Discriminator scores for real
        d_fake: Discriminator scores for fake
        gp: Gradient penalty (unscaled)
        lambda_gp: GP weight

    Returns:
        Dict with 'wasserstein', 'gp', 'total' losses
    """
    loss_w = wasserstein_discriminator_loss(d_real, d_fake)
    loss_gp = lambda_gp * gp

    return {
        'wasserstein': -loss_w,  # Positive = good (D(real) > D(fake))
        'gp': gp,
        'total': loss_w + loss_gp
    }


def generator_loss(d_fake: torch.Tensor,
                   z_fake: torch.Tensor = None,
                   z_real: torch.Tensor = None,
                   lambda_fm: float = 1.0,
                   use_feature_matching: bool = True) -> Dict[str, torch.Tensor]:
    """
    Full generator loss with optional feature matching.

    Args:
        d_fake: Discriminator scores for fake
        z_fake: Generated latent summaries (for feature matching)
        z_real: Real latent summaries (for feature matching, detached)
        lambda_fm: Feature matching weight
        use_feature_matching: Whether to use feature matching

    Returns:
        Dict with 'adversarial', 'feature_matching', 'total' losses
    """
    loss_adv = wasserstein_generator_loss(d_fake)

    losses = {
        'adversarial': loss_adv,
        'feature_matching': torch.tensor(0.0, device=d_fake.device)
    }

    if use_feature_matching and z_fake is not None and z_real is not None:
        loss_fm = feature_matching_loss(z_fake, z_real.detach())
        losses['feature_matching'] = loss_fm
        losses['total'] = loss_adv + lambda_fm * loss_fm
    else:
        losses['total'] = loss_adv

    return losses


# =============================================================================
# Auxiliary Losses (Optional)
# =============================================================================

def temporal_consistency_loss(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Encourage smooth trajectories (optional regularization).

    Penalizes large changes between consecutive timesteps.

    Args:
        x: (batch, seq_len, feature_dim) - trajectories
        lengths: (batch,) - sequence lengths

    Returns:
        loss: Scalar tensor
    """
    # Compute differences between consecutive timesteps
    diff = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq_len-1, feature_dim)

    # Create mask for valid differences
    mask = create_mask(lengths - 1, diff.size(1))
    mask_expanded = mask.unsqueeze(-1).float()

    # Squared differences
    sq_diff = (diff ** 2) * mask_expanded

    # Mean over valid positions
    total = sq_diff.sum()
    count = mask_expanded.sum() * x.size(2)

    if count == 0:
        return torch.tensor(0.0, device=x.device)

    return total / count


def diversity_loss(z_samples: torch.Tensor, min_dist: float = 0.1) -> torch.Tensor:
    """
    Encourage diversity in generated samples (anti-mode-collapse).

    Penalizes samples that are too similar to each other.

    Args:
        z_samples: (batch, dim) - generated latent vectors
        min_dist: Minimum desired pairwise distance

    Returns:
        loss: Scalar tensor (0 if diversity is sufficient)
    """
    batch_size = z_samples.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=z_samples.device)

    # Compute pairwise distances
    # (batch, 1, dim) - (1, batch, dim) -> (batch, batch, dim)
    diff = z_samples.unsqueeze(1) - z_samples.unsqueeze(0)
    dist = diff.norm(dim=-1)  # (batch, batch)

    # Only consider off-diagonal elements
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_samples.device)
    off_diag = dist[mask]

    # Penalize distances below min_dist
    violations = F.relu(min_dist - off_diag)

    return violations.mean()


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == '__main__':
    print("Testing V6 Loss Functions...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test data
    batch_size = 4
    seq_len = 50
    feature_dim = 8
    latent_dim = 48
    summary_dim = 96

    x_real = torch.randn(batch_size, seq_len, feature_dim)
    x_recon = torch.randn(batch_size, seq_len, feature_dim)
    h_seq = torch.randn(batch_size, seq_len, latent_dim)
    h_seq_recon = torch.randn(batch_size, seq_len, latent_dim)
    lengths = torch.tensor([50, 30, 45, 20])

    z_real = torch.randn(batch_size, summary_dim)
    z_fake = torch.randn(batch_size, summary_dim)
    d_real = torch.randn(batch_size, 1)
    d_fake = torch.randn(batch_size, 1)
    condition = torch.rand(batch_size, 1)

    # Test Stage 1 losses
    print("\n=== Stage 1 Losses ===")

    loss_recon = reconstruction_loss(x_real, x_recon, lengths)
    print(f"Reconstruction loss: {loss_recon.item():.4f}")
    assert loss_recon.item() > 0
    print("  PASS")

    loss_latent = latent_consistency_loss(h_seq, h_seq_recon, lengths)
    print(f"Latent consistency loss: {loss_latent.item():.4f}")
    assert loss_latent.item() > 0
    print("  PASS")

    ae_losses = autoencoder_loss(x_real, x_recon, h_seq, h_seq_recon, lengths)
    print(f"Autoencoder total loss: {ae_losses['total'].item():.4f}")
    assert 'recon' in ae_losses and 'latent' in ae_losses and 'total' in ae_losses
    print("  PASS")

    # Test Stage 2 losses
    print("\n=== Stage 2 Losses ===")

    loss_w_d = wasserstein_discriminator_loss(d_real, d_fake)
    print(f"Wasserstein D loss: {loss_w_d.item():.4f}")
    print("  PASS")

    loss_w_g = wasserstein_generator_loss(d_fake)
    print(f"Wasserstein G loss: {loss_w_g.item():.4f}")
    print("  PASS")

    loss_fm = feature_matching_loss(z_fake, z_real)
    print(f"Feature matching loss: {loss_fm.item():.4f}")
    assert loss_fm.item() >= 0
    print("  PASS")

    d_losses = discriminator_loss(d_real, d_fake, torch.tensor(0.5), lambda_gp=10.0)
    print(f"Discriminator total loss: {d_losses['total'].item():.4f}")
    assert 'wasserstein' in d_losses and 'gp' in d_losses and 'total' in d_losses
    print("  PASS")

    g_losses = generator_loss(d_fake, z_fake, z_real, lambda_fm=1.0)
    print(f"Generator total loss: {g_losses['total'].item():.4f}")
    assert 'adversarial' in g_losses and 'feature_matching' in g_losses
    print("  PASS")

    # Test auxiliary losses
    print("\n=== Auxiliary Losses ===")

    loss_temp = temporal_consistency_loss(x_real, lengths)
    print(f"Temporal consistency loss: {loss_temp.item():.4f}")
    print("  PASS")

    loss_div = diversity_loss(z_fake)
    print(f"Diversity loss: {loss_div.item():.4f}")
    print("  PASS")

    # Test gradient flow
    print("\n=== Gradient Flow ===")
    x_recon_grad = torch.randn(batch_size, seq_len, feature_dim, requires_grad=True)
    loss = reconstruction_loss(x_real, x_recon_grad, lengths)
    loss.backward()
    assert x_recon_grad.grad is not None
    print("Gradient flow: PASS")

    print("\n" + "=" * 60)
    print("ALL LOSS FUNCTION TESTS PASSED")
    print("=" * 60)
