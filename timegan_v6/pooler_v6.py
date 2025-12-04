"""
Latent Pooler for TimeGAN V6

Pools a sequence of latent vectors to a single trajectory summary vector.
This is a key component of the RTSGAN-style architecture.

The pooler compresses temporal information into a fixed-size representation
that captures global trajectory characteristics.

Pool Types:
- 'last': Use last hidden state (simple but loses early information)
- 'mean': Average pooling (uniform weighting)
- 'attention': Learned attention pooling (recommended)
- 'hybrid': Concatenate last + mean + max (most information, larger output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_v6 import create_mask, get_last_valid_output


class LatentPooler(nn.Module):
    """
    Pool sequence of latent vectors to single trajectory summary.

    This module takes the encoder output (sequence of latent vectors)
    and compresses it to a single summary vector that captures the
    essential characteristics of the entire trajectory.

    The summary vector is used by:
    1. The discriminator (in Stage 2) to distinguish real vs fake
    2. As the target distribution for the generator to match
    """

    def __init__(self, latent_dim: int, pool_type: str = 'attention',
                 output_dim: int = None, dropout: float = 0.1):
        """
        Args:
            latent_dim: Dimension of input latent vectors (from encoder)
            pool_type: Pooling strategy - 'last', 'mean', 'attention', or 'hybrid'
            output_dim: Dimension of output summary vector (defaults to latent_dim)
            dropout: Dropout rate for attention mechanism
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.pool_type = pool_type
        self.output_dim = output_dim or latent_dim

        if pool_type == 'attention':
            # Learned attention mechanism
            # Maps each timestep to an attention score
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim // 2, 1)
            )
            # Project pooled vector to output dimension
            self.project = nn.Linear(latent_dim, self.output_dim)

        elif pool_type == 'hybrid':
            # Concatenate last + mean + max → 3x latent_dim
            # Then project down to output_dim
            self.project = nn.Sequential(
                nn.Linear(latent_dim * 3, latent_dim * 2),
                nn.LayerNorm(latent_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim * 2, self.output_dim)
            )

        elif pool_type == 'mean':
            # Simple projection for mean pooling
            if self.output_dim != latent_dim:
                self.project = nn.Linear(latent_dim, self.output_dim)
            else:
                self.project = nn.Identity()

        elif pool_type == 'last':
            # Simple projection for last-state pooling
            if self.output_dim != latent_dim:
                self.project = nn.Linear(latent_dim, self.output_dim)
            else:
                self.project = nn.Identity()

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. "
                             f"Choose from 'last', 'mean', 'attention', 'hybrid'")

        # Output activation to bound values
        self.output_activation = nn.Tanh()

    def forward(self, h_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence of latent vectors to single summary.

        Args:
            h_seq: (batch, seq_len, latent_dim) - sequence of latent vectors
            lengths: (batch,) - actual sequence lengths (before padding)

        Returns:
            z_summary: (batch, output_dim) - trajectory summary vector in [-1, 1]
        """
        batch_size, seq_len, _ = h_seq.shape
        device = h_seq.device

        # Create mask for valid timesteps: (batch, seq_len)
        mask = create_mask(lengths, seq_len)

        if self.pool_type == 'attention':
            return self._attention_pool(h_seq, mask)

        elif self.pool_type == 'hybrid':
            return self._hybrid_pool(h_seq, mask, lengths)

        elif self.pool_type == 'mean':
            return self._mean_pool(h_seq, mask)

        elif self.pool_type == 'last':
            return self._last_pool(h_seq, lengths)

    def _attention_pool(self, h_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling.

        Learns to weight different timesteps based on their importance.
        """
        # Compute attention scores: (batch, seq_len, 1)
        attn_scores = self.attention(h_seq)

        # Mask padding positions with large negative value
        # mask: (batch, seq_len) -> (batch, seq_len, 1)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)

        # Softmax over sequence dimension
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum of hidden states
        pooled = (h_seq * attn_weights).sum(dim=1)  # (batch, latent_dim)

        # Project and activate
        z_summary = self.project(pooled)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def _hybrid_pool(self, h_seq: torch.Tensor, mask: torch.Tensor,
                     lengths: torch.Tensor) -> torch.Tensor:
        """
        Hybrid pooling: concatenate last + mean + max.

        Captures different aspects of the trajectory:
        - Last: Final state (endpoint characteristics)
        - Mean: Average behavior (overall trajectory style)
        - Max: Peak activations (extreme features)
        """
        # Get last valid timestep for each sequence
        h_last = get_last_valid_output(h_seq, lengths)  # (batch, latent_dim)

        # Mean pool (masked)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        h_sum = (h_seq * mask_expanded).sum(dim=1)  # (batch, latent_dim)
        h_mean = h_sum / lengths.unsqueeze(1).float().clamp(min=1)

        # Max pool (masked) - set padding to large negative before max
        h_masked = h_seq.clone()
        h_masked[~mask] = -1e9
        h_max = h_masked.max(dim=1)[0]  # (batch, latent_dim)

        # Concatenate all three
        h_concat = torch.cat([h_last, h_mean, h_max], dim=-1)  # (batch, 3*latent_dim)

        # Project and activate
        z_summary = self.project(h_concat)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def _mean_pool(self, h_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over valid timesteps.
        """
        mask_expanded = mask.unsqueeze(-1).float()
        h_sum = (h_seq * mask_expanded).sum(dim=1)
        num_valid = mask_expanded.sum(dim=1).clamp(min=1)
        h_mean = h_sum / num_valid

        z_summary = self.project(h_mean)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def _last_pool(self, h_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Last-state pooling: use final valid hidden state.
        """
        h_last = get_last_valid_output(h_seq, lengths)

        z_summary = self.project(h_last)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def get_attention_weights(self, h_seq: torch.Tensor,
                              lengths: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization (attention pooling only).

        Args:
            h_seq: (batch, seq_len, latent_dim)
            lengths: (batch,)

        Returns:
            attn_weights: (batch, seq_len) - attention weights per timestep
        """
        if self.pool_type != 'attention':
            raise ValueError("get_attention_weights only available for attention pooling")

        mask = create_mask(lengths, h_seq.size(1))
        attn_scores = self.attention(h_seq)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)

        return attn_weights.squeeze(-1)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing LatentPooler V6...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test parameters
    batch_size = 4
    seq_len = 50
    latent_dim = 48
    output_dim = 96

    # Create test inputs
    h_seq = torch.randn(batch_size, seq_len, latent_dim)
    lengths = torch.tensor([50, 30, 45, 20])

    # Test each pool type
    pool_types = ['last', 'mean', 'attention', 'hybrid']

    for pool_type in pool_types:
        print(f"\n=== Testing pool_type='{pool_type}' ===")

        pooler = LatentPooler(
            latent_dim=latent_dim,
            pool_type=pool_type,
            output_dim=output_dim
        )

        # Count parameters
        num_params = sum(p.numel() for p in pooler.parameters())
        print(f"Parameters: {num_params:,}")

        # Forward pass
        z_summary = pooler(h_seq, lengths)

        print(f"Input shape: {h_seq.shape}")
        print(f"Output shape: {z_summary.shape}")
        print(f"Output range: [{z_summary.min().item():.4f}, {z_summary.max().item():.4f}]")

        # Verify output shape
        assert z_summary.shape == (batch_size, output_dim), \
            f"Expected ({batch_size}, {output_dim}), got {z_summary.shape}"

        # Verify output in [-1, 1] (due to Tanh)
        assert z_summary.min() >= -1.001 and z_summary.max() <= 1.001, \
            f"Output not in [-1, 1] range"

        # Test gradient flow
        loss = z_summary.sum()
        loss.backward()
        print("Gradient flow: OK")

        # Reset gradients for next test
        pooler.zero_grad()

        print(f"pool_type='{pool_type}': PASS")

    # Test attention weights extraction
    print("\n=== Testing attention weight extraction ===")
    pooler_attn = LatentPooler(latent_dim=latent_dim, pool_type='attention')
    attn_weights = pooler_attn.get_attention_weights(h_seq, lengths)
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (per batch): {attn_weights.sum(dim=1).tolist()}")

    # Verify attention weights sum to ~1 for each sequence
    for i, (w_sum, length) in enumerate(zip(attn_weights.sum(dim=1), lengths)):
        assert abs(w_sum.item() - 1.0) < 0.01, \
            f"Attention weights should sum to 1, got {w_sum.item()}"
    print("Attention weights: PASS")

    # Test with different input/output dims
    print("\n=== Testing dimension flexibility ===")
    for in_dim, out_dim in [(32, 64), (64, 32), (48, 48)]:
        pooler = LatentPooler(latent_dim=in_dim, pool_type='attention', output_dim=out_dim)
        h = torch.randn(2, 20, in_dim)
        lengths = torch.tensor([20, 15])
        z = pooler(h, lengths)
        assert z.shape == (2, out_dim), f"Dimension mismatch: {z.shape} vs (2, {out_dim})"
    print("Dimension flexibility: PASS")

    print("\n" + "=" * 60)
    print("ALL LATENT POOLER TESTS PASSED")
    print("=" * 60)
