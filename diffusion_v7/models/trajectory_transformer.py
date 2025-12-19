"""
Trajectory Transformer for Diffusion Model

Adapted from MotionDiffuse transformer for mouse trajectory generation.
Replaces text conditioning with 3D goal conditioning (distance + angle).

Key differences from MotionDiffuse:
- Input/output: 263D motion → 8D trajectories
- Conditioning: Text (CLIP) → 3D goal (GoalConditioner)
- Architecture: Removed cross-attention, direct goal conditioning via FiLM
- Configurable dimensions for smoke testing

Architecture:
    Input (8D trajectory) → Linear → Latent (latent_dim)
    + Timestep embedding (sinusoidal)
    + Goal embedding (from GoalConditioner)
    → Transformer Blocks (temporal self-attention + FFN with FiLM)
    → Output projection → 8D trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# =============================================================================
# Utility Functions (from MotionDiffuse)
# =============================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: (batch,) - diffusion timestep indices
        dim: Embedding dimension
        max_period: Controls minimum frequency

    Returns:
        embedding: (batch, dim) - sinusoidal positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# =============================================================================
# Building Blocks
# =============================================================================

class StylizationBlock(nn.Module):
    """
    FiLM-style conditioning block.

    Modulates hidden states using scale and shift from conditioning embeddings.
    Used to inject timestep and goal information into the model.

    Args:
        latent_dim: Hidden dimension
        cond_embed_dim: Conditioning embedding dimension (timestep + goal)
        dropout: Dropout rate
    """

    def __init__(self, latent_dim, cond_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        Args:
            h: (batch, seq_len, latent_dim) - hidden states
            emb: (batch, cond_embed_dim) - conditioning embedding

        Returns:
            h: (batch, seq_len, latent_dim) - modulated hidden states
        """
        # (batch, 2 * latent_dim) → (batch, 1, 2 * latent_dim)
        emb_out = self.emb_layers(emb).unsqueeze(1)

        # Split into scale and shift
        scale, shift = torch.chunk(emb_out, 2, dim=2)

        # Apply FiLM: h = norm(h) * (1 + scale) + shift
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class TemporalSelfAttention(nn.Module):
    """
    Masked temporal self-attention.

    Attends over sequence timesteps with masking for variable-length sequences.

    Args:
        latent_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        cond_embed_dim: Conditioning embedding dimension
    """

    def __init__(self, latent_dim, num_heads, dropout, cond_embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)

        # QKV projections
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, cond_embed_dim, dropout)

    def forward(self, x, cond_emb, src_mask):
        """
        Args:
            x: (batch, seq_len, latent_dim) - input sequences
            cond_emb: (batch, cond_embed_dim) - conditioning embedding
            src_mask: (batch, seq_len) - mask (1=valid, 0=padding)

        Returns:
            x: (batch, seq_len, latent_dim) - attention output
        """
        B, T, D = x.shape
        H = self.num_heads

        # Normalize
        x_norm = self.norm(x)

        # Project to Q, K, V
        query = self.query(x_norm)  # (B, T, D)
        key = self.key(x_norm)      # (B, T, D)
        value = self.value(x_norm)  # (B, T, D)

        # Reshape for multi-head attention
        query = query.view(B, T, H, D // H).transpose(1, 2)  # (B, H, T, D//H)
        key = key.view(B, T, H, D // H).transpose(1, 2)      # (B, H, T, D//H)
        value = value.view(B, T, H, D // H).transpose(1, 2)  # (B, H, T, D//H)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D // H)  # (B, H, T, T)

        # Apply mask: (B, T) → (B, 1, 1, T) → broadcast
        # src_mask: 1=valid, 0=padding
        # attention_mask: 0=valid, -inf=padding
        attention_mask = (1 - src_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        scores = scores + attention_mask

        # Softmax over keys
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # (B, H, T, D//H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # Residual + FiLM conditioning
        x = x + self.proj_out(attn_output, cond_emb)

        return x


class FeedForward(nn.Module):
    """
    Position-wise feedforward network with FiLM conditioning.

    Args:
        latent_dim: Hidden dimension
        ff_size: Feedforward expansion size
        dropout: Dropout rate
        cond_embed_dim: Conditioning embedding dimension
        activation: Activation function ('gelu', 'relu', 'silu')
    """

    def __init__(self, latent_dim, ff_size, dropout, cond_embed_dim, activation='gelu'):
        super().__init__()

        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # FFN layers
        self.linear1 = nn.Linear(latent_dim, ff_size)
        self.linear2 = nn.Linear(ff_size, latent_dim)
        self.dropout = nn.Dropout(dropout)

        # FiLM conditioning
        self.norm = nn.LayerNorm(latent_dim)
        self.stylization = StylizationBlock(latent_dim, cond_embed_dim, dropout)

    def forward(self, x, cond_emb):
        """
        Args:
            x: (batch, seq_len, latent_dim)
            cond_emb: (batch, cond_embed_dim)

        Returns:
            x: (batch, seq_len, latent_dim)
        """
        # Normalize
        h = self.norm(x)

        # FFN
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)

        # Residual + FiLM conditioning
        x = x + self.stylization(h, cond_emb)

        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block: Self-Attention + FFN.

    Args:
        latent_dim: Hidden dimension
        num_heads: Number of attention heads
        ff_size: Feedforward expansion size
        dropout: Dropout rate
        cond_embed_dim: Conditioning embedding dimension
        activation: Activation function
    """

    def __init__(self, latent_dim, num_heads, ff_size, dropout, cond_embed_dim, activation='gelu'):
        super().__init__()

        self.self_attn = TemporalSelfAttention(latent_dim, num_heads, dropout, cond_embed_dim)
        self.ffn = FeedForward(latent_dim, ff_size, dropout, cond_embed_dim, activation)

    def forward(self, x, cond_emb, src_mask):
        """
        Args:
            x: (batch, seq_len, latent_dim)
            cond_emb: (batch, cond_embed_dim) - combined timestep + goal embedding
            src_mask: (batch, seq_len) - mask (1=valid, 0=padding)

        Returns:
            x: (batch, seq_len, latent_dim)
        """
        x = self.self_attn(x, cond_emb, src_mask)
        x = self.ffn(x, cond_emb)
        return x


# =============================================================================
# Main Model
# =============================================================================

class TrajectoryTransformer(nn.Module):
    """
    Transformer-based denoiser for trajectory diffusion.

    Predicts noise given noisy trajectory, diffusion timestep, and goal condition.

    Args:
        input_dim: Input trajectory dimension (8 for our features)
        latent_dim: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_size: Feedforward expansion size
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        activation: Activation function
        goal_latent_dim: Goal embedding dimension (from GoalConditioner)
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 512,
        num_layers: int = 9,
        num_heads: int = 8,
        ff_size: int = 1024,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        activation: str = 'gelu',
        goal_latent_dim: int = 512  # Should match GoalConditioner output
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.max_seq_len = max_seq_len

        # Conditioning embedding dimension (timestep + goal)
        self.time_embed_dim = latent_dim
        self.cond_embed_dim = self.time_embed_dim + goal_latent_dim

        # Input embedding: trajectory features → latent
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Positional encoding for sequences
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, latent_dim))

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                latent_dim=latent_dim,
                num_heads=num_heads,
                ff_size=ff_size,
                dropout=dropout,
                cond_embed_dim=self.cond_embed_dim,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Output projection: latent → trajectory features
        self.output_proj = zero_module(nn.Linear(latent_dim, input_dim))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        goal_embed: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) - noisy trajectories
            t: (batch,) - diffusion timesteps (0 to T-1)
            goal_embed: (batch, goal_latent_dim) - goal embeddings from GoalConditioner
            lengths: (batch,) - original sequence lengths before padding

        Returns:
            noise_pred: (batch, seq_len, input_dim) - predicted noise
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Input embedding
        h = self.input_proj(x)  # (batch, seq_len, latent_dim)

        # Add positional encoding
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        # Timestep embedding
        t_emb = timestep_embedding(t, self.latent_dim)  # (batch, latent_dim)
        t_emb = self.time_embed(t_emb)  # (batch, time_embed_dim)

        # Combine timestep + goal embeddings
        cond_emb = torch.cat([t_emb, goal_embed], dim=-1)  # (batch, cond_embed_dim)

        # Create attention mask from lengths
        src_mask = self.generate_src_mask(seq_len, lengths).to(device)  # (batch, seq_len)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond_emb, src_mask)

        # Output projection
        out = self.output_proj(h)  # (batch, seq_len, input_dim)

        return out

    def generate_src_mask(self, seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """
        Generate attention mask from sequence lengths.

        Args:
            seq_len: Maximum sequence length
            lengths: (batch,) - actual lengths

        Returns:
            mask: (batch, seq_len) - 1 for valid positions, 0 for padding
        """
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)

        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0

        return mask


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing TrajectoryTransformer...")
    print("=" * 70)

    # Test parameters
    batch_size = 4
    seq_len = 50
    input_dim = 8
    latent_dim = 128
    goal_latent_dim = 128

    # Create model
    print("\n=== Creating Model ===")
    model = TrajectoryTransformer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_layers=4,
        num_heads=4,
        ff_size=256,
        max_seq_len=200,
        dropout=0.1,
        activation='gelu',
        goal_latent_dim=goal_latent_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test inputs
    x = torch.randn(batch_size, seq_len, input_dim)
    t = torch.randint(0, 1000, (batch_size,))
    goal_embed = torch.randn(batch_size, goal_latent_dim)
    lengths = torch.tensor([50, 30, 45, 20])

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  goal_embed: {goal_embed.shape}")
    print(f"  lengths: {lengths}")

    # Forward pass
    print("\n=== Forward Pass ===")
    output = model(x, t, goal_embed, lengths)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"

    # Test gradient flow
    print("\n=== Gradient Flow ===")
    loss = output.mean()
    loss.backward()

    has_grad = 0
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
            total_grad_norm += param.grad.norm().item()

    print(f"Parameters with gradients: {has_grad}/{len(list(model.parameters()))}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    # Test masking
    print("\n=== Test Masking ===")
    mask = model.generate_src_mask(seq_len, lengths)
    print(f"Mask shape: {mask.shape}")
    for i, length in enumerate(lengths):
        valid_count = mask[i].sum().item()
        print(f"  Sequence {i}: length={length}, valid_positions={valid_count}")
        assert valid_count == length, f"Mask mismatch for sequence {i}"

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
