"""
Trajectory Transformer for Diffusion Model V8

V8 Key Changes from V7:
- input_dim: 8 -> 2 (x, y positions only)
- Removed goal_latent_dim parameter
- Removed goal_embed from forward signature
- cond_embed_dim = time_embed_dim (not time_embed_dim + goal_latent_dim)

Direction is controlled via inpainting during sampling, not goal embeddings.

Architecture:
    Input (2D positions) -> Linear -> Latent (latent_dim)
    + Timestep embedding (sinusoidal)
    -> Transformer Blocks (temporal self-attention + FFN with FiLM)
    -> Output projection -> 2D positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# =============================================================================
# Utility Functions
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
    V8: Only uses timestep embedding (no goal embedding).

    Args:
        latent_dim: Hidden dimension
        cond_embed_dim: Conditioning embedding dimension (timestep only in V8)
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
        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
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

        x_norm = self.norm(x)

        query = self.query(x_norm)
        key = self.key(x_norm)
        value = self.value(x_norm)

        query = query.view(B, T, H, D // H).transpose(1, 2)
        key = key.view(B, T, H, D // H).transpose(1, 2)
        value = value.view(B, T, H, D // H).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D // H)

        attention_mask = (1 - src_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

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

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.linear1 = nn.Linear(latent_dim, ff_size)
        self.linear2 = nn.Linear(ff_size, latent_dim)
        self.dropout = nn.Dropout(dropout)

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
        h = self.norm(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
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
            cond_emb: (batch, cond_embed_dim) - V8: timestep embedding only
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
    Transformer-based denoiser for trajectory diffusion V8.

    V8 Changes:
    - Input/output: 2D (x, y positions) instead of 8D features
    - No goal embedding - direction controlled via inpainting
    - cond_embed_dim = time_embed_dim (smaller than V7)

    Args:
        input_dim: Input dimension (2 for V8: x, y positions)
        latent_dim: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_size: Feedforward expansion size
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int = 2,         # V8: 2D positions (was 8 in V7)
        latent_dim: int = 512,
        num_layers: int = 9,
        num_heads: int = 8,
        ff_size: int = 1024,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        activation: str = 'gelu'
        # NOTE: goal_latent_dim removed in V8
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.max_seq_len = max_seq_len

        # V8: Conditioning is timestep only (no goal)
        self.time_embed_dim = latent_dim
        self.cond_embed_dim = self.time_embed_dim  # V8: no goal concatenation

        # Input embedding: 2D positions -> latent
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

        # Output projection: latent -> 2D positions
        self.output_proj = zero_module(nn.Linear(latent_dim, input_dim))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lengths: torch.Tensor
        # NOTE: goal_embed removed in V8
    ) -> torch.Tensor:
        """
        Forward pass.

        V8 Changes:
        - No goal_embed parameter
        - cond_emb is just timestep embedding

        Args:
            x: (batch, seq_len, 2) - noisy positions
            t: (batch,) - diffusion timesteps (0 to T-1)
            lengths: (batch,) - original sequence lengths before padding

        Returns:
            noise_pred: (batch, seq_len, 2) - predicted noise
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Input embedding
        h = self.input_proj(x)

        # Add positional encoding
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        # Timestep embedding
        t_emb = timestep_embedding(t, self.latent_dim)
        t_emb = self.time_embed(t_emb)

        # V8: cond_emb is just timestep (no goal concatenation)
        cond_emb = t_emb

        # Create attention mask from lengths
        src_mask = self.generate_src_mask(seq_len, lengths).to(device)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond_emb, src_mask)

        # Output projection
        out = self.output_proj(h)

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
    print("Testing TrajectoryTransformer V8...")
    print("=" * 70)

    # Test parameters
    batch_size = 4
    seq_len = 50
    input_dim = 2  # V8: 2D positions
    latent_dim = 128

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
        activation='gelu'
        # No goal_latent_dim in V8
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test inputs (V8: no goal_embed)
    x = torch.randn(batch_size, seq_len, input_dim)  # 2D positions
    t = torch.randint(0, 1000, (batch_size,))
    lengths = torch.tensor([50, 30, 45, 20])

    print(f"\nInput shapes:")
    print(f"  x: {x.shape} (2D positions)")
    print(f"  t: {t.shape}")
    print(f"  lengths: {lengths}")
    print(f"  (no goal_embed in V8)")

    # Forward pass
    print("\n=== Forward Pass ===")
    output = model(x, t, lengths)  # No goal_embed

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
        print(f"  Sequence {i}: length={length}, valid_positions={int(valid_count)}")
        assert valid_count == length, f"Mask mismatch for sequence {i}"

    print("\n" + "=" * 70)
    print("ALL V8 TESTS PASSED")
    print("=" * 70)
