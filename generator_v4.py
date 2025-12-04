"""
Generator Network (V4)
Generates fake latent sequences from noise: (seq_len, latent_dim) + condition -> (seq_len, latent_dim)

V4 Changes from V3:
- Larger hidden_dim (64 vs 32)
- Larger latent_dim (48 vs 32)
- Larger condition embedding (64 vs 50)
- TEMPORAL SELF-ATTENTION for capturing long-range dependencies

The generator operates entirely in latent space. It doesn't directly see or produce
the 8 V4 features - those are handled by recovery network.

Architecture:
    Noise (batch, seq, latent_dim) + Condition -> LSTM -> Attention -> FiLM -> Output (batch, seq, latent_dim)
"""

import torch
import torch.nn as nn
import math


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal sequences.

    Allows the model to attend to different positions in the sequence,
    capturing long-range dependencies that LSTM alone might miss.

    This is particularly important for mouse trajectories where:
    - Early movement patterns influence later motion
    - Rhythm/periodicity needs to be captured across the sequence
    - Style consistency requires global context
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads (embed_dim must be divisible by num_heads)
            dropout: Dropout rate for attention weights
        """
        super().__init__()

        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # Layer normalization for residual connection
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim) - input sequence
            mask: (batch, seq_len) - optional mask (1 = valid, 0 = padding)

        Returns:
            out: (batch, seq_len, embed_dim) - attended sequence with residual
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_scores: (batch, num_heads, seq_len, seq_len)

        # Apply mask if provided (for padding)
        if mask is not None:
            # Expand mask for attention: (batch, 1, 1, seq_len)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        attended = torch.matmul(attn_weights, v)
        # attended: (batch, num_heads, seq_len, head_dim)

        # Reshape back
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.out_proj(attended)

        # Residual connection with layer norm
        out = self.norm(x + out)

        return out


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Applies learned scale (gamma) and shift (beta) based on condition.
    """

    def __init__(self, condition_dim, feature_dim):
        super().__init__()

        # Condition to gamma (scale) and beta (shift)
        self.gamma_fc = nn.utils.spectral_norm(
            nn.Linear(condition_dim, feature_dim)
        )
        self.beta_fc = nn.utils.spectral_norm(
            nn.Linear(condition_dim, feature_dim)
        )

    def forward(self, x, condition):
        """
        Args:
            x: (batch, seq_len, feature_dim) - features to modulate
            condition: (batch, condition_dim) - conditioning variable

        Returns:
            x_modulated: (batch, seq_len, feature_dim)
        """
        # Compute scale and shift
        gamma = self.gamma_fc(condition)  # (batch, feature_dim)
        beta = self.beta_fc(condition)    # (batch, feature_dim)

        # Add dimension for broadcasting over sequence
        gamma = gamma.unsqueeze(1)  # (batch, 1, feature_dim)
        beta = beta.unsqueeze(1)    # (batch, 1, feature_dim)

        # Apply FiLM: y = gamma * x + beta
        return gamma * x + beta


class Generator(nn.Module):
    """
    Generator network with FiLM conditioning and Temporal Self-Attention (V4).
    Generates latent sequences from noise, conditioned on distance.

    Architecture: Noise -> LSTM -> Attention -> FiLM -> Output

    Operates in latent space (latent_dim), NOT feature space (8 features).
    The recovery network converts latent -> features.
    """

    def __init__(self, latent_dim=48, hidden_dim=64, num_layers=3, condition_dim=1,
                 condition_embed_dim=64, dropout=0.2, use_attention=True, num_attention_heads=4):
        """
        Args:
            latent_dim: Latent representation dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            condition_dim: Conditioning input dimension (distance)
            condition_embed_dim: Dimension for condition embedding
            dropout: Dropout rate for regularization (default: 0.2)
            use_attention: Whether to use temporal self-attention (default: True)
            num_attention_heads: Number of attention heads (default: 4)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Condition embedding with dropout
        self.condition_embed = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(condition_dim, condition_embed_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Linear(condition_embed_dim, condition_embed_dim)),
            nn.LeakyReLU(0.2)
        )

        # LSTM layers (with inter-layer dropout if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Temporal self-attention for long-range dependencies (optional)
        if use_attention:
            self.attention = TemporalSelfAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

        # FiLM layer for conditioning
        self.film = FiLMLayer(condition_embed_dim, hidden_dim)

        # Dropout after FiLM modulation (regularization before output)
        self.dropout = nn.Dropout(dropout)

        # Output projection with spectral normalization
        self.linear = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, latent_dim)
        )

        # Tanh activation for [-1, 1] outputs
        self.activation = nn.Tanh()

    def forward(self, z, condition, lengths):
        """
        Args:
            z: (batch, seq_len, latent_dim) - noise input
            condition: (batch, 1) - distance condition in [-1, 1]
            lengths: (batch,) - sequence lengths (CPU tensor)

        Returns:
            h_fake: (batch, seq_len, latent_dim) - generated latent sequences
        """
        batch_size, seq_len, _ = z.shape

        # Embed condition
        cond_embed = self.condition_embed(condition)  # (batch, condition_embed_dim)

        # Pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        lstm_out, _ = self.lstm(packed)

        # Unpack
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        # Apply temporal self-attention if enabled
        if self.use_attention:
            # Create attention mask from lengths (1 = valid, 0 = padding)
            positions = torch.arange(seq_len, device=z.device).unsqueeze(0)
            lengths_expanded = lengths.unsqueeze(1).to(z.device)
            attn_mask = (positions < lengths_expanded).float()  # (batch, seq_len)

            unpacked = self.attention(unpacked, mask=attn_mask)

        # Apply FiLM conditioning
        modulated = self.film(unpacked, cond_embed)

        # Apply dropout after FiLM modulation for regularization
        modulated = self.dropout(modulated)

        # Project to latent space
        h_fake = self.linear(modulated)
        h_fake = self.activation(h_fake)

        return h_fake


# Test code
if __name__ == '__main__':
    print("Testing Generator V4 with Temporal Self-Attention...")
    print("=" * 60)

    # Create dummy input
    batch_size = 4
    seq_len = 200
    latent_dim = 48

    z = torch.randn(batch_size, seq_len, latent_dim) * 2 - 1
    condition = torch.rand(batch_size, 1) * 2 - 1
    lengths = torch.randint(100, seq_len, (batch_size,))

    # Test Generator WITH attention (default)
    print("\n1. Testing Generator WITH attention...")
    generator_attn = Generator(
        latent_dim=48,
        hidden_dim=64,
        num_layers=3,
        condition_dim=1,
        condition_embed_dim=64,
        dropout=0.2,
        use_attention=True,
        num_attention_heads=4
    )

    total_params_attn = sum(p.numel() for p in generator_attn.parameters())
    print(f"   Total parameters (with attention): {total_params_attn:,}")

    h_fake_attn = generator_attn(z, condition, lengths)
    print(f"   Input shape: {z.shape}")
    print(f"   Output shape: {h_fake_attn.shape}")
    print(f"   Output range: [{h_fake_attn.min():.4f}, {h_fake_attn.max():.4f}]")

    # Test Generator WITHOUT attention (fallback)
    print("\n2. Testing Generator WITHOUT attention (fallback)...")
    generator_no_attn = Generator(
        latent_dim=48,
        hidden_dim=64,
        num_layers=3,
        condition_dim=1,
        condition_embed_dim=64,
        dropout=0.2,
        use_attention=False
    )

    total_params_no_attn = sum(p.numel() for p in generator_no_attn.parameters())
    print(f"   Total parameters (without attention): {total_params_no_attn:,}")
    print(f"   Attention adds {total_params_attn - total_params_no_attn:,} parameters")

    h_fake_no_attn = generator_no_attn(z, condition, lengths)
    print(f"   Output shape: {h_fake_no_attn.shape}")

    # Test attention module directly
    print("\n3. Testing TemporalSelfAttention module...")
    attention = TemporalSelfAttention(embed_dim=64, num_heads=4, dropout=0.1)
    x_test = torch.randn(batch_size, seq_len, 64)
    mask_test = torch.ones(batch_size, seq_len)
    mask_test[:, 150:] = 0  # Simulate padding

    attn_out = attention(x_test, mask=mask_test)
    print(f"   Input: {x_test.shape} -> Output: {attn_out.shape}")
    print(f"   Attention params: {sum(p.numel() for p in attention.parameters()):,}")

    # Verify outputs
    assert h_fake_attn.min() >= -1.001 and h_fake_attn.max() <= 1.001
    assert h_fake_attn.shape == (batch_size, seq_len, latent_dim)

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    generator_attn.zero_grad()
    loss = h_fake_attn.mean()
    loss.backward()

    grad_count = sum(1 for p in generator_attn.parameters() if p.grad is not None)
    total_count = sum(1 for p in generator_attn.parameters())
    print(f"   Gradients computed for {grad_count}/{total_count} parameters")

    print("\n" + "=" * 60)
    print("Generator V4 with Attention test PASSED!")
    print("=" * 60)
