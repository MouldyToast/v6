"""
Embedder Network (V4)
Maps real trajectories to latent space: (seq_len, feature_dim) -> (seq_len, latent_dim)

V4 Changes from V3:
- feature_dim=8 (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
- Larger hidden_dim (64 vs 32) for increased feature complexity
- Larger latent_dim (48 vs 32)

Feature Index Reference:
    [0] dx       - Relative x from start
    [1] dy       - Relative y from start
    [2] speed    - Movement speed (px/s)
    [3] accel    - Acceleration (px/s^2)
    [4] sin_h    - Sine of heading
    [5] cos_h    - Cosine of heading
    [6] ang_vel  - Angular velocity (rad/s)
    [7] dt       - Time delta (s)

Architecture:
    Input (batch, seq, 8) -> LSTM -> Linear -> Tanh -> Output (batch, seq, latent_dim)
"""

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
    Embedder with spectral normalization and tanh activation (V4).
    Maps (seq_len, feature_dim) -> (seq_len, latent_dim)

    V4: feature_dim=8 (position-independent style features)
    """

    def __init__(self, feature_dim=8, hidden_dim=64, latent_dim=48, num_layers=3):
        """
        Args:
            feature_dim: Input dimension (V4: 8 features)
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent representation dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Linear projection with spectral normalization
        self.linear = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, latent_dim)
        )

        # Tanh activation for [-1, 1] outputs
        self.activation = nn.Tanh()

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, seq_len, feature_dim) - padded sequences in [-1,1]
               V4: feature_dim=8 (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
            lengths: (batch,) - original lengths before padding (CPU tensor)

        Returns:
            h: (batch, seq_len, latent_dim) - embedded representation in [-1,1]
        """
        # Pack sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        lstm_out, _ = self.lstm(packed)

        # Unpack
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x.size(1)
        )

        # Map to latent space
        h = self.linear(unpacked)
        h = self.activation(h)  # Tanh: [-1, 1]

        return h


# Test code
if __name__ == '__main__':
    print("Testing Embedder V4...")
    print("=" * 50)

    # Create model with V4 config
    embedder = Embedder(
        feature_dim=8,    # V4: 8 features
        hidden_dim=64,    # Increased from V3
        latent_dim=48,    # Increased from V3
        num_layers=3
    )

    # Print model info
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dummy input
    batch_size = 4
    seq_len = 200
    feature_dim = 8

    x = torch.randn(batch_size, seq_len, feature_dim)
    x = torch.clamp(x, -1, 1)  # Simulate normalized data

    lengths = torch.randint(100, seq_len, (batch_size,))

    # Forward pass
    h = embedder(x, lengths)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {h.shape}")
    print(f"Output range: [{h.min():.4f}, {h.max():.4f}]")

    # Verify output in [-1, 1]
    assert h.min() >= -1.001 and h.max() <= 1.001, "Output not in [-1, 1] range"
    assert x.shape[2] == 8, f"Expected feature_dim=8, got {x.shape[2]}"
    assert h.shape[2] == 48, f"Expected latent_dim=48, got {h.shape[2]}"

    print("\nEmbedder V4 test PASSED!")
    print("=" * 50)
