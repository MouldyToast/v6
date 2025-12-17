"""
Recovery Network (V4)
Maps latent space back to feature space: (seq_len, latent_dim) -> (seq_len, feature_dim)

V4 Changes from V3:
- feature_dim=8 (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
- Larger hidden_dim (64 vs 32) for increased feature complexity
- Larger latent_dim (48 vs 32)

Feature Index Reference (Output):
    [0] dx       - Relative x from start
    [1] dy       - Relative y from start
    [2] speed    - Movement speed (px/s)
    [3] accel    - Acceleration (px/s^2)
    [4] sin_h    - Sine of heading
    [5] cos_h    - Cosine of heading
    [6] ang_vel  - Angular velocity (rad/s)
    [7] dt       - Time delta (s)

Architecture:
    Input (batch, seq, latent_dim) -> LSTM -> Linear -> Tanh -> Output (batch, seq, 8)
"""

import torch
import torch.nn as nn


class Recovery(nn.Module):
    """
    Recovery network (decoder) with spectral normalization and tanh activation (V4).
    Maps (seq_len, latent_dim) -> (seq_len, feature_dim)

    V4: Outputs 8 features (position-independent style features)
    """

    def __init__(self, latent_dim=48, hidden_dim=64, feature_dim=8, num_layers=3):
        """
        Args:
            latent_dim: Latent representation dimension (input)
            hidden_dim: LSTM hidden dimension
            feature_dim: Output dimension (V4: 8 features)
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Linear projection with spectral normalization
        self.linear = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, feature_dim)
        )

        # Tanh activation for [-1, 1] outputs
        self.activation = nn.Tanh()

    def forward(self, h, lengths):
        """
        Args:
            h: (batch, seq_len, latent_dim) - latent sequences in [-1,1]
            lengths: (batch,) - original lengths before padding (CPU tensor)

        Returns:
            x: (batch, seq_len, feature_dim) - recovered features in [-1,1]
               V4: 8 features (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
        """
        # Pack sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        lstm_out, _ = self.lstm(packed)

        # Unpack
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=h.size(1)
        )

        # Map to feature space
        x = self.linear(unpacked)
        x = self.activation(x)  # Tanh: [-1, 1]

        return x


# Test code
if __name__ == '__main__':
    print("Testing Recovery V4...")
    print("=" * 50)

    # Create model with V4 config
    recovery = Recovery(
        latent_dim=48,    # V4 latent dim
        hidden_dim=64,    # Increased from V3
        feature_dim=8,    # V4: 8 features
        num_layers=3
    )

    # Print model info
    total_params = sum(p.numel() for p in recovery.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dummy input (latent representation)
    batch_size = 4
    seq_len = 200
    latent_dim = 48

    h = torch.randn(batch_size, seq_len, latent_dim)
    h = torch.clamp(h, -1, 1)  # Simulate embedder output

    lengths = torch.randint(100, seq_len, (batch_size,))

    # Forward pass
    x = recovery(h, lengths)

    print(f"\nInput shape: {h.shape}")
    print(f"Output shape: {x.shape}")
    print(f"Output range: [{x.min():.4f}, {x.max():.4f}]")

    # Verify output
    assert x.min() >= -1.001 and x.max() <= 1.001, "Output not in [-1, 1] range"
    assert h.shape[2] == 48, f"Expected latent_dim=48, got {h.shape[2]}"
    assert x.shape[2] == 8, f"Expected feature_dim=8, got {x.shape[2]}"

    print("\nRecovery V4 test PASSED!")
    print("=" * 50)

    # Test embedder + recovery roundtrip
    print("\nTesting Embedder + Recovery roundtrip...")

    from embedder_v4 import Embedder

    embedder = Embedder(feature_dim=8, hidden_dim=64, latent_dim=48, num_layers=3)

    # Create input features
    x_input = torch.randn(batch_size, seq_len, 8)
    x_input = torch.clamp(x_input, -1, 1)

    # Embed
    h = embedder(x_input, lengths)
    print(f"Embedded shape: {h.shape}")

    # Recover
    x_recovered = recovery(h, lengths)
    print(f"Recovered shape: {x_recovered.shape}")

    # Check reconstruction is reasonable (won't be perfect without training)
    print(f"Reconstruction MSE: {((x_input - x_recovered) ** 2).mean():.4f}")
    print("(MSE will be high without training - this is expected)")

    print("\nRoundtrip test PASSED!")
    print("=" * 50)
