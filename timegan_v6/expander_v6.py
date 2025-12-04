"""
Latent Expander for TimeGAN V6

Expands a latent summary vector back to a sequence of latent vectors.
This is the inverse operation of the Pooler.

The expander takes a trajectory summary (from the generator in Stage 2)
and produces a sequence that can be fed to the frozen decoder to
generate actual trajectory features.

Expansion Methods:
- 'lstm': Autoregressive LSTM generation (default, most expressive)
- 'mlp': Per-timestep MLP (faster, less temporal coherence)
- 'repeat': Simple repeat + linear (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentExpander(nn.Module):
    """
    Expand latent summary back to sequence of latent vectors.

    The generator produces z_summary (a single vector capturing trajectory
    characteristics). This module expands it back to h_seq (a sequence of
    latent vectors) that can be fed to the frozen decoder.

    Key design decisions:
    1. LSTM-based: Captures temporal dependencies in the output
    2. Summary conditions initial state: z_summary → (h0, c0)
    3. Autoregressive: Each output depends on previous output
    4. Tanh output: Bounds values to [-1, 1] like encoder output
    """

    def __init__(self, summary_dim: int, latent_dim: int,
                 hidden_dim: int = 128, num_layers: int = 2,
                 expand_type: str = 'lstm', dropout: float = 0.1):
        """
        Args:
            summary_dim: Dimension of input summary vector (from pooler/generator)
            latent_dim: Dimension of output latent vectors (to match encoder)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            expand_type: Expansion method - 'lstm', 'mlp', or 'repeat'
            dropout: Dropout rate
        """
        super().__init__()

        self.summary_dim = summary_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.expand_type = expand_type

        if expand_type == 'lstm':
            self._build_lstm_expander(dropout)
        elif expand_type == 'mlp':
            self._build_mlp_expander(dropout)
        elif expand_type == 'repeat':
            self._build_repeat_expander()
        else:
            raise ValueError(f"Unknown expand_type: {expand_type}. "
                             f"Choose from 'lstm', 'mlp', 'repeat'")

    def _build_lstm_expander(self, dropout: float):
        """Build LSTM-based autoregressive expander."""
        # Transform summary to initial LSTM hidden state
        # Each LSTM layer needs its own h0 and c0
        self.init_hidden = nn.Linear(self.summary_dim, self.hidden_dim * self.num_layers)
        self.init_cell = nn.Linear(self.summary_dim, self.hidden_dim * self.num_layers)

        # LSTM for autoregressive generation
        # Input: previous output (latent_dim) + summary (summary_dim) for conditioning
        self.lstm = nn.LSTM(
            input_size=self.latent_dim + self.summary_dim,  # Previous output + summary
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh()  # Bound to [-1, 1]
        )

    def _build_mlp_expander(self, dropout: float):
        """Build MLP-based per-timestep expander."""
        # Positional encoding dimension
        self.pos_dim = 32

        # MLP that takes summary + positional encoding
        self.mlp = nn.Sequential(
            nn.Linear(self.summary_dim + self.pos_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh()
        )

    def _build_repeat_expander(self):
        """Build simple repeat + transform expander."""
        # Transform repeated summary to latent dimension
        self.transform = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh()
        )

    def forward(self, z_summary: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Expand summary vector to sequence of latent vectors.

        Args:
            z_summary: (batch, summary_dim) - trajectory summary
            seq_len: int - desired output sequence length

        Returns:
            h_seq: (batch, seq_len, latent_dim) - sequence of latent vectors
        """
        if self.expand_type == 'lstm':
            return self._lstm_expand(z_summary, seq_len)
        elif self.expand_type == 'mlp':
            return self._mlp_expand(z_summary, seq_len)
        elif self.expand_type == 'repeat':
            return self._repeat_expand(z_summary, seq_len)

    def _lstm_expand(self, z_summary: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Autoregressive LSTM expansion.

        Uses the summary to initialize LSTM state, then generates
        the sequence one timestep at a time.
        """
        batch_size = z_summary.size(0)
        device = z_summary.device

        # Initialize LSTM state from summary
        # init_hidden output: (batch, hidden_dim * num_layers)
        # Reshape to: (num_layers, batch, hidden_dim)
        h0 = self.init_hidden(z_summary)
        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_dim)

        c0 = self.init_cell(z_summary)
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()

        # Start token: zeros
        h_prev = torch.zeros(batch_size, self.latent_dim, device=device)

        outputs = []
        hidden = (h0, c0)

        # Autoregressive generation
        for t in range(seq_len):
            # Concatenate previous output with summary for conditioning
            lstm_input = torch.cat([h_prev, z_summary], dim=-1)  # (batch, latent_dim + summary_dim)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, input_dim)

            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: (batch, 1, hidden_dim)

            # Project to latent space
            h_t = self.output(lstm_out.squeeze(1))  # (batch, latent_dim)

            outputs.append(h_t)
            h_prev = h_t

        # Stack outputs: list of (batch, latent_dim) -> (batch, seq_len, latent_dim)
        h_seq = torch.stack(outputs, dim=1)

        return h_seq

    def _mlp_expand(self, z_summary: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        MLP-based expansion with positional encoding.

        Generates each timestep independently using the summary
        and a positional encoding.
        """
        batch_size = z_summary.size(0)
        device = z_summary.device

        # Create positional encodings
        positions = torch.arange(seq_len, device=device).float() / seq_len
        pos_encoding = self._get_positional_encoding(positions)  # (seq_len, pos_dim)

        # Expand summary and positional encoding for batched processing
        # summary: (batch, 1, summary_dim) -> repeat -> (batch, seq_len, summary_dim)
        z_expanded = z_summary.unsqueeze(1).expand(-1, seq_len, -1)

        # pos_encoding: (1, seq_len, pos_dim) -> repeat -> (batch, seq_len, pos_dim)
        pos_expanded = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate and process
        mlp_input = torch.cat([z_expanded, pos_expanded], dim=-1)  # (batch, seq_len, summary_dim + pos_dim)

        # Reshape for batch processing
        mlp_input = mlp_input.view(batch_size * seq_len, -1)
        h_flat = self.mlp(mlp_input)
        h_seq = h_flat.view(batch_size, seq_len, self.latent_dim)

        return h_seq

    def _get_positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings.

        Args:
            positions: (seq_len,) - normalized positions [0, 1]

        Returns:
            encoding: (seq_len, pos_dim) - positional encodings
        """
        seq_len = positions.size(0)
        device = positions.device

        # Create frequency bands
        freqs = torch.arange(self.pos_dim // 2, device=device).float()
        freqs = 10000 ** (-2 * freqs / self.pos_dim)

        # Compute sin and cos encodings
        positions = positions.unsqueeze(1)  # (seq_len, 1)
        freqs = freqs.unsqueeze(0)  # (1, pos_dim//2)

        angles = positions * freqs * 2 * 3.14159  # (seq_len, pos_dim//2)

        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return encoding

    def _repeat_expand(self, z_summary: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Simple repeat and transform expansion.

        The same transformation is applied at each timestep.
        This is a baseline - less expressive but very simple.
        """
        batch_size = z_summary.size(0)

        # Transform summary to latent dimension
        h_single = self.transform(z_summary)  # (batch, latent_dim)

        # Repeat for each timestep
        h_seq = h_single.unsqueeze(1).expand(-1, seq_len, -1)

        # Make contiguous for downstream operations
        h_seq = h_seq.contiguous()

        return h_seq

    def expand_with_noise(self, z_summary: torch.Tensor, seq_len: int,
                          noise_scale: float = 0.1) -> torch.Tensor:
        """
        Expand with added noise for diversity.

        Useful during training to encourage robustness.

        Args:
            z_summary: (batch, summary_dim)
            seq_len: int
            noise_scale: Standard deviation of Gaussian noise

        Returns:
            h_seq: (batch, seq_len, latent_dim) - noisy expansion
        """
        h_seq = self.forward(z_summary, seq_len)

        if noise_scale > 0 and self.training:
            noise = torch.randn_like(h_seq) * noise_scale
            h_seq = h_seq + noise
            # Re-apply tanh to ensure bounds
            h_seq = torch.tanh(h_seq)

        return h_seq


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing LatentExpander V6...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test parameters
    batch_size = 4
    summary_dim = 96
    latent_dim = 48
    seq_len = 50

    # Create test input
    z_summary = torch.randn(batch_size, summary_dim)
    z_summary = torch.tanh(z_summary)  # Simulate pooler output in [-1, 1]

    # Test each expand type
    expand_types = ['lstm', 'mlp', 'repeat']

    for expand_type in expand_types:
        print(f"\n=== Testing expand_type='{expand_type}' ===")

        expander = LatentExpander(
            summary_dim=summary_dim,
            latent_dim=latent_dim,
            hidden_dim=128,
            num_layers=2,
            expand_type=expand_type
        )

        # Count parameters
        num_params = sum(p.numel() for p in expander.parameters())
        print(f"Parameters: {num_params:,}")

        # Forward pass
        h_seq = expander(z_summary, seq_len)

        print(f"Input shape: {z_summary.shape}")
        print(f"Output shape: {h_seq.shape}")
        print(f"Output range: [{h_seq.min().item():.4f}, {h_seq.max().item():.4f}]")

        # Verify output shape
        expected_shape = (batch_size, seq_len, latent_dim)
        assert h_seq.shape == expected_shape, \
            f"Expected {expected_shape}, got {h_seq.shape}"

        # Verify output in [-1, 1] (due to Tanh)
        assert h_seq.min() >= -1.001 and h_seq.max() <= 1.001, \
            f"Output not in [-1, 1] range"

        # Test gradient flow
        loss = h_seq.sum()
        loss.backward()
        print("Gradient flow: OK")

        # Reset gradients
        expander.zero_grad()

        print(f"expand_type='{expand_type}': PASS")

    # Test different sequence lengths
    print("\n=== Testing variable sequence lengths ===")
    expander = LatentExpander(summary_dim=summary_dim, latent_dim=latent_dim)

    for seq_len in [10, 50, 100, 200]:
        h_seq = expander(z_summary, seq_len)
        assert h_seq.shape == (batch_size, seq_len, latent_dim), \
            f"Wrong shape for seq_len={seq_len}"
    print("Variable sequence lengths: PASS")

    # Test expand_with_noise
    print("\n=== Testing expand_with_noise ===")
    expander.train()  # Noise only applied in training mode

    h_seq_clean = expander(z_summary, seq_len)
    h_seq_noisy = expander.expand_with_noise(z_summary, seq_len, noise_scale=0.1)

    # Should be different due to noise
    diff = (h_seq_clean - h_seq_noisy).abs().mean()
    print(f"Mean difference with noise: {diff.item():.4f}")

    # Noisy output should still be bounded
    assert h_seq_noisy.min() >= -1.001 and h_seq_noisy.max() <= 1.001, \
        "Noisy output not in [-1, 1] range"
    print("expand_with_noise: PASS")

    # Test pooler + expander roundtrip
    print("\n=== Testing Pooler + Expander roundtrip ===")
    from timegan_v6.pooler_v6 import LatentPooler

    # Create matching pooler and expander
    pooler = LatentPooler(latent_dim=latent_dim, pool_type='attention', output_dim=summary_dim)
    expander = LatentExpander(summary_dim=summary_dim, latent_dim=latent_dim, expand_type='lstm')

    # Create random latent sequence (simulating encoder output)
    h_original = torch.randn(batch_size, seq_len, latent_dim)
    h_original = torch.tanh(h_original)
    lengths = torch.tensor([50, 30, 45, 20])

    # Pool -> Expand
    z_summary = pooler(h_original, lengths)
    h_reconstructed = expander(z_summary, seq_len)

    print(f"Original shape: {h_original.shape}")
    print(f"Summary shape: {z_summary.shape}")
    print(f"Reconstructed shape: {h_reconstructed.shape}")

    # Shapes should match
    assert h_original.shape == h_reconstructed.shape, "Shape mismatch in roundtrip"

    # Note: Values won't match exactly without training
    mse = F.mse_loss(h_original, h_reconstructed)
    print(f"Reconstruction MSE (untrained): {mse.item():.4f}")
    print("(High MSE expected - models need joint training)")

    print("\nPooler + Expander roundtrip: PASS")

    print("\n" + "=" * 60)
    print("ALL LATENT EXPANDER TESTS PASSED")
    print("=" * 60)
