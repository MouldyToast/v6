"""
Latent Generator for TimeGAN V6

MLP generator that produces latent summary vectors for WGAN-GP training.

This is fundamentally different from TimeGAN's sequence generator:
- Operates on fixed-dimension vectors, not sequences
- Much simpler architecture (MLP vs RNN)
- Faster training and more stable

The generator learns to map:
    noise (z_dim) + condition (distance) → latent summary (summary_dim)

The generated summary is then expanded by LatentExpander and decoded
by the frozen decoder to produce actual trajectory features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGenerator(nn.Module):
    """
    MLP generator that produces latent summary vectors.

    Key design decisions:
    1. Condition embedding: Transform scalar condition to learned representation
    2. Layer normalization: More stable than batch norm for WGAN-GP
    3. LeakyReLU: Standard for GAN generators
    4. Tanh output: Match the [-1, 1] range of real latent summaries
    5. Residual connections: Help with gradient flow in deeper networks
    """

    def __init__(self, noise_dim: int = 128, condition_dim: int = 1,
                 summary_dim: int = 96, hidden_dims: list = None,
                 use_residual: bool = True, dropout: float = 0.0):
        """
        Args:
            noise_dim: Dimension of input noise vector
            condition_dim: Dimension of condition (e.g., 1 for distance)
            summary_dim: Dimension of output latent summary
            hidden_dims: List of hidden layer dimensions (default: [256, 256, 256])
            use_residual: Whether to use residual connections
            dropout: Dropout rate (usually 0 for generator)
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.summary_dim = summary_dim
        self.use_residual = use_residual

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Condition embedding network
        # Transforms scalar condition to richer representation
        self.cond_embed_dim = 64
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.cond_embed_dim),
            nn.LeakyReLU(0.2)
        )

        # Main generator network
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_dim = noise_dim + self.cond_embed_dim

        for i, h_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(in_dim, h_dim))

            # Layer normalization
            self.norms.append(nn.LayerNorm(h_dim))

            # Residual projection if dimensions don't match
            if use_residual and in_dim != h_dim:
                self.residual_projs.append(nn.Linear(in_dim, h_dim))
            else:
                self.residual_projs.append(None)

            in_dim = h_dim

        # Output layer
        self.output = nn.Linear(in_dim, summary_dim)

        # Output activation
        self.output_activation = nn.Tanh()

        # Dropout (usually 0 for generator)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate latent summary from noise and condition.

        Args:
            z: (batch, noise_dim) - random noise sampled from N(0, 1)
            condition: (batch, condition_dim) - conditioning variable (e.g., distance)

        Returns:
            z_summary: (batch, summary_dim) - generated latent summary in [-1, 1]
        """
        # Embed condition
        cond_emb = self.cond_embed(condition)  # (batch, cond_embed_dim)

        # Concatenate noise and condition
        x = torch.cat([z, cond_emb], dim=-1)  # (batch, noise_dim + cond_embed_dim)

        # Process through hidden layers
        for layer, norm, residual_proj in zip(self.layers, self.norms, self.residual_projs):
            identity = x

            # Linear + Norm + Activation
            x = layer(x)
            x = norm(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Residual connection
            if self.use_residual:
                if residual_proj is not None:
                    identity = residual_proj(identity)
                x = x + identity

        # Output projection
        z_summary = self.output(x)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noise from standard normal distribution.

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            z: (batch_size, noise_dim) - noise samples
        """
        return torch.randn(batch_size, self.noise_dim, device=device)

    def generate(self, batch_size: int, condition: torch.Tensor,
                 device: torch.device = None) -> torch.Tensor:
        """
        Convenience method to generate samples.

        Args:
            batch_size: Number of samples to generate
            condition: (batch_size, condition_dim) - conditions
            device: Device (inferred from condition if not provided)

        Returns:
            z_summary: (batch_size, summary_dim) - generated summaries
        """
        if device is None:
            device = condition.device

        z = self.sample_noise(batch_size, device)
        return self.forward(z, condition)


class ConditionalLatentGenerator(LatentGenerator):
    """
    Generator with stronger condition integration.

    Uses FiLM (Feature-wise Linear Modulation) to inject condition
    information at each layer, not just the input.
    """

    def __init__(self, noise_dim: int = 128, condition_dim: int = 1,
                 summary_dim: int = 96, hidden_dims: list = None,
                 dropout: float = 0.0):
        """
        Args:
            noise_dim: Dimension of input noise vector
            condition_dim: Dimension of condition
            summary_dim: Dimension of output latent summary
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        # Don't call parent __init__ - we're building a different architecture
        nn.Module.__init__(self)

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.summary_dim = summary_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Condition embedding for FiLM
        self.cond_embed_dim = 64
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.cond_embed_dim)
        )

        # Main layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # FiLM parameters for each layer (scale and shift)
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()

        in_dim = noise_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.norms.append(nn.LayerNorm(h_dim))

            # FiLM: condition -> (gamma, beta) for this layer
            self.film_scale.append(nn.Linear(self.cond_embed_dim, h_dim))
            self.film_shift.append(nn.Linear(self.cond_embed_dim, h_dim))

            in_dim = h_dim

        # Output layer
        self.output = nn.Linear(in_dim, summary_dim)
        self.output_activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize FiLM scale to 1 and shift to 0
        for scale_layer, shift_layer in zip(self.film_scale, self.film_shift):
            nn.init.ones_(scale_layer.weight)
            nn.init.zeros_(scale_layer.bias)
            nn.init.zeros_(shift_layer.weight)
            nn.init.zeros_(shift_layer.bias)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate with FiLM conditioning.

        Args:
            z: (batch, noise_dim) - random noise
            condition: (batch, condition_dim) - conditioning variable

        Returns:
            z_summary: (batch, summary_dim) - generated latent summary
        """
        # Embed condition once
        cond_emb = self.cond_embed(condition)

        x = z

        for layer, norm, film_s, film_b in zip(
            self.layers, self.norms, self.film_scale, self.film_shift
        ):
            # Linear
            x = layer(x)

            # Layer norm
            x = norm(x)

            # FiLM modulation: gamma * x + beta
            gamma = film_s(cond_emb)  # (batch, h_dim)
            beta = film_b(cond_emb)   # (batch, h_dim)
            x = gamma * x + beta

            # Activation
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

        # Output
        z_summary = self.output(x)
        z_summary = self.output_activation(z_summary)

        return z_summary

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise from standard normal distribution."""
        return torch.randn(batch_size, self.noise_dim, device=device)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing LatentGenerator V6...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test parameters
    batch_size = 8
    noise_dim = 128
    condition_dim = 1
    summary_dim = 96

    # Test basic generator
    print("\n=== Testing LatentGenerator ===")

    generator = LatentGenerator(
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        summary_dim=summary_dim,
        hidden_dims=[256, 256, 256],
        use_residual=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Parameters: {num_params:,}")

    # Create inputs
    z = torch.randn(batch_size, noise_dim)
    condition = torch.rand(batch_size, condition_dim)  # e.g., normalized distance

    # Forward pass
    z_summary = generator(z, condition)

    print(f"Noise shape: {z.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape: {z_summary.shape}")
    print(f"Output range: [{z_summary.min().item():.4f}, {z_summary.max().item():.4f}]")

    # Verify output
    assert z_summary.shape == (batch_size, summary_dim), \
        f"Expected ({batch_size}, {summary_dim}), got {z_summary.shape}"
    assert z_summary.min() >= -1.001 and z_summary.max() <= 1.001, \
        "Output not in [-1, 1] range"

    # Test gradient flow
    loss = z_summary.sum()
    loss.backward()
    print("Gradient flow: OK")
    generator.zero_grad()

    print("LatentGenerator: PASS")

    # Test convenience methods
    print("\n=== Testing convenience methods ===")

    z_sampled = generator.sample_noise(batch_size, device)
    assert z_sampled.shape == (batch_size, noise_dim)
    print(f"sample_noise: OK (shape {z_sampled.shape})")

    z_generated = generator.generate(batch_size, condition, device)
    assert z_generated.shape == (batch_size, summary_dim)
    print(f"generate: OK (shape {z_generated.shape})")

    print("Convenience methods: PASS")

    # Test ConditionalLatentGenerator (FiLM)
    print("\n=== Testing ConditionalLatentGenerator (FiLM) ===")

    cond_generator = ConditionalLatentGenerator(
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        summary_dim=summary_dim,
        hidden_dims=[256, 256, 256]
    )

    num_params = sum(p.numel() for p in cond_generator.parameters())
    print(f"Parameters: {num_params:,}")

    z_summary_cond = cond_generator(z, condition)

    print(f"Output shape: {z_summary_cond.shape}")
    print(f"Output range: [{z_summary_cond.min().item():.4f}, {z_summary_cond.max().item():.4f}]")

    assert z_summary_cond.shape == (batch_size, summary_dim)
    assert z_summary_cond.min() >= -1.001 and z_summary_cond.max() <= 1.001

    # Test gradient flow
    loss = z_summary_cond.sum()
    loss.backward()
    print("Gradient flow: OK")

    print("ConditionalLatentGenerator: PASS")

    # Test condition influence
    print("\n=== Testing condition influence ===")

    generator.eval()
    z_fixed = torch.randn(1, noise_dim)

    outputs = []
    for cond_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cond = torch.tensor([[cond_val]])
        out = generator(z_fixed, cond)
        outputs.append(out)

    # Check that different conditions produce different outputs
    outputs = torch.cat(outputs, dim=0)
    pairwise_diff = (outputs.unsqueeze(0) - outputs.unsqueeze(1)).abs().sum(dim=-1)

    # Off-diagonal elements should be non-zero
    off_diagonal = pairwise_diff[~torch.eye(5, dtype=bool)]
    assert off_diagonal.min() > 0.01, "Condition should influence output"

    print(f"Mean pairwise difference: {off_diagonal.mean().item():.4f}")
    print("Condition influence: PASS")

    # Test without residual connections
    print("\n=== Testing without residual connections ===")

    generator_no_res = LatentGenerator(
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        summary_dim=summary_dim,
        use_residual=False
    )

    z_summary_no_res = generator_no_res(z, condition)
    assert z_summary_no_res.shape == (batch_size, summary_dim)
    print("No residual: PASS")

    # Test different architectures
    print("\n=== Testing different architectures ===")

    for hidden_dims in [[128], [256, 256], [512, 256, 128]]:
        gen = LatentGenerator(
            noise_dim=noise_dim,
            condition_dim=condition_dim,
            summary_dim=summary_dim,
            hidden_dims=hidden_dims
        )
        out = gen(z, condition)
        assert out.shape == (batch_size, summary_dim)
        num_p = sum(p.numel() for p in gen.parameters())
        print(f"  hidden_dims={hidden_dims}: {num_p:,} params - OK")

    print("Different architectures: PASS")

    print("\n" + "=" * 60)
    print("ALL LATENT GENERATOR TESTS PASSED")
    print("=" * 60)
