"""
Latent Discriminator for TimeGAN V6

MLP discriminator/critic for WGAN-GP that operates on latent summary vectors.

This is the key insight of the RTSGAN approach:
- Discriminator operates on latent summaries, NOT decoded sequences
- Generator receives direct, dense feedback about latent quality
- Much simpler than sequence-level discrimination

For WGAN-GP:
- No sigmoid activation (outputs raw Wasserstein distance estimate)
- Layer normalization (not batch norm) for stability
- Gradient penalty enforced externally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiscriminator(nn.Module):
    """
    MLP discriminator/critic for WGAN-GP in latent space.

    Key design decisions:
    1. No sigmoid: WGAN uses raw scores, not probabilities
    2. Layer normalization: Stable with gradient penalty
    3. LeakyReLU: Standard for discriminators
    4. Condition embedding: Discriminate conditioned on trajectory distance
    5. Spectral normalization (optional): Additional stability
    """

    def __init__(self, summary_dim: int = 96, condition_dim: int = 1,
                 hidden_dims: list = None, use_spectral_norm: bool = False,
                 dropout: float = 0.0):
        """
        Args:
            summary_dim: Dimension of input latent summary
            condition_dim: Dimension of condition (e.g., 1 for distance)
            hidden_dims: List of hidden layer dimensions (default: [256, 256, 256])
            use_spectral_norm: Apply spectral normalization to layers
            dropout: Dropout rate
        """
        super().__init__()

        self.summary_dim = summary_dim
        self.condition_dim = condition_dim
        self.use_spectral_norm = use_spectral_norm

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Condition embedding
        self.cond_embed_dim = 64
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.cond_embed_dim),
            nn.LeakyReLU(0.2)
        )

        # Main discriminator network
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = summary_dim + self.cond_embed_dim

        for h_dim in hidden_dims:
            # Linear layer (optionally with spectral norm)
            linear = nn.Linear(in_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            self.layers.append(linear)

            # Layer normalization (works well with WGAN-GP)
            self.norms.append(nn.LayerNorm(h_dim))

            in_dim = h_dim

        # Output layer - single score (no sigmoid for WGAN)
        output_linear = nn.Linear(in_dim, 1)
        if use_spectral_norm:
            output_linear = nn.utils.spectral_norm(output_linear)
        self.output = output_linear

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip if spectral norm is applied (has weight_orig)
                if hasattr(module, 'weight_orig'):
                    nn.init.xavier_uniform_(module.weight_orig)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z_summary: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Compute Wasserstein distance estimate for latent summary.

        Args:
            z_summary: (batch, summary_dim) - latent summary (real or fake)
            condition: (batch, condition_dim) - conditioning variable

        Returns:
            score: (batch, 1) - Wasserstein distance estimate (higher = more real)
        """
        # Embed condition
        cond_emb = self.cond_embed(condition)  # (batch, cond_embed_dim)

        # Concatenate summary and condition
        x = torch.cat([z_summary, cond_emb], dim=-1)

        # Process through hidden layers
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

        # Output score (no activation - raw Wasserstein estimate)
        score = self.output(x)

        return score

    def get_features(self, z_summary: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features for feature matching loss.

        Returns features from the penultimate layer.

        Args:
            z_summary: (batch, summary_dim)
            condition: (batch, condition_dim)

        Returns:
            features: (batch, last_hidden_dim)
        """
        cond_emb = self.cond_embed(condition)
        x = torch.cat([z_summary, cond_emb], dim=-1)

        # Process through all but last hidden layer
        for layer, norm in zip(self.layers[:-1], self.norms[:-1]):
            x = layer(x)
            x = norm(x)
            x = F.leaky_relu(x, 0.2)

        # Last hidden layer without further processing
        x = self.layers[-1](x)
        x = self.norms[-1](x)

        return x


class ConditionalLatentDiscriminator(nn.Module):
    """
    Discriminator with projection-based conditioning.

    Uses the projection discriminator approach from cGAN literature,
    which has been shown to be more effective for conditional generation.
    """

    def __init__(self, summary_dim: int = 96, condition_dim: int = 1,
                 hidden_dims: list = None, use_spectral_norm: bool = False):
        """
        Args:
            summary_dim: Dimension of input latent summary
            condition_dim: Dimension of condition
            hidden_dims: List of hidden layer dimensions
            use_spectral_norm: Apply spectral normalization
        """
        super().__init__()

        self.summary_dim = summary_dim
        self.condition_dim = condition_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Condition embedding to match final hidden dim
        self.cond_embed_dim = hidden_dims[-1]
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.cond_embed_dim)
        )

        # Main discriminator (processes summary only)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = summary_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            self.layers.append(linear)
            self.norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim

        # Output layer (unconditional part)
        self.output = nn.Linear(in_dim, 1)
        if use_spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_orig'):
                    nn.init.xavier_uniform_(module.weight_orig)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z_summary: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Compute score using projection discriminator.

        Score = f(z_summary) + <embed(condition), phi(z_summary)>

        Args:
            z_summary: (batch, summary_dim)
            condition: (batch, condition_dim)

        Returns:
            score: (batch, 1)
        """
        x = z_summary

        # Process through hidden layers
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.leaky_relu(x, 0.2)

        # Unconditional output
        out_uncond = self.output(x)  # (batch, 1)

        # Conditional projection: inner product of condition embedding and features
        cond_emb = self.cond_embed(condition)  # (batch, hidden_dim)
        out_cond = (x * cond_emb).sum(dim=-1, keepdim=True)  # (batch, 1)

        # Combined score
        score = out_uncond + out_cond

        return score


class MultiScaleLatentDiscriminator(nn.Module):
    """
    Multi-scale discriminator that looks at the latent summary at different scales.

    Useful for capturing both fine-grained and global patterns in the latent space.
    """

    def __init__(self, summary_dim: int = 96, condition_dim: int = 1,
                 n_scales: int = 3, hidden_dim: int = 128):
        """
        Args:
            summary_dim: Dimension of input latent summary
            condition_dim: Dimension of condition
            n_scales: Number of scales (discriminators)
            hidden_dim: Hidden dimension for each discriminator
        """
        super().__init__()

        self.n_scales = n_scales

        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        self.downscalers = nn.ModuleList()

        current_dim = summary_dim

        for i in range(n_scales):
            # Discriminator for this scale
            disc = LatentDiscriminator(
                summary_dim=current_dim,
                condition_dim=condition_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                use_spectral_norm=True
            )
            self.discriminators.append(disc)

            # Downscaler for next scale (except last)
            if i < n_scales - 1:
                next_dim = current_dim // 2
                downscale = nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.LeakyReLU(0.2)
                )
                self.downscalers.append(downscale)
                current_dim = next_dim

    def forward(self, z_summary: torch.Tensor,
                condition: torch.Tensor) -> list:
        """
        Compute scores at multiple scales.

        Args:
            z_summary: (batch, summary_dim)
            condition: (batch, condition_dim)

        Returns:
            scores: List of (batch, 1) tensors, one per scale
        """
        scores = []
        x = z_summary

        for i, disc in enumerate(self.discriminators):
            # Discriminate at this scale
            score = disc(x, condition)
            scores.append(score)

            # Downscale for next iteration (except last)
            if i < self.n_scales - 1:
                x = self.downscalers[i](x)

        return scores

    def forward_single(self, z_summary: torch.Tensor,
                       condition: torch.Tensor) -> torch.Tensor:
        """
        Compute single aggregated score (mean of all scales).

        Args:
            z_summary: (batch, summary_dim)
            condition: (batch, condition_dim)

        Returns:
            score: (batch, 1) - mean score across scales
        """
        scores = self.forward(z_summary, condition)
        return torch.stack(scores, dim=0).mean(dim=0)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing LatentDiscriminator V6...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test parameters
    batch_size = 8
    summary_dim = 96
    condition_dim = 1

    # Create test inputs
    z_summary = torch.randn(batch_size, summary_dim)
    z_summary = torch.tanh(z_summary)  # Simulate real latent summaries
    condition = torch.rand(batch_size, condition_dim)

    # Test basic discriminator
    print("\n=== Testing LatentDiscriminator ===")

    discriminator = LatentDiscriminator(
        summary_dim=summary_dim,
        condition_dim=condition_dim,
        hidden_dims=[256, 256, 256],
        use_spectral_norm=False
    )

    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Parameters: {num_params:,}")

    # Forward pass
    score = discriminator(z_summary, condition)

    print(f"Input shape: {z_summary.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape: {score.shape}")
    print(f"Output range: [{score.min().item():.4f}, {score.max().item():.4f}]")

    # Verify output shape
    assert score.shape == (batch_size, 1), \
        f"Expected ({batch_size}, 1), got {score.shape}"

    # Test gradient flow
    loss = score.sum()
    loss.backward()
    print("Gradient flow: OK")
    discriminator.zero_grad()

    print("LatentDiscriminator: PASS")

    # Test with spectral normalization
    print("\n=== Testing with spectral normalization ===")

    disc_sn = LatentDiscriminator(
        summary_dim=summary_dim,
        condition_dim=condition_dim,
        use_spectral_norm=True
    )

    score_sn = disc_sn(z_summary, condition)
    assert score_sn.shape == (batch_size, 1)

    loss = score_sn.sum()
    loss.backward()
    print("Spectral norm: PASS")

    # Test feature extraction
    print("\n=== Testing feature extraction ===")

    features = discriminator.get_features(z_summary, condition)
    print(f"Features shape: {features.shape}")
    assert features.shape[0] == batch_size
    print("Feature extraction: PASS")

    # Test ConditionalLatentDiscriminator (projection)
    print("\n=== Testing ConditionalLatentDiscriminator ===")

    cond_disc = ConditionalLatentDiscriminator(
        summary_dim=summary_dim,
        condition_dim=condition_dim,
        hidden_dims=[256, 256, 256]
    )

    num_params = sum(p.numel() for p in cond_disc.parameters())
    print(f"Parameters: {num_params:,}")

    score_cond = cond_disc(z_summary, condition)
    print(f"Output shape: {score_cond.shape}")
    assert score_cond.shape == (batch_size, 1)

    loss = score_cond.sum()
    loss.backward()
    print("Gradient flow: OK")

    print("ConditionalLatentDiscriminator: PASS")

    # Test MultiScaleLatentDiscriminator
    print("\n=== Testing MultiScaleLatentDiscriminator ===")

    ms_disc = MultiScaleLatentDiscriminator(
        summary_dim=summary_dim,
        condition_dim=condition_dim,
        n_scales=3,
        hidden_dim=128
    )

    num_params = sum(p.numel() for p in ms_disc.parameters())
    print(f"Parameters: {num_params:,}")

    scores_ms = ms_disc(z_summary, condition)
    print(f"Number of scales: {len(scores_ms)}")
    for i, s in enumerate(scores_ms):
        print(f"  Scale {i} score shape: {s.shape}")
        assert s.shape == (batch_size, 1)

    # Test single output
    score_single = ms_disc.forward_single(z_summary, condition)
    print(f"Aggregated score shape: {score_single.shape}")
    assert score_single.shape == (batch_size, 1)

    # Test gradient flow
    loss = sum(s.sum() for s in scores_ms)
    loss.backward()
    print("Gradient flow: OK")

    print("MultiScaleLatentDiscriminator: PASS")

    # Test WGAN loss computation
    print("\n=== Testing WGAN loss computation ===")

    discriminator.zero_grad()

    # Real and fake samples
    z_real = torch.randn(batch_size, summary_dim)
    z_fake = torch.randn(batch_size, summary_dim)

    d_real = discriminator(z_real, condition)
    d_fake = discriminator(z_fake, condition)

    # WGAN discriminator loss: E[D(fake)] - E[D(real)]
    loss_d = d_fake.mean() - d_real.mean()
    print(f"D(real) mean: {d_real.mean().item():.4f}")
    print(f"D(fake) mean: {d_fake.mean().item():.4f}")
    print(f"WGAN D loss: {loss_d.item():.4f}")

    loss_d.backward()
    print("WGAN loss backward: OK")

    print("WGAN loss computation: PASS")

    # Test gradient penalty
    print("\n=== Testing gradient penalty integration ===")

    from timegan_v6.utils_v6 import compute_gradient_penalty_latent

    discriminator.zero_grad()

    gp = compute_gradient_penalty_latent(
        discriminator, z_real, z_fake, condition
    )
    print(f"Gradient penalty: {gp.item():.4f}")

    # Full WGAN-GP loss
    lambda_gp = 10.0
    loss_d_gp = d_fake.mean() - d_real.mean() + lambda_gp * gp
    print(f"WGAN-GP D loss: {loss_d_gp.item():.4f}")

    print("Gradient penalty integration: PASS")

    print("\n" + "=" * 60)
    print("ALL LATENT DISCRIMINATOR TESTS PASSED")
    print("=" * 60)
