"""
TimeGAN v6 Model - RTSGAN-Style Two-Stage Training

Main model wrapper that composes all components:
- Encoder (from v6): Maps trajectories to latent sequences
- Decoder (from v6): Maps latent sequences to trajectories
- Pooler: Compresses latent sequences to summary vectors
- Expander: Expands summary vectors to latent sequences
- Generator: Produces latent summaries from noise + condition
- Discriminator: WGAN-GP critic in latent space

The model supports three modes:
1. Autoencoder mode (Stage 1): Train encoder + pooler + expander + decoder
2. GAN mode (Stage 2): Train generator + discriminator in frozen latent space
3. Generation mode: Generate new trajectories

Usage:
    from timegan_v6.model_v6 import TimeGANv6
    from timegan_v6.config_model_v6 import TimeGANv6Config

    config = TimeGANv6Config()
    model = TimeGANv6(config)

    # Stage 1: Autoencoder training
    x_recon, losses = model.forward_autoencoder(x_real, lengths)

    # Stage 2: GAN training (after freezing autoencoder)
    model.freeze_autoencoder()
    d_loss = model.forward_discriminator(x_real, condition, lengths)
    g_loss = model.forward_generator(batch_size, condition)

    # Generation
    x_fake = model.generate(n_samples, conditions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for v6 imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pooler_v6 import LatentPooler
from .expander_v6 import LatentExpander
from .latent_generator_v6 import LatentGenerator, ConditionalLatentGenerator
from .latent_discriminator_v6 import (
    LatentDiscriminator,
    ConditionalLatentDiscriminator,
    MultiScaleLatentDiscriminator
)
from .utils_v6 import masked_mse_loss, compute_gradient_penalty_latent
from .config_model_v6 import TimeGANv6Config

# Import v6 encoder/decoder
from embedder_v6 import Embedder
from recovery_v6 import Recovery


class TimeGANv6(nn.Module):
    """
    TimeGAN v6: RTSGAN-Style Two-Stage Training Model.

    Architecture:
        Stage 1 (Autoencoder):
            x_real → Encoder → h_seq → Pooler → z_summary → Expander → h_seq_recon → Decoder → x_recon

        Stage 2 (WGAN-GP):
            Real: x_real → Encoder* → h_seq → Pooler* → z_real
            Fake: noise + condition → Generator → z_fake
            Discriminator(z_real, condition) vs Discriminator(z_fake, condition)
            (* = frozen)

        Generation:
            noise + condition → Generator → z_summary → Expander* → h_seq → Decoder* → x_fake
    """

    def __init__(self, config: TimeGANv6Config = None):
        """
        Initialize TimeGAN v6 model.

        Args:
            config: TimeGANv6Config instance (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = TimeGANv6Config()

        self.config = config
        self._build_model()

        # Track training state
        self._autoencoder_frozen = False
        self._current_stage = 1

    def _build_model(self):
        """Build all model components based on configuration."""
        cfg = self.config

        # =====================================================================
        # Encoder (from v6)
        # =====================================================================
        self.encoder = Embedder(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
            num_layers=cfg.encoder_num_layers
        )

        # =====================================================================
        # Decoder (from v6)
        # =====================================================================
        self.decoder = Recovery(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            feature_dim=cfg.feature_dim,
            num_layers=cfg.decoder_num_layers
        )

        # =====================================================================
        # Pooler (NEW in v6)
        # =====================================================================
        self.pooler = LatentPooler(
            latent_dim=cfg.latent_dim,
            pool_type=cfg.pool_type,
            output_dim=cfg.summary_dim,
            dropout=cfg.pooler_dropout
        )

        # =====================================================================
        # Expander (NEW in v6)
        # =====================================================================
        self.expander = LatentExpander(
            summary_dim=cfg.summary_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.expander_hidden_dim,
            num_layers=cfg.expander_num_layers,
            expand_type=cfg.expand_type,
            dropout=cfg.expander_dropout
        )

        # =====================================================================
        # Generator (NEW in v6)
        # =====================================================================
        if cfg.generator_type == 'film':
            self.generator = ConditionalLatentGenerator(
                noise_dim=cfg.noise_dim,
                condition_dim=cfg.condition_dim,
                summary_dim=cfg.summary_dim,
                hidden_dims=cfg.generator_hidden_dims
            )
        else:
            self.generator = LatentGenerator(
                noise_dim=cfg.noise_dim,
                condition_dim=cfg.condition_dim,
                summary_dim=cfg.summary_dim,
                hidden_dims=cfg.generator_hidden_dims,
                use_residual=cfg.generator_use_residual
            )

        # =====================================================================
        # Discriminator (NEW in v6)
        # =====================================================================
        if cfg.discriminator_type == 'projection':
            self.discriminator = ConditionalLatentDiscriminator(
                summary_dim=cfg.summary_dim,
                condition_dim=cfg.condition_dim,
                hidden_dims=cfg.discriminator_hidden_dims,
                use_spectral_norm=cfg.discriminator_use_spectral_norm
            )
        elif cfg.discriminator_type == 'multiscale':
            self.discriminator = MultiScaleLatentDiscriminator(
                summary_dim=cfg.summary_dim,
                condition_dim=cfg.condition_dim,
                n_scales=cfg.discriminator_n_scales,
                hidden_dim=cfg.discriminator_hidden_dims[0]
            )
        else:
            self.discriminator = LatentDiscriminator(
                summary_dim=cfg.summary_dim,
                condition_dim=cfg.condition_dim,
                hidden_dims=cfg.discriminator_hidden_dims,
                use_spectral_norm=cfg.discriminator_use_spectral_norm
            )

    # =========================================================================
    # Stage 1: Autoencoder Training
    # =========================================================================

    def forward_autoencoder(self, x_real: torch.Tensor, lengths: torch.Tensor,
                            return_intermediates: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for Stage 1 autoencoder training.

        Pipeline:
            x_real → Encoder → h_seq → Pooler → z_summary → Expander → h_seq_recon → Decoder → x_recon

        Also computes direct reconstruction (bypassing bottleneck) to give
        encoder stronger gradients:
            x_real → Encoder → h_seq → Decoder → x_direct

        Args:
            x_real: (batch, seq_len, feature_dim) - real trajectories
            lengths: (batch,) - sequence lengths
            return_intermediates: Whether to return intermediate tensors

        Returns:
            x_recon: (batch, seq_len, feature_dim) - reconstructed trajectories
            losses: Dict with 'recon', 'latent', 'direct' losses
            (optionally) intermediates: Dict with intermediate tensors
        """
        batch_size, seq_len, _ = x_real.shape

        # Encode: x → h_seq
        h_seq = self.encoder(x_real, lengths)

        # Pool: h_seq → z_summary
        z_summary = self.pooler(h_seq, lengths)

        # Expand: z_summary → h_seq_recon
        h_seq_recon = self.expander(z_summary, seq_len)

        # Decode through bottleneck: h_seq_recon → x_recon
        x_recon = self.decoder(h_seq_recon, lengths)

        # Direct decode (bypass bottleneck): h_seq → x_direct
        # This gives encoder direct gradients like v6, preventing vanishing gradients
        x_direct = self.decoder(h_seq, lengths)

        # Compute losses
        loss_recon = masked_mse_loss(x_real, x_recon, lengths)
        loss_latent = masked_mse_loss(h_seq.detach(), h_seq_recon, lengths)
        loss_direct = masked_mse_loss(x_real, x_direct, lengths)

        # Total: direct loss trains encoder, recon+latent train bottleneck
        lambda_direct = getattr(self.config, 'lambda_direct', 1.0)
        losses = {
            'recon': loss_recon,
            'latent': loss_latent,
            'direct': loss_direct,
            'total': (self.config.lambda_recon * loss_recon +
                     self.config.lambda_latent * loss_latent +
                     lambda_direct * loss_direct)
        }

        if return_intermediates:
            intermediates = {
                'h_seq': h_seq,
                'z_summary': z_summary,
                'h_seq_recon': h_seq_recon
            }
            return x_recon, losses, intermediates

        return x_recon, losses

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectories to latent summaries.

        Args:
            x: (batch, seq_len, feature_dim)
            lengths: (batch,)

        Returns:
            z_summary: (batch, summary_dim)
        """
        h_seq = self.encoder(x, lengths)
        z_summary = self.pooler(h_seq, lengths)
        return z_summary

    def decode(self, z_summary: torch.Tensor, seq_len: int,
               lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Decode latent summaries to trajectories.

        Args:
            z_summary: (batch, summary_dim)
            seq_len: Target sequence length
            lengths: (batch,) - optional, defaults to seq_len for all

        Returns:
            x: (batch, seq_len, feature_dim)
        """
        batch_size = z_summary.size(0)

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, device=z_summary.device)

        h_seq = self.expander(z_summary, seq_len)
        x = self.decoder(h_seq, lengths)

        return x

    # =========================================================================
    # Stage 2: GAN Training
    # =========================================================================

    def forward_discriminator(self, x_real: torch.Tensor, condition: torch.Tensor,
                              lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for discriminator training in Stage 2.

        Args:
            x_real: (batch, seq_len, feature_dim) - real trajectories
            condition: (batch, condition_dim) - conditions (e.g., distance)
            lengths: (batch,) - sequence lengths

        Returns:
            losses: Dict with 'd_loss', 'd_real', 'd_fake', 'gp'
        """
        batch_size = x_real.size(0)
        device = x_real.device

        # Get real latent summaries (frozen encoder)
        with torch.no_grad():
            h_seq_real = self.encoder(x_real, lengths)
            z_real = self.pooler(h_seq_real, lengths)

        # Generate fake latent summaries
        noise = torch.randn(batch_size, self.config.noise_dim, device=device)
        z_fake = self.generator(noise, condition)

        # Discriminator scores
        d_real = self._discriminator_forward(z_real, condition)
        d_fake = self._discriminator_forward(z_fake.detach(), condition)

        # Gradient penalty
        gp = compute_gradient_penalty_latent(
            self.discriminator, z_real, z_fake.detach(), condition
        )

        # WGAN-GP discriminator loss: E[D(fake)] - E[D(real)] + lambda_gp * GP
        d_loss = d_fake.mean() - d_real.mean() + self.config.lambda_gp * gp

        return {
            'd_loss': d_loss,
            'd_real': d_real.mean(),
            'd_fake': d_fake.mean(),
            'gp': gp,
            'wasserstein': d_real.mean() - d_fake.mean()  # Wasserstein estimate
        }

    def forward_generator(self, batch_size: int, condition: torch.Tensor,
                          x_real: torch.Tensor = None,
                          lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for generator training in Stage 2.

        Args:
            batch_size: Number of samples to generate
            condition: (batch, condition_dim) - conditions
            x_real: Optional real data for feature matching
            lengths: Optional sequence lengths for feature matching

        Returns:
            losses: Dict with 'g_loss', 'g_adv', 'g_fm' (if feature matching)
        """
        device = condition.device

        # Generate fake latent summaries
        noise = torch.randn(batch_size, self.config.noise_dim, device=device)
        z_fake = self.generator(noise, condition)

        # Discriminator score for fake
        d_fake = self._discriminator_forward(z_fake, condition)

        # Generator adversarial loss: -E[D(fake)]
        g_adv = -d_fake.mean()

        losses = {'g_adv': g_adv}

        # Feature matching loss (optional)
        if self.config.use_feature_matching and x_real is not None:
            with torch.no_grad():
                h_seq_real = self.encoder(x_real, lengths)
                z_real = self.pooler(h_seq_real, lengths)

            # Match mean and std of latent distributions
            g_fm = F.mse_loss(z_fake.mean(dim=0), z_real.mean(dim=0))
            g_fm = g_fm + F.mse_loss(z_fake.std(dim=0), z_real.std(dim=0))

            losses['g_fm'] = g_fm
            losses['g_loss'] = g_adv + self.config.lambda_fm * g_fm
        else:
            losses['g_fm'] = torch.tensor(0.0, device=device)
            losses['g_loss'] = g_adv

        return losses

    def _discriminator_forward(self, z: torch.Tensor,
                               condition: torch.Tensor) -> torch.Tensor:
        """
        Helper for discriminator forward (handles multiscale case).
        """
        if isinstance(self.discriminator, MultiScaleLatentDiscriminator):
            return self.discriminator.forward_single(z, condition)
        return self.discriminator(z, condition)

    # =========================================================================
    # Generation
    # =========================================================================

    @torch.no_grad()
    def generate(self, n_samples: int, conditions: torch.Tensor,
                 seq_len: int = None, truncation_psi: float = None) -> torch.Tensor:
        """
        Generate synthetic trajectories.

        Pipeline:
            noise + condition → Generator → z_summary → Expander → h_seq → Decoder → x_fake

        Args:
            n_samples: Number of trajectories to generate
            conditions: (n_samples, condition_dim) - conditions for each sample
            seq_len: Target sequence length (defaults to config.max_seq_len)
            truncation_psi: Noise truncation factor (defaults to config value)

        Returns:
            x_fake: (n_samples, seq_len, feature_dim) - generated trajectories
        """
        self.eval()
        device = conditions.device

        if seq_len is None:
            seq_len = self.config.max_seq_len

        if truncation_psi is None:
            truncation_psi = self.config.truncation_psi

        # Sample noise (with optional truncation)
        noise = torch.randn(n_samples, self.config.noise_dim, device=device)

        if truncation_psi < 1.0:
            # Truncation trick: clip noise to reduce diversity but increase quality
            noise = torch.clamp(noise, -truncation_psi * 2, truncation_psi * 2)

        # Generate latent summaries
        z_summary = self.generator(noise, conditions)

        # Expand to sequence
        h_seq = self.expander(z_summary, seq_len)

        # Decode to features
        lengths = torch.full((n_samples,), seq_len, device=device)
        x_fake = self.decoder(h_seq, lengths)

        # Clamp to valid range
        x_fake = torch.clamp(x_fake, -1, 1)

        return x_fake

    @torch.no_grad()
    def generate_from_latent(self, z_summary: torch.Tensor,
                             seq_len: int = None) -> torch.Tensor:
        """
        Generate trajectories from pre-computed latent summaries.

        Useful for interpolation experiments.

        Args:
            z_summary: (batch, summary_dim) - latent summaries
            seq_len: Target sequence length

        Returns:
            x_fake: (batch, seq_len, feature_dim)
        """
        self.eval()

        if seq_len is None:
            seq_len = self.config.max_seq_len

        batch_size = z_summary.size(0)
        device = z_summary.device

        h_seq = self.expander(z_summary, seq_len)
        lengths = torch.full((batch_size,), seq_len, device=device)
        x_fake = self.decoder(h_seq, lengths)

        return torch.clamp(x_fake, -1, 1)

    @torch.no_grad()
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor,
                    n_steps: int = 10, seq_len: int = None) -> torch.Tensor:
        """
        Interpolate between two latent summaries.

        Args:
            z1: (summary_dim,) or (1, summary_dim) - start latent
            z2: (summary_dim,) or (1, summary_dim) - end latent
            n_steps: Number of interpolation steps
            seq_len: Target sequence length

        Returns:
            x_interp: (n_steps, seq_len, feature_dim) - interpolated trajectories
        """
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if z2.dim() == 1:
            z2 = z2.unsqueeze(0)

        # Linear interpolation weights
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)

        # Interpolate in latent space
        z_interp = torch.stack([
            (1 - alpha) * z1.squeeze(0) + alpha * z2.squeeze(0)
            for alpha in alphas
        ])

        return self.generate_from_latent(z_interp, seq_len)

    # =========================================================================
    # Freezing/Unfreezing
    # =========================================================================

    def freeze_autoencoder(self):
        """
        Freeze autoencoder components for Stage 2 training.

        Freezes: encoder, decoder, pooler, expander
        """
        for component in [self.encoder, self.decoder, self.pooler, self.expander]:
            for param in component.parameters():
                param.requires_grad = False

        self._autoencoder_frozen = True
        self._current_stage = 2

    def unfreeze_autoencoder(self):
        """
        Unfreeze autoencoder components (e.g., for fine-tuning).

        Unfreezes: encoder, decoder, pooler, expander
        """
        for component in [self.encoder, self.decoder, self.pooler, self.expander]:
            for param in component.parameters():
                param.requires_grad = True

        self._autoencoder_frozen = False
        self._current_stage = 1

    def freeze_generator(self):
        """Freeze generator (for discriminator-only training)."""
        for param in self.generator.parameters():
            param.requires_grad = False

    def unfreeze_generator(self):
        """Unfreeze generator."""
        for param in self.generator.parameters():
            param.requires_grad = True

    def freeze_discriminator(self):
        """Freeze discriminator (for generator-only training)."""
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def unfreeze_discriminator(self):
        """Unfreeze discriminator."""
        for param in self.discriminator.parameters():
            param.requires_grad = True

    # =========================================================================
    # Parameter Groups (for optimizer setup)
    # =========================================================================

    def get_autoencoder_parameters(self):
        """Get parameters for autoencoder training (Stage 1)."""
        return list(self.encoder.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.pooler.parameters()) + \
               list(self.expander.parameters())

    def get_generator_parameters(self):
        """Get generator parameters (Stage 2)."""
        return list(self.generator.parameters())

    def get_discriminator_parameters(self):
        """Get discriminator parameters (Stage 2)."""
        return list(self.discriminator.parameters())

    # =========================================================================
    # Utilities
    # =========================================================================

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        counts = {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'pooler': sum(p.numel() for p in self.pooler.parameters()),
            'expander': sum(p.numel() for p in self.expander.parameters()),
            'generator': sum(p.numel() for p in self.generator.parameters()),
            'discriminator': sum(p.numel() for p in self.discriminator.parameters()),
        }
        counts['total'] = sum(counts.values())
        counts['autoencoder'] = counts['encoder'] + counts['decoder'] + \
                                counts['pooler'] + counts['expander']
        counts['gan'] = counts['generator'] + counts['discriminator']
        return counts

    @property
    def is_autoencoder_frozen(self) -> bool:
        """Check if autoencoder is frozen."""
        return self._autoencoder_frozen

    @property
    def current_stage(self) -> int:
        """Get current training stage (1 or 2)."""
        return self._current_stage

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    def summary(self) -> str:
        """Get model summary string."""
        counts = self.count_parameters()
        lines = [
            "TimeGAN v6 Model Summary",
            "=" * 50,
            f"Stage: {self.current_stage}",
            f"Autoencoder frozen: {self.is_autoencoder_frozen}",
            "",
            "Architecture:",
            f"  feature_dim: {self.config.feature_dim}",
            f"  latent_dim: {self.config.latent_dim}",
            f"  summary_dim: {self.config.summary_dim}",
            f"  noise_dim: {self.config.noise_dim}",
            f"  pool_type: {self.config.pool_type}",
            f"  expand_type: {self.config.expand_type}",
            "",
            "Parameters:",
            f"  Encoder: {counts['encoder']:,}",
            f"  Decoder: {counts['decoder']:,}",
            f"  Pooler: {counts['pooler']:,}",
            f"  Expander: {counts['expander']:,}",
            f"  Generator: {counts['generator']:,}",
            f"  Discriminator: {counts['discriminator']:,}",
            f"  ─────────────────────",
            f"  Autoencoder total: {counts['autoencoder']:,}",
            f"  GAN total: {counts['gan']:,}",
            f"  TOTAL: {counts['total']:,}",
        ]
        return "\n".join(lines)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing TimeGAN v6 Model...")
    print("=" * 60)

    device = torch.device('cpu')

    # Test parameters
    batch_size = 4
    seq_len = 50
    feature_dim = 8
    condition_dim = 1

    # Create model with default config
    print("\n=== Creating Model ===")
    config = TimeGANv6Config(device='cpu')
    model = TimeGANv6(config)
    model.to(device)

    print(model.summary())

    # Create test data
    x_real = torch.randn(batch_size, seq_len, feature_dim, device=device)
    x_real = torch.clamp(x_real, -1, 1)
    lengths = torch.tensor([50, 30, 45, 20], device=device)
    condition = torch.rand(batch_size, condition_dim, device=device)

    # Test Stage 1: Autoencoder
    print("\n=== Testing Stage 1: Autoencoder ===")
    x_recon, losses = model.forward_autoencoder(x_real, lengths)
    print(f"Input shape: {x_real.shape}")
    print(f"Recon shape: {x_recon.shape}")
    print(f"Recon loss: {losses['recon'].item():.4f}")
    print(f"Latent loss: {losses['latent'].item():.4f}")
    print(f"Total loss: {losses['total'].item():.4f}")

    # Test gradient flow
    losses['total'].backward()
    print("Gradient flow: OK")
    model.zero_grad()

    # Test with intermediates
    x_recon, losses, intermediates = model.forward_autoencoder(x_real, lengths, return_intermediates=True)
    print(f"h_seq shape: {intermediates['h_seq'].shape}")
    print(f"z_summary shape: {intermediates['z_summary'].shape}")
    print("Stage 1: PASS")

    # Test encode/decode
    print("\n=== Testing Encode/Decode ===")
    z = model.encode(x_real, lengths)
    x_decoded = model.decode(z, seq_len, lengths)
    print(f"Encoded shape: {z.shape}")
    print(f"Decoded shape: {x_decoded.shape}")
    print("Encode/Decode: PASS")

    # Test Stage 2: GAN (freeze autoencoder first)
    print("\n=== Testing Stage 2: GAN ===")
    model.freeze_autoencoder()
    assert model.is_autoencoder_frozen
    assert model.current_stage == 2
    print("Autoencoder frozen: OK")

    # Discriminator forward
    d_losses = model.forward_discriminator(x_real, condition, lengths)
    print(f"D loss: {d_losses['d_loss'].item():.4f}")
    print(f"D(real): {d_losses['d_real'].item():.4f}")
    print(f"D(fake): {d_losses['d_fake'].item():.4f}")
    print(f"GP: {d_losses['gp'].item():.4f}")
    print(f"Wasserstein: {d_losses['wasserstein'].item():.4f}")

    # Test gradient flow (only GAN components should have gradients)
    d_losses['d_loss'].backward()
    assert model.encoder.lstm.weight_ih_l0.grad is None, "Encoder should be frozen"
    print("D gradient flow: OK")
    model.zero_grad()

    # Generator forward
    g_losses = model.forward_generator(batch_size, condition, x_real, lengths)
    print(f"G loss: {g_losses['g_loss'].item():.4f}")
    print(f"G adv: {g_losses['g_adv'].item():.4f}")
    print(f"G fm: {g_losses['g_fm'].item():.4f}")

    g_losses['g_loss'].backward()
    print("G gradient flow: OK")

    print("Stage 2: PASS")

    # Test generation
    print("\n=== Testing Generation ===")
    n_samples = 8
    conditions = torch.rand(n_samples, condition_dim, device=device)
    x_fake = model.generate(n_samples, conditions, seq_len=100)
    print(f"Generated shape: {x_fake.shape}")
    print(f"Generated range: [{x_fake.min().item():.4f}, {x_fake.max().item():.4f}]")
    assert x_fake.min() >= -1 and x_fake.max() <= 1, "Output should be in [-1, 1]"
    print("Generation: PASS")

    # Test interpolation
    print("\n=== Testing Interpolation ===")
    z1 = model.encode(x_real[:1], lengths[:1])
    z2 = model.encode(x_real[1:2], lengths[1:2])
    x_interp = model.interpolate(z1, z2, n_steps=5, seq_len=50)
    print(f"Interpolated shape: {x_interp.shape}")
    print("Interpolation: PASS")

    # Test unfreeze
    print("\n=== Testing Unfreeze ===")
    model.unfreeze_autoencoder()
    assert not model.is_autoencoder_frozen
    assert model.current_stage == 1
    print("Unfreeze: PASS")

    # Test parameter groups
    print("\n=== Testing Parameter Groups ===")
    ae_params = model.get_autoencoder_parameters()
    g_params = model.get_generator_parameters()
    d_params = model.get_discriminator_parameters()
    print(f"Autoencoder params: {sum(p.numel() for p in ae_params):,}")
    print(f"Generator params: {sum(p.numel() for p in g_params):,}")
    print(f"Discriminator params: {sum(p.numel() for p in d_params):,}")
    print("Parameter groups: PASS")

    print("\n" + "=" * 60)
    print("ALL TIMEGAN v6 MODEL TESTS PASSED")
    print("=" * 60)


