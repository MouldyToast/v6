"""
TimeGAN V6: RTSGAN-Style Two-Stage Training

This module implements the V6 architecture with strictly decoupled two-stage training:
- Stage 1: Train autoencoder to convergence (stable latent space)
- Stage 2: Train WGAN-GP in frozen latent space

Key insight: By training the autoencoder first, then training a WGAN generator
in the frozen latent space, we eliminate the encoder-generator semantic gap.

Usage:
    from timegan_v6 import TimeGANV6, TimeGANV6Config

    config = TimeGANV6Config()
    model = TimeGANV6(config)

    # Stage 1: Autoencoder
    x_recon, losses = model.forward_autoencoder(x_real, lengths)

    # Stage 2: GAN
    model.freeze_autoencoder()
    d_losses = model.forward_discriminator(x_real, condition, lengths)
    g_losses = model.forward_generator(batch_size, condition)

    # Generate
    x_fake = model.generate(n_samples, conditions)
"""

# Phase 1: Core components
from .pooler_v6 import LatentPooler
from .expander_v6 import LatentExpander
from .latent_generator_v6 import LatentGenerator, ConditionalLatentGenerator
from .latent_discriminator_v6 import (
    LatentDiscriminator,
    ConditionalLatentDiscriminator,
    MultiScaleLatentDiscriminator
)
from .utils_v6 import (
    create_mask,
    masked_mse_loss,
    masked_l1_loss,
    compute_gradient_penalty_latent,
    get_last_valid_output,
    initialize_weights
)

# Phase 2: Model and configuration
from .config_model_v6 import (
    TimeGANV6Config,
    get_default_config,
    get_fast_config,
    get_large_config
)
from .model_v6 import TimeGANV6

__all__ = [
    # Main model
    'TimeGANV6',
    'TimeGANV6Config',

    # Configuration presets
    'get_default_config',
    'get_fast_config',
    'get_large_config',

    # Core components (Phase 1)
    'LatentPooler',
    'LatentExpander',
    'LatentGenerator',
    'ConditionalLatentGenerator',
    'LatentDiscriminator',
    'ConditionalLatentDiscriminator',
    'MultiScaleLatentDiscriminator',

    # Utilities
    'create_mask',
    'masked_mse_loss',
    'masked_l1_loss',
    'compute_gradient_penalty_latent',
    'get_last_valid_output',
    'initialize_weights',
]

__version__ = '6.0.0'
