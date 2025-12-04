"""
TimeGAN V6: RTSGAN-Style Two-Stage Training

This module implements the V6 architecture with strictly decoupled two-stage training:
- Stage 1: Train autoencoder to convergence (stable latent space)
- Stage 2: Train WGAN-GP in frozen latent space

Key insight: By training the autoencoder first, then training a WGAN generator
in the frozen latent space, we eliminate the encoder-generator semantic gap.
"""

from .pooler_v6 import LatentPooler
from .expander_v6 import LatentExpander
from .latent_generator_v6 import LatentGenerator
from .latent_discriminator_v6 import LatentDiscriminator
from .utils_v6 import create_mask, masked_mse_loss, compute_gradient_penalty_latent

__all__ = [
    'LatentPooler',
    'LatentExpander',
    'LatentGenerator',
    'LatentDiscriminator',
    'create_mask',
    'masked_mse_loss',
    'compute_gradient_penalty_latent',
]

__version__ = '6.0.0'
