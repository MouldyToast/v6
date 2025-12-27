"""
V8 Sampling

Inpainting-based sampling instead of CFG.
Endpoints are fixed at each denoising step to guarantee direction.
"""

from .inpainting_sampler import InpaintingSampler

__all__ = [
    'InpaintingSampler',
]
