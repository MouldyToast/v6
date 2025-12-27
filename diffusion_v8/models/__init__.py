"""
V8 Models

- TrajectoryTransformer: Denoising model (2D input, no goal embedding)
- GaussianDiffusion: DDPM/DDIM diffusion process

NOTE: No GoalConditioner in V8 - direction is controlled via inpainting.
"""

from .trajectory_transformer import TrajectoryTransformer
from .gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

__all__ = [
    'TrajectoryTransformer',
    'GaussianDiffusion',
    'get_named_beta_schedule',
    'create_named_schedule_sampler',
    'ModelMeanType',
    'ModelVarType',
    'LossType',
]
