"""
Diffusion V7 Models Module

Contains trajectory diffusion model components.
"""

from .goal_conditioner import GoalConditioner
from .trajectory_transformer import TrajectoryTransformer
from .gaussian_diffusion import GaussianDiffusion

# Note: MotionTransformer (original MotionDiffuse) is available but not imported
# by default to avoid dependency issues (requires CLIP). Import directly if needed:
# from .transformer import MotionTransformer

__all__ = [
    'GoalConditioner',
    'TrajectoryTransformer',
    'GaussianDiffusion',
]
