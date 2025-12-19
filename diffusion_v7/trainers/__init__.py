"""
Diffusion V7 Trainers Module

Contains training loop implementations.
"""

from .trajectory_trainer import TrajectoryDiffusionTrainer

# Note: DDPMTrainer (original MotionDiffuse) is available but not imported
# by default due to dependency issues. Import directly if needed:
# from .ddpm_trainer import DDPMTrainer

__all__ = [
    'TrajectoryDiffusionTrainer',
]
