"""
V8 Trainers

Unconditional training - no CFG, no GoalConditioner.
The model learns general trajectory dynamics.
"""

from .trajectory_trainer import TrajectoryDiffusionTrainer

__all__ = [
    'TrajectoryDiffusionTrainer',
]
