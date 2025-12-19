"""
Diffusion V7 Datasets Module

Contains trajectory dataset and dataloader utilities.
"""

from .trajectory_dataset import (
    TrajectoryDataset,
    create_dataloader,
    create_dataloaders,
    create_single_batch_dataset
)

# Note: Original MotionDiffuse dataset classes are available but not imported
# by default to avoid dependency issues. Import directly if needed:
# from .dataset import Text2MotionDataset
# from .evaluator import EvaluationDataset, etc.

__all__ = [
    'TrajectoryDataset',
    'create_dataloader',
    'create_dataloaders',
    'create_single_batch_dataset',
]
