"""
V8 Datasets

Loads 2D position data (x, y) instead of V7's 8D features.
No condition loading - V8 uses inpainting for direction control.
"""

from .trajectory_dataset import (
    TrajectoryDataset,
    create_dataloader,
    create_dataloaders,
    create_single_batch_dataset
)

__all__ = [
    'TrajectoryDataset',
    'create_dataloader',
    'create_dataloaders',
    'create_single_batch_dataset',
]
