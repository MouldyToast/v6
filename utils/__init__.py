"""
Utils module for TimeGAN

Contains loss functions, augmentation, and data loading utilities.

V6 Data Loaders:
    - data_loader_v4: V4-compatible data loader (basic)
    - data_loader_v6: V6-optimized data loader with stage-specific sampling
        - Condition-stratified sampling for Stage 2 (WGAN-GP)
        - Length-aware batching for Stage 1 (Autoencoder)
        - Quality filtering for cleaner latent space learning

Usage:
    from utils import create_stage_loaders, create_dataloaders_v6

    # Stage-specific loaders (recommended for V6)
    stage1_loader, stage2_loader = create_stage_loaders(data_dir)

    # Standard loaders (backward compatible)
    train_loader, val_loader, test_loader = create_dataloaders_v6(data_dir)
"""

# V4 data loading (backward compatible)
from .data_loader_v4 import (
    MouseTrajectoryDatasetV4,
    collate_fn_v4,
    create_dataloaders_v4,
    create_dataloader_v4,
    load_normalization_params,
    load_v4_data
)

# V6 data loading (optimized)
from .data_loader_v6 import (
    MouseTrajectoryDatasetV6,
    ConditionStratifiedSampler,
    LengthAwareSampler,
    collate_fn_v6,
    collate_fn_v6_full,
    create_dataloaders_v6,
    create_stage1_loader,
    create_stage2_loader,
    create_stage_loaders,
    load_v6_data,
    validate_v6_data
)

__all__ = [
    # Legacy modules
    'losses', 'augmentation_v3', 'data_loader_v4', 'data_loader_v6',

    # V4 (backward compatible)
    'MouseTrajectoryDatasetV4',
    'collate_fn_v4',
    'create_dataloaders_v4',
    'create_dataloader_v4',
    'load_normalization_params',
    'load_v4_data',

    # V6 (optimized)
    'MouseTrajectoryDatasetV6',
    'ConditionStratifiedSampler',
    'LengthAwareSampler',
    'collate_fn_v6',
    'collate_fn_v6_full',
    'create_dataloaders_v6',
    'create_stage1_loader',
    'create_stage2_loader',
    'create_stage_loaders',
    'load_v6_data',
    'validate_v6_data',
]

