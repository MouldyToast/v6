"""
Diffusion V8 - Inpainting-based Trajectory Generation

V8 Key Changes from V7:
- No CFG (Classifier-Free Guidance)
- No GoalConditioner
- 2D positions (x, y) instead of 8D features
- Direction controlled via endpoint inpainting at inference

Modules:
- config_trajectory: Configuration dataclass
- models: TrajectoryTransformer, GaussianDiffusion
- datasets: TrajectoryDataset for 2D positions
- trainers: TrajectoryDiffusionTrainer (unconditional)
- sampling: InpaintingSampler for generation
- evaluation: Metrics and visualization
"""

from .config_trajectory import (
    TrajectoryDiffusionConfig,
    get_smoke_test_config,
    get_medium_config,
    get_full_config,
    print_config
)

__version__ = '8.0.0'
__all__ = [
    'TrajectoryDiffusionConfig',
    'get_smoke_test_config',
    'get_medium_config',
    'get_full_config',
    'print_config',
]
