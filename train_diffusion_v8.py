"""
Training CLI for V8 Trajectory Diffusion

V8: Unconditional training with inpainting for direction control.

Usage:
    # Smoke test (single-batch overfitting)
    python train_diffusion_v8.py --mode smoke_test

    # Medium model training
    python train_diffusion_v8.py --mode medium --epochs 500

    # Full model training
    python train_diffusion_v8.py --mode full --epochs 1000

    # Custom settings
    python train_diffusion_v8.py --mode medium --epochs 300 --lr 2e-4 --batch_size 32
"""

import argparse
import torch

from diffusion_v8.config_trajectory import (
    get_smoke_test_config,
    get_medium_config,
    get_full_config,
    print_config
)
from diffusion_v8.trainers import TrajectoryDiffusionTrainer
from diffusion_v8.datasets import create_dataloaders, create_single_batch_dataset


def main():
    parser = argparse.ArgumentParser(description='Train V8 trajectory diffusion model')

    # Mode
    parser.add_argument('--mode', type=str, default='smoke_test',
                        choices=['smoke_test', 'medium', 'full'],
                        help='Training mode/model size')

    # Training overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')

    # Data
    parser.add_argument('--data_dir', type=str, default='processed_data_v8',
                        help='Directory with preprocessed V8 data')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detect if not specified')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Get base config
    if args.mode == 'smoke_test':
        config = get_smoke_test_config()
    elif args.mode == 'medium':
        config = get_medium_config()
    else:
        config = get_full_config()

    # Apply overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.data_dir:
        config.data_dir = args.data_dir

    # Device
    if args.device:
        config.device = args.device
    elif not torch.cuda.is_available():
        config.device = 'cpu'
        print("CUDA not available, using CPU")

    # Print config
    print_config(config)

    # Create trainer
    trainer = TrajectoryDiffusionTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Training
    if config.smoke_test:
        # Smoke test: overfit on single batch
        print("\nLoading single batch for smoke test...")
        single_batch, dataset = create_single_batch_dataset(
            data_dir=config.data_dir,
            batch_size=config.smoke_test_batch_size
        )
        trainer.train_smoke_test(single_batch)
    else:
        # Full training
        print("\nLoading data...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        trainer.train_full(train_loader, val_loader)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
