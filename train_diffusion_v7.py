#!/usr/bin/env python3
"""
Main Training Script for Trajectory Diffusion Model

Usage:
    # Smoke test (single-batch overfitting)
    python train_diffusion_v7.py --mode smoke_test --data_dir processed_data_v6

    # Medium model training
    python train_diffusion_v7.py --mode medium --data_dir processed_data_v6

    # Full model training
    python train_diffusion_v7.py --mode full --data_dir processed_data_v6
"""

import argparse
import sys
import torch

from diffusion_v7.config_trajectory import (
    TrajectoryDiffusionConfig,
    get_smoke_test_config,
    get_medium_config,
    get_full_config,
    print_config
)
from diffusion_v7.trainers import TrajectoryDiffusionTrainer
from diffusion_v7.datasets import (
    create_dataloaders,
    create_single_batch_dataset
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Trajectory Diffusion Model')

    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['smoke_test', 'medium', 'full'],
        required=True,
        help='Training mode: smoke_test (single-batch), medium, or full'
    )

    # Data
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory with processed V6 data (X_train.npy, etc.)'
    )

    # Override config parameters
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--device', type=str, help='Device: cuda or cpu')

    # Paths
    parser.add_argument('--save_dir', type=str, help='Override checkpoint save directory')
    parser.add_argument('--log_dir', type=str, help='Override TensorBoard log directory')

    # Resume
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    return parser.parse_args()


def main():
    args = parse_args()

    # Get base config based on mode
    if args.mode == 'smoke_test':
        config = get_smoke_test_config()
        print("\n🔬 SMOKE TEST MODE")
        print("Goal: Overfit on 64 samples to verify architecture")
    elif args.mode == 'medium':
        config = get_medium_config()
        print("\n📊 MEDIUM MODEL MODE")
        print("Goal: Balance between speed and capacity")
    elif args.mode == 'full':
        config = get_full_config()
        print("\n🚀 FULL MODEL MODE")
        print("Goal: Maximum capacity for best results")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Override config with command-line arguments
    config.data_dir = args.data_dir

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.device:
        config.device = args.device
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.log_dir:
        config.log_dir = args.log_dir

    # Print configuration
    print_config(config)

    # Check CUDA availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, falling back to CPU")
        config.device = 'cpu'

    # Create trainer
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)

    trainer = TrajectoryDiffusionTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run appropriate training mode
    if config.smoke_test:
        # Smoke test: single-batch overfitting
        print("\n" + "=" * 70)
        print("PHASE 2.1: SINGLE-BATCH OVERFITTING TEST")
        print("=" * 70)
        print("This test verifies the architecture can memorize a small batch.")
        print("Expected: Loss should go to near-zero within 1000 steps.")
        print("If loss doesn't decrease, there's a bug in:")
        print("  - Data loading (check shapes, normalization)")
        print("  - Model architecture (check forward pass, masking)")
        print("  - Training loop (check gradients, loss computation)")
        print("=" * 70 + "\n")

        # Create single batch
        single_batch, dataset = create_single_batch_dataset(
            data_dir=config.data_dir,
            batch_size=config.smoke_test_batch_size,
            split='train'
        )

        # Move batch to device
        single_batch = {
            k: v.to(config.device) for k, v in single_batch.items()
        }

        # Run smoke test
        best_loss = trainer.train_smoke_test(single_batch)

        # Verdict
        print("\n" + "=" * 70)
        print("SMOKE TEST VERDICT")
        print("=" * 70)
        if best_loss < 0.01:
            print("✅ PASS: Architecture is working correctly")
            print("   → Ready to proceed with full training")
        elif best_loss < 0.1:
            print("⚠️  PARTIAL: Loss decreased but not to zero")
            print("   → May need more steps or hyperparameter tuning")
            print("   → Check learning rate, model capacity")
        else:
            print("❌ FAIL: Loss did not decrease significantly")
            print("   → Critical bug in data loading or model architecture")
            print("   → Check data shapes, normalization, forward pass")
        print("=" * 70 + "\n")

    else:
        # Full training: create dataloaders
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            max_seq_len=config.max_seq_len
        )

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val: {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
        print(f"  Test: {len(test_loader.dataset)} samples ({len(test_loader)} batches)")

        # Run full training
        trainer.train_full(train_loader, val_loader)

    print("\n✅ Training complete!")
    print(f"Checkpoints saved to: {config.save_dir}")
    if config.log_dir:
        print(f"TensorBoard logs: {config.log_dir}")
        print(f"   View with: tensorboard --logdir {config.log_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
