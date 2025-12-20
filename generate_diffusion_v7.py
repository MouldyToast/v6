"""
Generate Trajectories using Trained Diffusion Model

Usage (Windows PowerShell / Command Prompt - single line):
    python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth --num_samples 100 --cfg_scale 2.0 --output_dir results/

    python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/final.pth --num_samples 100

    python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/checkpoint_epoch_50.pth --num_samples 100

Usage (Unix/Linux/Mac - with line continuation):
    python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth \
                                     --num_samples 100 \
                                     --cfg_scale 2.0 \
                                     --output_dir results/

This script:
1. Loads a trained diffusion model
2. Generates trajectories from test conditions
3. Evaluates goal accuracy and realism
4. Saves generated trajectories and metrics
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from diffusion_v7.config_trajectory import TrajectoryDiffusionConfig
from diffusion_v7.models import GoalConditioner, TrajectoryTransformer, GaussianDiffusion
from diffusion_v7.models.gaussian_diffusion import (
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType
)
from diffusion_v7.sampling import TrajectoryGenerator
from diffusion_v7.evaluation import evaluate_trajectories, print_metrics_report
from diffusion_v7.datasets import TrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate trajectories with trained diffusion model')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (if not in checkpoint)')

    # Generation
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of trajectories to generate')
    parser.add_argument('--samples_per_condition', type=int, default=1,
                        help='Number of samples per condition (for diversity)')
    parser.add_argument('--method', type=str, default='ddim', choices=['ddim', 'ddpm'],
                        help='Sampling method')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM steps (only for method=ddim)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='DDIM eta (0=deterministic, 1=stochastic)')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale (1.0=no guidance)')
    parser.add_argument('--no_cfg', action='store_true',
                        help='Disable CFG')

    # Data
    parser.add_argument('--data_dir', type=str, default='processed_data_v6',
                        help='Directory with processed data (for test set and normalization)')
    parser.add_argument('--use_test_conditions', action='store_true',
                        help='Use conditions from test set instead of random')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/diffusion_v7',
                        help='Output directory for results')
    parser.add_argument('--save_trajectories', action='store_true',
                        help='Save generated trajectories as .npy')

    # Evaluation
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='Run evaluation metrics')
    parser.add_argument('--compare_real', action='store_true',
                        help='Compare with real trajectories from test set')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load trained model checkpoint.

    Returns:
        model, goal_conditioner, diffusion, config, checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Loaded config from checkpoint")
    else:
        # Fallback to default config
        print("Warning: No config in checkpoint, using default smoke_test config")
        config = TrajectoryDiffusionConfig.smoke_test()

    # Create models
    goal_conditioner = GoalConditioner(
        condition_dim=3,
        latent_dim=config.latent_dim,
        use_cfg=config.use_cfg,
        dropout=config.cfg_dropout
    ).to(device)

    model = TrajectoryTransformer(
        input_dim=8,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_size=config.ff_size,
        goal_latent_dim=config.latent_dim,
        max_seq_len=200
    ).to(device)

    # Create diffusion process (match trainer initialization)
    betas = get_named_beta_schedule(
        config.noise_schedule,
        config.diffusion_steps
    )

    # Convert config strings to enums
    model_mean_type = {
        'epsilon': ModelMeanType.EPSILON,
        'x0': ModelMeanType.START_X,
        'previous_x': ModelMeanType.PREVIOUS_X
    }.get(config.model_mean_type, ModelMeanType.EPSILON)

    model_var_type = {
        'fixed_small': ModelVarType.FIXED_SMALL,
        'fixed_large': ModelVarType.FIXED_LARGE,
        'learned': ModelVarType.LEARNED,
        'learned_range': ModelVarType.LEARNED_RANGE
    }.get(config.model_var_type, ModelVarType.FIXED_SMALL)

    loss_type = {
        'mse': LossType.MSE,
        'l1': LossType.RESCALED_MSE,
        'kl': LossType.KL
    }.get(config.loss_type, LossType.MSE)

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type
    )

    # Load weights
    goal_conditioner.load_state_dict(checkpoint['goal_conditioner_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  - latent_dim: {config.latent_dim}")
    print(f"  - num_layers: {config.num_layers}")
    print(f"  - use_cfg: {config.use_cfg}")

    return model, goal_conditioner, diffusion, config, checkpoint


def generate_random_conditions(num_samples: int, device: torch.device):
    """
    Generate random goal conditions.

    Returns:
        conditions: (num_samples, 3) - [distance_norm, cos(angle), sin(angle)]
        lengths: (num_samples,) - sequence lengths
    """
    # Random normalized distances [-1, 1]
    distances = torch.rand(num_samples, device=device) * 2 - 1

    # Random angles [0, 2π]
    angles = torch.rand(num_samples, device=device) * 2 * np.pi

    # Convert to condition format
    conditions = torch.stack([
        distances,
        torch.cos(angles),
        torch.sin(angles)
    ], dim=1)

    # Random lengths [30, 150] (reasonable range)
    lengths = torch.randint(30, 150, (num_samples,), device=device)

    return conditions, lengths


def load_test_conditions(data_dir: str, num_samples: int, device: torch.device):
    """
    Load conditions from test set.

    Returns:
        conditions: (num_samples, 3)
        lengths: (num_samples,)
        test_dataset: TrajectoryDataset for comparison
    """
    print(f"Loading test conditions from {data_dir}...")

    test_dataset = TrajectoryDataset(data_dir, split='test', max_seq_len=200)

    # Sample random test conditions
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)

    conditions = []
    lengths = []

    for idx in indices:
        sample = test_dataset[idx]
        conditions.append(sample['condition'])
        lengths.append(sample['length'])

    conditions = torch.stack(conditions).to(device)
    lengths = torch.stack(lengths).to(device)

    print(f"✓ Loaded {num_samples} test conditions")

    return conditions, lengths, test_dataset


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DIFFUSION V7 TRAJECTORY GENERATION")
    print("=" * 70)

    # Device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Load model
    model, goal_conditioner, diffusion, config, checkpoint = load_checkpoint(
        args.checkpoint, device
    )

    # Create generator
    print("\nCreating trajectory generator...")
    generator = TrajectoryGenerator(
        model=model,
        goal_conditioner=goal_conditioner,
        diffusion=diffusion,
        device=device
    )
    print("✓ Generator ready")

    # Get conditions
    test_dataset = None
    if args.use_test_conditions:
        conditions, lengths, test_dataset = load_test_conditions(
            args.data_dir, args.num_samples, device
        )
    else:
        conditions, lengths = generate_random_conditions(args.num_samples, device)
        print(f"\nGenerated {args.num_samples} random conditions")

    # Generate trajectories
    print("\n" + "-" * 70)
    print("GENERATING TRAJECTORIES")
    print("-" * 70)
    print(f"Method: {args.method}")
    print(f"CFG scale: {args.cfg_scale if not args.no_cfg else 'disabled'}")
    print(f"Samples per condition: {args.samples_per_condition}")
    if args.method == 'ddim':
        print(f"DDIM steps: {args.ddim_steps}")
        print(f"DDIM eta: {args.ddim_eta}")
    print()

    trajectories = generator.generate(
        conditions=conditions,
        lengths=lengths,
        num_samples=args.samples_per_condition,
        method=args.method,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        cfg_scale=args.cfg_scale,
        use_cfg=not args.no_cfg,
        show_progress=True
    )

    print(f"\n✓ Generated {trajectories.shape[0]} trajectories")
    print(f"  Shape: {trajectories.shape}")

    # Save trajectories
    if args.save_trajectories:
        traj_path = output_dir / 'generated_trajectories.npy'
        cond_path = output_dir / 'generation_conditions.npy'
        len_path = output_dir / 'generation_lengths.npy'

        # Expand conditions/lengths if num_samples > 1
        if args.samples_per_condition > 1:
            save_conditions = conditions.repeat_interleave(args.samples_per_condition, dim=0)
            save_lengths = lengths.repeat_interleave(args.samples_per_condition, dim=0)
        else:
            save_conditions = conditions
            save_lengths = lengths

        np.save(traj_path, trajectories.cpu().numpy())
        np.save(cond_path, save_conditions.cpu().numpy())
        np.save(len_path, save_lengths.cpu().numpy())

        print(f"\n✓ Saved trajectories to {traj_path}")
        print(f"  Conditions: {cond_path}")
        print(f"  Lengths: {len_path}")

    # Evaluation
    if args.evaluate:
        print("\n" + "-" * 70)
        print("EVALUATION")
        print("-" * 70)

        # Expand conditions/lengths for evaluation
        if args.samples_per_condition > 1:
            eval_conditions = conditions.repeat_interleave(args.samples_per_condition, dim=0)
            eval_lengths = lengths.repeat_interleave(args.samples_per_condition, dim=0)
        else:
            eval_conditions = conditions
            eval_lengths = lengths

        # Load real trajectories for comparison if requested
        real_trajectories = None
        real_lengths = None
        if args.compare_real:
            if test_dataset is None:
                test_dataset = TrajectoryDataset(args.data_dir, split='test', max_seq_len=200)

            # Use a subset of test set for comparison
            num_real = min(500, len(test_dataset))
            real_indices = np.random.choice(len(test_dataset), size=num_real, replace=False)

            real_trajs = []
            real_lens = []
            for idx in real_indices:
                sample = test_dataset[idx]
                real_trajs.append(sample['motion'])
                real_lens.append(sample['length'])

            real_trajectories = torch.stack(real_trajs).to(device)
            real_lengths = torch.stack(real_lens).to(device)

            print(f"Using {num_real} real trajectories for comparison")

        # Load normalization params if available
        norm_params = None
        norm_path_json = Path(args.data_dir) / 'normalization_params.json'
        norm_path_npy = Path(args.data_dir) / 'normalization_params.npy'

        if norm_path_json.exists():
            with open(norm_path_json, 'r') as f:
                norm_params = json.load(f)
            print(f"Loaded normalization params from {norm_path_json}")
        elif norm_path_npy.exists():
            norm_params = np.load(norm_path_npy, allow_pickle=True).item()
            print(f"Loaded normalization params from {norm_path_npy}")
            # Show key parameters
            if 'goal_dist_min' in norm_params and 'goal_dist_max' in norm_params:
                print(f"  Distance range: {norm_params['goal_dist_min']:.2f} to {norm_params['goal_dist_max']:.2f} pixels")
        else:
            print(f"⚠️  Warning: No normalization params found at {norm_path_npy}")

        # Run evaluation
        metrics = evaluate_trajectories(
            trajectories=trajectories,
            conditions=eval_conditions,
            lengths=eval_lengths,
            real_trajectories=real_trajectories,
            real_lengths=real_lengths,
            norm_params=norm_params
        )

        # Print report
        print_metrics_report(metrics)

        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # numpy scalar
                metrics_json[key] = value.item()
            elif isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value

        # Save metrics
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")

    # Save generation info
    info = {
        'checkpoint': str(args.checkpoint),
        'num_samples': args.num_samples,
        'samples_per_condition': args.samples_per_condition,
        'method': args.method,
        'ddim_steps': args.ddim_steps if args.method == 'ddim' else None,
        'ddim_eta': args.ddim_eta if args.method == 'ddim' else None,
        'cfg_scale': args.cfg_scale if not args.no_cfg else None,
        'use_cfg': not args.no_cfg,
        'model_config': {
            'latent_dim': config.latent_dim,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'ff_size': config.ff_size,
        },
        'epoch': checkpoint.get('epoch', None),
    }

    info_path = output_dir / 'generation_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\n✓ Saved generation info to {info_path}")

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()


