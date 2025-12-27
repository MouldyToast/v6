"""
Generation CLI for V8 Trajectory Diffusion

V8: Uses inpainting for direction control (no CFG).

Usage:
    # Generate to specific endpoint
    python generate_diffusion_v8.py --checkpoint best.pth --target_x 0.5 --target_y 0.3 --length 100

    # Generate using test set endpoints
    python generate_diffusion_v8.py --checkpoint best.pth --use_test_endpoints --num_samples 100

    # Batch generation
    python generate_diffusion_v8.py --checkpoint best.pth --use_test_endpoints --batch_size 32
"""

import argparse
import numpy as np
import torch
import json
from pathlib import Path

from diffusion_v8.config_trajectory import TrajectoryDiffusionConfig
from diffusion_v8.models import TrajectoryTransformer, GaussianDiffusion, get_named_beta_schedule
from diffusion_v8.sampling import InpaintingSampler


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = TrajectoryTransformer(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_size=config.ff_size,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        activation=config.activation
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create diffusion
    betas = get_named_beta_schedule(config.noise_schedule, config.diffusion_steps)
    diffusion = GaussianDiffusion(betas=betas)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model, diffusion, config


def generate_single(sampler, target_x, target_y, length, ddim_steps, eta):
    """Generate a single trajectory."""
    trajectory = sampler.generate_single(
        target_x=target_x,
        target_y=target_y,
        length=length,
        ddim_steps=ddim_steps,
        eta=eta,
        show_progress=True
    )
    return trajectory


def generate_from_test(sampler, data_dir, num_samples, ddim_steps, eta):
    """Generate trajectories using test set endpoints."""
    # Load test endpoints and lengths
    endpoints = np.load(f'{data_dir}/endpoints_test.npy')
    lengths = np.load(f'{data_dir}/L_test.npy')

    # Limit to num_samples
    n = min(num_samples, len(endpoints))
    endpoints = endpoints[:n]
    lengths = lengths[:n]

    # Load normalization params to convert endpoints
    norm_params = np.load(f'{data_dir}/normalization_params.npy', allow_pickle=True).item()
    position_scale = norm_params['position_scale']

    # Normalize endpoints
    endpoints_norm = endpoints / position_scale

    print(f"\nGenerating {n} trajectories from test endpoints...")

    trajectories = sampler.generate_batch(
        endpoints=endpoints_norm,
        lengths=lengths,
        ddim_steps=ddim_steps,
        eta=eta,
        show_progress=True
    )

    return trajectories, endpoints_norm, lengths


def compute_metrics(trajectories, endpoints, lengths):
    """Compute evaluation metrics."""
    n = len(trajectories)

    endpoint_errors = []
    for i in range(n):
        L = lengths[i]
        final_pos = trajectories[i, L-1, :]
        target_pos = endpoints[i]
        error = np.linalg.norm(final_pos - target_pos)
        endpoint_errors.append(error)

    metrics = {
        'num_samples': n,
        'endpoint_error_mean': float(np.mean(endpoint_errors)),
        'endpoint_error_std': float(np.std(endpoint_errors)),
        'endpoint_error_max': float(np.max(endpoint_errors)),
    }

    return metrics


def save_results(trajectories, endpoints, lengths, metrics, output_dir):
    """Save generated trajectories and metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / 'generated_trajectories.npy', trajectories)
    np.save(output_path / 'generation_endpoints.npy', endpoints)
    np.save(output_path / 'generation_lengths.npy', lengths)

    with open(output_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate trajectories with V8 diffusion')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Target endpoint (for single generation)
    parser.add_argument('--target_x', type=float, default=None,
                        help='Target x position (normalized)')
    parser.add_argument('--target_y', type=float, default=None,
                        help='Target y position (normalized)')
    parser.add_argument('--length', type=int, default=100,
                        help='Trajectory length')

    # Batch generation from test set
    parser.add_argument('--use_test_endpoints', action='store_true',
                        help='Generate using test set endpoints')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')

    # Generation settings
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM sampling steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta (0=deterministic, 1=stochastic)')

    # Data
    parser.add_argument('--data_dir', type=str, default='processed_data_v8',
                        help='Directory with preprocessed V8 data')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/diffusion_v8',
                        help='Output directory')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # Load model
    model, diffusion, config = load_model(args.checkpoint, device)

    # Create sampler
    sampler = InpaintingSampler(
        model=model,
        diffusion=diffusion,
        max_seq_len=config.max_seq_len,
        device=device
    )

    # Generate
    if args.use_test_endpoints:
        # Batch generation from test set
        trajectories, endpoints, lengths = generate_from_test(
            sampler=sampler,
            data_dir=args.data_dir,
            num_samples=args.num_samples,
            ddim_steps=args.ddim_steps,
            eta=args.eta
        )

        # Compute metrics
        metrics = compute_metrics(trajectories, endpoints, lengths)
        print(f"\nMetrics:")
        print(f"  Endpoint error (mean): {metrics['endpoint_error_mean']:.6f}")
        print(f"  Endpoint error (std): {metrics['endpoint_error_std']:.6f}")
        print(f"  Endpoint error (max): {metrics['endpoint_error_max']:.6f}")

        # Save
        save_results(trajectories, endpoints, lengths, metrics, args.output_dir)

    elif args.target_x is not None and args.target_y is not None:
        # Single trajectory generation
        print(f"\nGenerating trajectory to ({args.target_x}, {args.target_y})...")
        trajectory = generate_single(
            sampler=sampler,
            target_x=args.target_x,
            target_y=args.target_y,
            length=args.length,
            ddim_steps=args.ddim_steps,
            eta=args.eta
        )

        print(f"\nGenerated trajectory shape: {trajectory.shape}")
        print(f"Start: ({trajectory[0, 0]:.4f}, {trajectory[0, 1]:.4f})")
        print(f"End: ({trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f})")
        print(f"Target: ({args.target_x}, {args.target_y})")

        endpoint_error = np.linalg.norm(trajectory[-1] - np.array([args.target_x, args.target_y]))
        print(f"Endpoint error: {endpoint_error:.6f}")

        # Save
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / 'single_trajectory.npy', trajectory)
        print(f"\nSaved to: {output_path / 'single_trajectory.npy'}")

    else:
        print("ERROR: Specify either --target_x/--target_y or --use_test_endpoints")
        return

    print("\nGeneration complete!")


if __name__ == '__main__':
    main()
