"""
Trajectory Visualization for V8 Diffusion

CRITICAL: This script tests visualization on TRAINING DATA first,
before being used on generated data. This ensures any issues we see
in generated trajectories are from the model, not the visualization.

Usage:
    # Step 1: Verify visualization with training data
    python visualize_trajectories_v8.py --mode training --data_dir processed_data_v8

    # Step 2: After training, visualize generated trajectories
    python visualize_trajectories_v8.py --mode generated --input generated_trajectories.npy

    # Step 3: Compare training vs generated side-by-side
    python visualize_trajectories_v8.py --mode compare --data_dir processed_data_v8 --input generated_trajectories.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_training_data(data_dir: str, split: str = 'train', num_samples: int = 10):
    """
    Load training trajectories for visualization verification.

    Returns:
        trajectories: list of (length, 2) arrays
        endpoints: (num_samples, 2) array
        norm_params: dict with position_scale
    """
    X = np.load(f'{data_dir}/X_{split}.npy')  # (N, max_len, 2)
    L = np.load(f'{data_dir}/L_{split}.npy')  # (N,)
    endpoints = np.load(f'{data_dir}/endpoints_{split}.npy')  # (N, 2)
    norm_params = np.load(f'{data_dir}/normalization_params.npy', allow_pickle=True).item()

    # Extract actual trajectories (remove padding)
    trajectories = []
    selected_endpoints = []
    for i in range(min(num_samples, len(X))):
        length = int(L[i])
        traj = X[i, :length, :]  # (length, 2)
        trajectories.append(traj)
        selected_endpoints.append(endpoints[i])

    return trajectories, np.array(selected_endpoints), norm_params


def verify_trajectory(traj: np.ndarray, expected_endpoint: np.ndarray = None,
                      position_scale: float = None, name: str = "trajectory"):
    """
    Print numerical verification of a trajectory.

    This helps catch visualization bugs by checking actual values.
    """
    print(f"\n{'='*50}")
    print(f"Verifying: {name}")
    print(f"{'='*50}")
    print(f"Shape: {traj.shape}")
    print(f"Length: {len(traj)} points")
    print(f"")
    print(f"Start position: ({traj[0, 0]:.6f}, {traj[0, 1]:.6f})")
    print(f"End position:   ({traj[-1, 0]:.6f}, {traj[-1, 1]:.6f})")

    if expected_endpoint is not None:
        # Normalize expected endpoint for comparison
        if position_scale is not None:
            expected_norm = expected_endpoint / position_scale
        else:
            expected_norm = expected_endpoint
        print(f"Expected end:   ({expected_norm[0]:.6f}, {expected_norm[1]:.6f})")

        error = np.linalg.norm(traj[-1] - expected_norm)
        print(f"Endpoint error: {error:.6f}")

    # Check if start is at origin (should be for V8 data)
    start_dist = np.linalg.norm(traj[0])
    if start_dist > 1e-6:
        print(f"WARNING: Start is not at origin! Distance: {start_dist:.6f}")
    else:
        print(f"Start at origin: OK")

    # Basic stats
    print(f"")
    print(f"X range: [{traj[:, 0].min():.4f}, {traj[:, 0].max():.4f}]")
    print(f"Y range: [{traj[:, 1].min():.4f}, {traj[:, 1].max():.4f}]")

    # Path length
    diffs = np.diff(traj, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))
    direct_distance = np.linalg.norm(traj[-1] - traj[0])
    print(f"Path length: {path_length:.4f}")
    print(f"Direct distance: {direct_distance:.4f}")
    print(f"Efficiency (direct/path): {direct_distance/path_length:.2%}" if path_length > 0 else "")


def plot_single_trajectory(ax, traj: np.ndarray, color: str = 'blue',
                           label: str = None, show_points: bool = True):
    """
    Plot a single trajectory with consistent styling.

    Args:
        ax: matplotlib axes
        traj: (length, 2) array of positions
        color: line color
        label: legend label
        show_points: whether to show start/end markers
    """
    # Plot the path
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
            alpha=0.7, label=label)

    if show_points:
        # Start point (green circle)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100,
                   marker='o', zorder=5, edgecolors='black', linewidths=1)

        # End point (red X)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100,
                   marker='X', zorder=5, edgecolors='black', linewidths=1)


def setup_axes(ax, title: str = None):
    """
    Configure axes with proper settings for trajectory visualization.
    """
    ax.set_aspect('equal')  # CRITICAL: prevents distortion
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    if title:
        ax.set_title(title)


def visualize_training_data(data_dir: str, num_samples: int = 9, save_path: str = None):
    """
    Visualize training data to verify our visualization code is correct.

    This is the FIRST step - if training data looks wrong, fix visualization.
    If training data looks right, we can trust visualization for generated data.
    """
    print("\n" + "="*70)
    print("STEP 1: VISUALIZING TRAINING DATA")
    print("="*70)
    print("Purpose: Verify visualization code is correct")
    print("Expected: Smooth mouse-like paths from origin to various endpoints")
    print("="*70)

    # Load data
    trajectories, endpoints, norm_params = load_training_data(
        data_dir, split='train', num_samples=num_samples
    )
    position_scale = norm_params['position_scale']

    print(f"\nLoaded {len(trajectories)} training trajectories")
    print(f"Position scale: {position_scale:.2f} pixels")

    # Verify first few numerically
    for i in range(min(3, len(trajectories))):
        verify_trajectory(
            trajectories[i],
            endpoints[i],
            position_scale,
            name=f"Training trajectory {i+1}"
        )

    # Plot grid
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, num_samples))

    for i, traj in enumerate(trajectories):
        ax = axes[i]
        plot_single_trajectory(ax, traj, color=colors[i])
        setup_axes(ax, title=f'Training #{i+1} (len={len(traj)})')

    # Hide unused subplots
    for i in range(len(trajectories), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Training Data Verification\n(Green=Start, Red=End)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")

    plt.show()

    print("\n" + "="*70)
    print("VERIFICATION CHECKLIST:")
    print("="*70)
    print("[ ] All trajectories start at origin (green dot at 0,0)?")
    print("[ ] Paths look like realistic mouse movements?")
    print("[ ] Paths are smooth, not random noise?")
    print("[ ] End points (red X) are at various locations?")
    print("[ ] Aspect ratio looks correct (not stretched)?")
    print("="*70)


def visualize_generated(input_path: str, lengths_path: str = None,
                        endpoints_path: str = None, num_samples: int = 9,
                        save_path: str = None):
    """
    Visualize generated trajectories.

    Only use this AFTER verifying training data looks correct.
    """
    print("\n" + "="*70)
    print("STEP 2: VISUALIZING GENERATED DATA")
    print("="*70)
    print("WARNING: Only interpret these results if training data looked correct!")
    print("="*70)

    # Load generated data
    generated = np.load(input_path)  # (batch, max_len, 2)
    print(f"Loaded generated trajectories: {generated.shape}")

    # Load lengths if provided
    if lengths_path:
        lengths = np.load(lengths_path)
    else:
        # Assume full length
        lengths = np.full(len(generated), generated.shape[1])

    # Load endpoints if provided (for verification)
    expected_endpoints = None
    if endpoints_path:
        expected_endpoints = np.load(endpoints_path)

    # Extract trajectories
    trajectories = []
    for i in range(min(num_samples, len(generated))):
        length = int(lengths[i])
        traj = generated[i, :length, :]
        trajectories.append(traj)

    # Verify first few numerically
    for i in range(min(3, len(trajectories))):
        expected = expected_endpoints[i] if expected_endpoints is not None else None
        verify_trajectory(
            trajectories[i],
            expected,
            position_scale=None,  # Already normalized
            name=f"Generated trajectory {i+1}"
        )

    # Plot grid
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    colors = plt.cm.plasma(np.linspace(0, 1, num_samples))

    for i, traj in enumerate(trajectories):
        ax = axes[i]
        plot_single_trajectory(ax, traj, color=colors[i])

        # Show expected endpoint if available
        if expected_endpoints is not None:
            ax.scatter(expected_endpoints[i, 0], expected_endpoints[i, 1],
                      c='yellow', s=150, marker='*', zorder=4,
                      edgecolors='black', linewidths=1, label='Target')

        setup_axes(ax, title=f'Generated #{i+1} (len={len(traj)})')

    # Hide unused
    for i in range(len(trajectories), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Generated Trajectories\n(Green=Start, Red=End, Yellow Star=Target)',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")

    plt.show()


def visualize_comparison(data_dir: str, input_path: str,
                         num_samples: int = 6, save_path: str = None):
    """
    Side-by-side comparison of training vs generated trajectories.

    This is the best way to assess if the model is working correctly.
    """
    print("\n" + "="*70)
    print("STEP 3: TRAINING vs GENERATED COMPARISON")
    print("="*70)

    # Load training data
    train_trajs, train_endpoints, norm_params = load_training_data(
        data_dir, split='train', num_samples=num_samples
    )

    # Load generated data
    generated = np.load(input_path)
    gen_trajs = [generated[i] for i in range(min(num_samples, len(generated)))]

    # Plot comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))

    for i in range(min(num_samples, len(train_trajs), len(gen_trajs))):
        # Training (top row)
        ax_train = axes[0, i]
        plot_single_trajectory(ax_train, train_trajs[i], color='blue')
        setup_axes(ax_train, title=f'Training #{i+1}')

        # Generated (bottom row)
        ax_gen = axes[1, i]
        plot_single_trajectory(ax_gen, gen_trajs[i], color='orange')
        setup_axes(ax_gen, title=f'Generated #{i+1}')

    axes[0, 0].set_ylabel('TRAINING DATA', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('GENERATED', fontsize=12, fontweight='bold')

    plt.suptitle('Training vs Generated Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")

    plt.show()

    print("\n" + "="*70)
    print("COMPARISON CHECKLIST:")
    print("="*70)
    print("[ ] Generated paths have similar smoothness to training?")
    print("[ ] Generated paths reach their endpoints?")
    print("[ ] Generated paths don't look like random noise?")
    print("[ ] Generated paths show natural curvature?")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize V8 trajectories')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'generated', 'compare'],
                        help='Visualization mode')
    parser.add_argument('--data_dir', type=str, default='processed_data_v8',
                        help='Directory with training data')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to generated trajectories .npy file')
    parser.add_argument('--lengths', type=str, default=None,
                        help='Path to lengths .npy file (for generated)')
    parser.add_argument('--endpoints', type=str, default=None,
                        help='Path to expected endpoints .npy file')
    parser.add_argument('--num_samples', type=int, default=9,
                        help='Number of samples to visualize')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to path')

    args = parser.parse_args()

    if args.mode == 'training':
        visualize_training_data(args.data_dir, args.num_samples, args.save)

    elif args.mode == 'generated':
        if args.input is None:
            print("ERROR: --input required for generated mode")
            return
        visualize_generated(args.input, args.lengths, args.endpoints,
                           args.num_samples, args.save)

    elif args.mode == 'compare':
        if args.input is None:
            print("ERROR: --input required for compare mode")
            return
        visualize_comparison(args.data_dir, args.input, args.num_samples, args.save)


if __name__ == '__main__':
    main()
