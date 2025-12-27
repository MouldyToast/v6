"""
Visualization Utilities for Trajectory Diffusion

Functions for plotting and visualizing generated trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple
from pathlib import Path


def plot_trajectory(
    trajectory: np.ndarray,
    length: int,
    title: str = "Trajectory",
    color: str = 'blue',
    alpha: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show_endpoints: bool = True,
    show_grid: bool = True
) -> plt.Axes:
    """
    Plot a single trajectory.

    Args:
        trajectory: (seq_len, 8) - trajectory features
        length: Actual length (excluding padding)
        title: Plot title
        color: Line color
        alpha: Line transparency
        ax: Matplotlib axes (creates new if None)
        show_endpoints: Show start/end markers
        show_grid: Show grid

    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Extract positions (cumulative dx, dy)
    # trajectory[:, 0] is dx, trajectory[:, 1] is dy
    # We need to compute cumulative positions
    dx = trajectory[:length, 0]
    dy = trajectory[:length, 1]

    # Compute absolute positions
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    # Add starting point at origin
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])

    # Plot trajectory
    ax.plot(x, y, color=color, alpha=alpha, linewidth=2)

    if show_endpoints:
        # Start point
        ax.scatter([0], [0], color='green', s=100, marker='o',
                   label='Start', zorder=5, edgecolors='black', linewidths=1.5)
        # End point
        ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='X',
                   label='End', zorder=5, edgecolors='black', linewidths=1.5)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.set_aspect('equal')

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_endpoints:
        ax.legend()

    return ax


def plot_multiple_trajectories(
    trajectories: np.ndarray,
    lengths: np.ndarray,
    max_plot: int = 10,
    title: str = "Generated Trajectories",
    save_path: Optional[str] = None
):
    """
    Plot multiple trajectories on same axes.

    Args:
        trajectories: (batch, seq_len, 8)
        lengths: (batch,)
        max_plot: Maximum number to plot
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    num_plot = min(max_plot, len(trajectories))
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))

    for i in range(num_plot):
        traj = trajectories[i]
        length = int(lengths[i])

        dx = traj[:length, 0]
        dy = traj[:length, 1]

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        x = np.concatenate([[0], x])
        y = np.concatenate([[0], y])

        ax.plot(x, y, color=colors[i], alpha=0.6, linewidth=1.5)

    # Mark common start
    ax.scatter([0], [0], color='green', s=150, marker='o',
               label='Start', zorder=5, edgecolors='black', linewidths=2)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"{title} (n={num_plot})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_trajectory_comparison(
    gen_trajectory: np.ndarray,
    real_trajectory: np.ndarray,
    gen_length: int,
    real_length: int,
    title: str = "Generated vs Real",
    save_path: Optional[str] = None
):
    """
    Plot generated and real trajectory side by side.

    Args:
        gen_trajectory: (seq_len, 8) - generated
        real_trajectory: (seq_len, 8) - real
        gen_length: Generated trajectory length
        real_length: Real trajectory length
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot generated
    plot_trajectory(gen_trajectory, gen_length, "Generated",
                    color='blue', ax=ax1)

    # Plot real
    plot_trajectory(real_trajectory, real_length, "Real",
                    color='orange', ax=ax2)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_feature_distributions(
    trajectories: np.ndarray,
    lengths: np.ndarray,
    real_trajectories: Optional[np.ndarray] = None,
    real_lengths: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot distributions of trajectory features.

    Args:
        trajectories: (batch, seq_len, 8) - generated
        lengths: (batch,)
        real_trajectories: Optional real trajectories for comparison
        real_lengths: Optional real lengths
        save_path: Path to save figure
    """
    # Feature names
    feature_names = ['dx', 'dy', 'speed', 'accel', 'distance',
                     'direction_x', 'ang_vel', 'ang_accel']

    # Extract all valid features
    gen_features = []
    for i in range(len(trajectories)):
        length = int(lengths[i])
        gen_features.append(trajectories[i, :length])
    gen_features = np.concatenate(gen_features, axis=0)  # (total_timesteps, 8)

    # Create subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig)

    for feat_idx in range(8):
        ax = fig.add_subplot(gs[feat_idx // 4, feat_idx % 4])

        # Plot generated distribution
        gen_vals = gen_features[:, feat_idx]
        ax.hist(gen_vals, bins=50, alpha=0.6, color='blue',
                label='Generated', density=True)

        # Plot real distribution if provided
        if real_trajectories is not None and real_lengths is not None:
            real_features = []
            for i in range(len(real_trajectories)):
                length = int(real_lengths[i])
                real_features.append(real_trajectories[i, :length])
            real_features = np.concatenate(real_features, axis=0)

            real_vals = real_features[:, feat_idx]
            ax.hist(real_vals, bins=50, alpha=0.6, color='orange',
                    label='Real', density=True)

        ax.set_xlabel(feature_names[feat_idx])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Feature Distributions: Generated vs Real',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_diversity_by_condition(
    trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    num_conditions: int = 3,
    samples_per_condition: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot multiple samples for the same condition to visualize diversity.

    Args:
        trajectories: (batch, seq_len, 8)
        conditions: (batch, 3) - [distance_norm, cos(angle), sin(angle)]
        lengths: (batch,)
        num_conditions: Number of different conditions to plot
        samples_per_condition: Number of samples to show per condition
        save_path: Path to save figure
    """
    # Group trajectories by condition
    condition_groups = {}
    for i in range(len(trajectories)):
        cond_key = tuple(conditions[i])
        if cond_key not in condition_groups:
            condition_groups[cond_key] = []
        condition_groups[cond_key].append((trajectories[i], lengths[i]))

    # Select conditions with enough samples
    valid_conditions = [k for k, v in condition_groups.items()
                        if len(v) >= samples_per_condition]

    if len(valid_conditions) == 0:
        print("Warning: No conditions with enough samples for diversity plot")
        return None

    num_plot = min(num_conditions, len(valid_conditions))
    selected_conditions = np.random.choice(len(valid_conditions),
                                           size=num_plot, replace=False)

    # Create subplots
    fig, axes = plt.subplots(1, num_plot, figsize=(6 * num_plot, 5))
    if num_plot == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, samples_per_condition))

    for plot_idx, cond_idx in enumerate(selected_conditions):
        ax = axes[plot_idx]
        cond_key = valid_conditions[cond_idx]
        samples = condition_groups[cond_key][:samples_per_condition]

        # Extract target from condition
        distance_norm, cos_angle, sin_angle = cond_key
        target_angle = np.arctan2(sin_angle, cos_angle)

        for sample_idx, (traj, length) in enumerate(samples):
            dx = traj[:int(length), 0]
            dy = traj[:int(length), 1]

            x = np.cumsum(dx)
            y = np.cumsum(dy)
            x = np.concatenate([[0], x])
            y = np.concatenate([[0], y])

            ax.plot(x, y, color=colors[sample_idx], alpha=0.7,
                    linewidth=2, label=f'Sample {sample_idx + 1}')

        # Mark start
        ax.scatter([0], [0], color='green', s=150, marker='o',
                   zorder=5, edgecolors='black', linewidths=2)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f"Condition {plot_idx + 1}\nAngle: {np.degrees(target_angle):.0f}°")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f'Diversity: {samples_per_condition} Samples per Condition',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_goal_accuracy_scatter(
    trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Scatter plot showing goal accuracy (target vs achieved).

    Args:
        trajectories: (batch, seq_len, 8)
        conditions: (batch, 3) - [distance_norm, cos(angle), sin(angle)]
        lengths: (batch,)
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    target_distances = []
    actual_distances = []
    target_angles = []
    actual_angles = []

    for i in range(len(trajectories)):
        length = int(lengths[i])

        # Get endpoint
        final_dx = trajectories[i, length - 1, 0]
        final_dy = trajectories[i, length - 1, 1]

        # Actual endpoint
        actual_dist = np.sqrt(final_dx**2 + final_dy**2)
        actual_angle = np.arctan2(final_dy, final_dx)

        # Target endpoint
        target_dist = conditions[i, 0]  # Normalized distance
        target_cos = conditions[i, 1]
        target_sin = conditions[i, 2]
        target_angle = np.arctan2(target_sin, target_cos)

        target_distances.append(target_dist)
        actual_distances.append(actual_dist)
        target_angles.append(np.degrees(target_angle))
        actual_angles.append(np.degrees(actual_angle))

    target_distances = np.array(target_distances)
    actual_distances = np.array(actual_distances)
    target_angles = np.array(target_angles)
    actual_angles = np.array(actual_angles)

    # Distance scatter
    ax1.scatter(target_distances, actual_distances, alpha=0.5, s=20)

    # Perfect accuracy line
    dist_range = [min(target_distances.min(), actual_distances.min()),
                  max(target_distances.max(), actual_distances.max())]
    ax1.plot(dist_range, dist_range, 'r--', linewidth=2, label='Perfect')

    ax1.set_xlabel('Target Distance (normalized)')
    ax1.set_ylabel('Actual Distance')
    ax1.set_title('Distance Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Angle scatter
    ax2.scatter(target_angles, actual_angles, alpha=0.5, s=20)

    # Perfect accuracy line
    angle_range = [-180, 180]
    ax2.plot(angle_range, angle_range, 'r--', linewidth=2, label='Perfect')

    ax2.set_xlabel('Target Angle (degrees)')
    ax2.set_ylabel('Actual Angle (degrees)')
    ax2.set_title('Angle Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Goal Accuracy: Target vs Achieved',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def create_evaluation_report(
    trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    real_trajectories: Optional[np.ndarray] = None,
    real_lengths: Optional[np.ndarray] = None,
    output_dir: str = 'results/diffusion_v7/plots'
):
    """
    Create comprehensive visualization report.

    Generates all plots and saves them to output directory.

    Args:
        trajectories: (batch, seq_len, 8) - generated
        conditions: (batch, 3)
        lengths: (batch,)
        real_trajectories: Optional real trajectories
        real_lengths: Optional real lengths
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION REPORT")
    print("=" * 70)

    # 1. Multiple trajectories
    print("\n[1/5] Plotting multiple trajectories...")
    plot_multiple_trajectories(
        trajectories, lengths, max_plot=20,
        save_path=output_path / 'trajectories_overview.png'
    )
    plt.close()

    # 2. Feature distributions
    print("[2/5] Plotting feature distributions...")
    plot_feature_distributions(
        trajectories, lengths, real_trajectories, real_lengths,
        save_path=output_path / 'feature_distributions.png'
    )
    plt.close()

    # 3. Diversity
    print("[3/5] Plotting diversity by condition...")
    plot_diversity_by_condition(
        trajectories, conditions, lengths,
        num_conditions=3, samples_per_condition=5,
        save_path=output_path / 'diversity.png'
    )
    plt.close()

    # 4. Goal accuracy
    print("[4/5] Plotting goal accuracy...")
    plot_goal_accuracy_scatter(
        trajectories, conditions, lengths,
        save_path=output_path / 'goal_accuracy.png'
    )
    plt.close()

    # 5. Individual trajectory examples
    print("[5/5] Plotting individual examples...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        if i < len(trajectories):
            plot_trajectory(
                trajectories[i], int(lengths[i]),
                title=f"Example {i + 1}",
                ax=axes[i], color='blue'
            )

    plt.tight_layout()
    plt.savefig(output_path / 'individual_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path / 'individual_examples.png'}")

    print("\n✓ Report complete!")
    print(f"  All plots saved to: {output_path}")
    print("=" * 70 + "\n")


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing visualization utilities...")
    print("=" * 70)

    # Create synthetic test data
    batch_size = 20
    seq_len = 100

    trajectories = np.random.randn(batch_size, seq_len, 8) * 0.1
    lengths = np.random.randint(50, seq_len, size=batch_size)
    conditions = np.random.randn(batch_size, 3)

    print("\n=== Test 1: Single Trajectory ===")
    fig = plt.figure()
    plot_trajectory(trajectories[0], int(lengths[0]), "Test Trajectory")
    plt.savefig('test_single_trajectory.png')
    plt.close()
    print("✓ Saved test_single_trajectory.png")

    print("\n=== Test 2: Multiple Trajectories ===")
    plot_multiple_trajectories(trajectories, lengths, max_plot=10,
                                save_path='test_multiple_trajectories.png')
    plt.close()
    print("✓ Saved test_multiple_trajectories.png")

    print("\n=== Test 3: Feature Distributions ===")
    plot_feature_distributions(trajectories, lengths,
                                save_path='test_feature_distributions.png')
    plt.close()
    print("✓ Saved test_feature_distributions.png")

    print("\n=== Test 4: Goal Accuracy ===")
    plot_goal_accuracy_scatter(trajectories, conditions, lengths,
                                save_path='test_goal_accuracy.png')
    plt.close()
    print("✓ Saved test_goal_accuracy.png")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
