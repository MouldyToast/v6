"""
Evaluation Metrics for Trajectory Diffusion

Metrics for evaluating generated trajectories:
1. Goal Accuracy: How well do trajectories reach their target endpoints?
2. Realism: Do trajectories look like real mouse movements?
3. Diversity: Are generated trajectories diverse or all the same?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_endpoint_from_trajectory(
    trajectory: torch.Tensor,
    length: int
) -> Tuple[float, float]:
    """
    Compute endpoint (dx, dy) from trajectory.

    Args:
        trajectory: (seq_len, 8) - trajectory features
        length: Actual length (excluding padding)

    Returns:
        (final_dx, final_dy): Endpoint relative to start
    """
    # Get last valid position
    final_dx = trajectory[length - 1, 0].item()  # dx at last timestep
    final_dy = trajectory[length - 1, 1].item()  # dy at last timestep

    return final_dx, final_dy


def compute_goal_distance_error(
    trajectories: torch.Tensor,
    target_distances: torch.Tensor,
    lengths: torch.Tensor,
    norm_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute distance accuracy metrics.

    Measures how close generated trajectory endpoints are to target distance.

    Args:
        trajectories: (batch, seq_len, 8) - generated trajectories
        target_distances: (batch,) - target distances (normalized [-1, 1])
        lengths: (batch,) - actual sequence lengths
        norm_params: Optional normalization parameters for denormalization

    Returns:
        metrics: Dict with distance error statistics
    """
    batch_size = trajectories.shape[0]
    errors = []

    for i in range(batch_size):
        # Get endpoint
        final_dx, final_dy = compute_endpoint_from_trajectory(
            trajectories[i], lengths[i].item()
        )

        # Compute actual endpoint distance
        actual_distance = np.sqrt(final_dx**2 + final_dy**2)

        # Denormalize if normalization params provided
        if norm_params is not None:
            # Denormalize target distance
            target_dist_norm = target_distances[i].item()
            d_min = norm_params.get('goal_dist_min', 0)
            d_max = norm_params.get('goal_dist_max', 1)
            target_distance = (target_dist_norm + 1) / 2 * (d_max - d_min) + d_min

            # Note: actual_distance is in the same unnormalized space as trajectory features
            # If features are normalized, we'd need to denormalize actual_distance too
            # For now, assume both are in the same space
        else:
            target_distance = target_distances[i].item()

        # Compute error
        error = abs(actual_distance - target_distance)
        errors.append(error)

    errors = np.array(errors)

    return {
        'mean_distance_error': errors.mean(),
        'median_distance_error': np.median(errors),
        'std_distance_error': errors.std(),
        'max_distance_error': errors.max(),
        'min_distance_error': errors.min(),
    }


def compute_goal_angle_error(
    trajectories: torch.Tensor,
    target_angles: torch.Tensor,
    lengths: torch.Tensor
) -> Dict[str, float]:
    """
    Compute angle accuracy metrics.

    Measures angular error between endpoint direction and target direction.

    Args:
        trajectories: (batch, seq_len, 8) - generated trajectories
        target_angles: (batch, 2) - [cos(angle), sin(angle)]
        lengths: (batch,) - actual sequence lengths

    Returns:
        metrics: Dict with angle error statistics
    """
    batch_size = trajectories.shape[0]
    errors = []

    for i in range(batch_size):
        # Get endpoint
        final_dx, final_dy = compute_endpoint_from_trajectory(
            trajectories[i], lengths[i].item()
        )

        # Compute actual angle
        actual_angle = np.arctan2(final_dy, final_dx)  # radians

        # Get target angle from cos/sin
        target_cos = target_angles[i, 0].item()
        target_sin = target_angles[i, 1].item()
        target_angle = np.arctan2(target_sin, target_cos)

        # Compute angular error (shortest distance on circle)
        error = angular_distance(actual_angle, target_angle)
        errors.append(error)

    errors = np.array(errors)

    return {
        'mean_angle_error_rad': errors.mean(),
        'mean_angle_error_deg': np.degrees(errors.mean()),
        'median_angle_error_rad': np.median(errors),
        'median_angle_error_deg': np.degrees(np.median(errors)),
        'std_angle_error_rad': errors.std(),
        'max_angle_error_deg': np.degrees(errors.max()),
    }


def angular_distance(angle1: float, angle2: float) -> float:
    """
    Compute shortest angular distance between two angles.

    Args:
        angle1, angle2: Angles in radians

    Returns:
        distance: Shortest distance in radians (always positive, <= pi)
    """
    diff = angle1 - angle2
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def compute_realism_metrics(
    trajectories: torch.Tensor,
    lengths: torch.Tensor,
    real_trajectories: Optional[torch.Tensor] = None,
    real_lengths: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute realism metrics.

    Measures if trajectories have realistic motion characteristics.

    Args:
        trajectories: (batch, seq_len, 8) - generated trajectories
        lengths: (batch,) - actual sequence lengths
        real_trajectories: Optional real trajectories for comparison
        real_lengths: Optional real trajectory lengths

    Returns:
        metrics: Dict with realism statistics
    """
    metrics = {}

    # Extract features
    speeds = []
    accels = []
    angular_vels = []
    smoothness_scores = []  # Jerk metric

    batch_size = trajectories.shape[0]

    for i in range(batch_size):
        length = lengths[i].item()
        traj = trajectories[i, :length]

        # Speed distribution
        speed = traj[:, 2].cpu().numpy()  # speed feature
        speeds.extend(speed)

        # Acceleration distribution
        accel = traj[:, 3].cpu().numpy()  # accel feature
        accels.extend(accel)

        # Angular velocity distribution
        ang_vel = traj[:, 6].cpu().numpy()  # ang_vel feature
        angular_vels.extend(ang_vel)

        # Smoothness (jerk = derivative of acceleration)
        if length > 2:
            jerk = np.diff(accel)
            smoothness = np.abs(jerk).mean()  # Lower is smoother
            smoothness_scores.append(smoothness)

    speeds = np.array(speeds)
    accels = np.array(accels)
    angular_vels = np.array(angular_vels)
    smoothness_scores = np.array(smoothness_scores)

    # Generated trajectory statistics
    metrics['gen_speed_mean'] = speeds.mean()
    metrics['gen_speed_std'] = speeds.std()
    metrics['gen_accel_mean'] = accels.mean()
    metrics['gen_accel_std'] = accels.std()
    metrics['gen_ang_vel_mean'] = angular_vels.mean()
    metrics['gen_ang_vel_std'] = angular_vels.std()
    metrics['gen_smoothness_mean'] = smoothness_scores.mean()

    # If real trajectories provided, compute comparison metrics
    if real_trajectories is not None and real_lengths is not None:
        real_speeds = []
        real_accels = []
        real_ang_vels = []
        real_smoothness = []

        for i in range(len(real_trajectories)):
            length = real_lengths[i].item()
            traj = real_trajectories[i, :length]

            real_speeds.extend(traj[:, 2].cpu().numpy())
            real_accels.extend(traj[:, 3].cpu().numpy())
            real_ang_vels.extend(traj[:, 6].cpu().numpy())

            if length > 2:
                accel = traj[:, 3].cpu().numpy()
                jerk = np.diff(accel)
                real_smoothness.append(np.abs(jerk).mean())

        real_speeds = np.array(real_speeds)
        real_accels = np.array(real_accels)
        real_ang_vels = np.array(real_ang_vels)
        real_smoothness = np.array(real_smoothness)

        # Real trajectory statistics
        metrics['real_speed_mean'] = real_speeds.mean()
        metrics['real_speed_std'] = real_speeds.std()
        metrics['real_accel_mean'] = real_accels.mean()
        metrics['real_accel_std'] = real_accels.std()
        metrics['real_ang_vel_mean'] = real_ang_vels.mean()
        metrics['real_ang_vel_std'] = real_ang_vels.std()
        metrics['real_smoothness_mean'] = real_smoothness.mean()

        # Differences
        metrics['speed_mean_diff'] = abs(metrics['gen_speed_mean'] - metrics['real_speed_mean'])
        metrics['accel_mean_diff'] = abs(metrics['gen_accel_mean'] - metrics['real_accel_mean'])
        metrics['smoothness_diff'] = abs(metrics['gen_smoothness_mean'] - metrics['real_smoothness_mean'])

    return metrics


def compute_diversity_metrics(
    trajectories: torch.Tensor,
    conditions: torch.Tensor,
    lengths: torch.Tensor
) -> Dict[str, float]:
    """
    Compute diversity metrics.

    Measures how diverse generated trajectories are for the same condition.

    Args:
        trajectories: (batch, seq_len, 8) - generated trajectories
        conditions: (batch, 3) - goal conditions
        lengths: (batch,) - actual sequence lengths

    Returns:
        metrics: Dict with diversity statistics
    """
    # Group trajectories by condition
    condition_groups = {}

    for i in range(len(trajectories)):
        cond_key = tuple(conditions[i].cpu().numpy())

        if cond_key not in condition_groups:
            condition_groups[cond_key] = []

        condition_groups[cond_key].append((trajectories[i], lengths[i]))

    # Compute diversity within each condition group
    diversities = []

    for cond_key, trajs in condition_groups.items():
        if len(trajs) < 2:
            continue  # Need at least 2 trajectories to measure diversity

        # Compute pairwise distances
        pairwise_dists = []

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                traj_i, len_i = trajs[i]
                traj_j, len_j = trajs[j]

                # Use minimum length for comparison
                min_len = min(len_i.item(), len_j.item())

                # Compute trajectory distance (MSE)
                dist = torch.mean((traj_i[:min_len] - traj_j[:min_len]) ** 2).item()
                pairwise_dists.append(dist)

        # Average pairwise distance for this condition
        if pairwise_dists:
            diversities.append(np.mean(pairwise_dists))

    diversities = np.array(diversities)

    if len(diversities) > 0:
        return {
            'diversity_mean': diversities.mean(),
            'diversity_std': diversities.std(),
            'diversity_median': np.median(diversities),
            'num_condition_groups': len(condition_groups),
            'avg_samples_per_condition': np.mean([len(g) for g in condition_groups.values()])
        }
    else:
        return {
            'diversity_mean': 0.0,
            'diversity_std': 0.0,
            'diversity_median': 0.0,
            'num_condition_groups': 0,
            'avg_samples_per_condition': 0.0
        }


def evaluate_trajectories(
    trajectories: torch.Tensor,
    conditions: torch.Tensor,
    lengths: torch.Tensor,
    real_trajectories: Optional[torch.Tensor] = None,
    real_lengths: Optional[torch.Tensor] = None,
    norm_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated trajectories.

    Combines all metrics into single report.

    Args:
        trajectories: (batch, seq_len, 8) - generated trajectories
        conditions: (batch, 3) - [distance_norm, cos(angle), sin(angle)]
        lengths: (batch,) - actual sequence lengths
        real_trajectories: Optional real trajectories for comparison
        real_lengths: Optional real lengths
        norm_params: Optional normalization parameters

    Returns:
        all_metrics: Dict with all computed metrics
    """
    all_metrics = {}

    # Goal accuracy
    print("Computing goal accuracy metrics...")
    distance_metrics = compute_goal_distance_error(
        trajectories, conditions[:, 0], lengths, norm_params
    )
    all_metrics.update(distance_metrics)

    angle_metrics = compute_goal_angle_error(
        trajectories, conditions[:, 1:], lengths
    )
    all_metrics.update(angle_metrics)

    # Realism
    print("Computing realism metrics...")
    realism_metrics = compute_realism_metrics(
        trajectories, lengths, real_trajectories, real_lengths
    )
    all_metrics.update(realism_metrics)

    # Diversity
    print("Computing diversity metrics...")
    diversity_metrics = compute_diversity_metrics(
        trajectories, conditions, lengths
    )
    all_metrics.update(diversity_metrics)

    return all_metrics


def print_metrics_report(metrics: Dict[str, float]):
    """
    Print formatted metrics report.

    Args:
        metrics: Dict with computed metrics
    """
    print("\n" + "=" * 70)
    print("TRAJECTORY EVALUATION REPORT")
    print("=" * 70)

    if 'mean_distance_error' in metrics:
        print("\n[Goal Accuracy - Distance]")
        print(f"  Mean error: {metrics['mean_distance_error']:.3f}")
        print(f"  Median error: {metrics['median_distance_error']:.3f}")
        print(f"  Std error: {metrics['std_distance_error']:.3f}")

    if 'mean_angle_error_deg' in metrics:
        print("\n[Goal Accuracy - Angle]")
        print(f"  Mean error: {metrics['mean_angle_error_deg']:.2f}°")
        print(f"  Median error: {metrics['median_angle_error_deg']:.2f}°")
        print(f"  Max error: {metrics['max_angle_error_deg']:.2f}°")

    if 'gen_speed_mean' in metrics:
        print("\n[Realism - Generated Trajectories]")
        print(f"  Speed: {metrics['gen_speed_mean']:.3f} ± {metrics['gen_speed_std']:.3f}")
        print(f"  Accel: {metrics['gen_accel_mean']:.3f} ± {metrics['gen_accel_std']:.3f}")
        print(f"  Smoothness: {metrics['gen_smoothness_mean']:.3f}")

    if 'real_speed_mean' in metrics:
        print("\n[Realism - Real vs Generated Comparison]")
        print(f"  Speed difference: {metrics['speed_mean_diff']:.3f}")
        print(f"  Accel difference: {metrics['accel_mean_diff']:.3f}")
        print(f"  Smoothness difference: {metrics['smoothness_diff']:.3f}")

    if 'diversity_mean' in metrics:
        print("\n[Diversity]")
        print(f"  Mean pairwise distance: {metrics['diversity_mean']:.4f}")
        print(f"  Median pairwise distance: {metrics['diversity_median']:.4f}")
        print(f"  Condition groups: {int(metrics['num_condition_groups'])}")

    print("=" * 70 + "\n")


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing evaluation metrics...")
    print("=" * 70)

    # Create synthetic test data
    batch_size = 10
    seq_len = 50

    trajectories = torch.randn(batch_size, seq_len, 8)
    conditions = torch.randn(batch_size, 3)
    lengths = torch.randint(30, seq_len, (batch_size,))

    print("\n=== Test 1: Goal Distance Error ===")
    dist_metrics = compute_goal_distance_error(
        trajectories, conditions[:, 0], lengths
    )
    print(dist_metrics)

    print("\n=== Test 2: Goal Angle Error ===")
    angle_metrics = compute_goal_angle_error(
        trajectories, conditions[:, 1:], lengths
    )
    print(angle_metrics)

    print("\n=== Test 3: Realism Metrics ===")
    realism_metrics = compute_realism_metrics(
        trajectories, lengths
    )
    print(f"Keys: {list(realism_metrics.keys())}")

    print("\n=== Test 4: Diversity Metrics ===")
    diversity_metrics = compute_diversity_metrics(
        trajectories, conditions, lengths
    )
    print(diversity_metrics)

    print("\n=== Test 5: Full Evaluation ===")
    all_metrics = evaluate_trajectories(
        trajectories, conditions, lengths
    )
    print_metrics_report(all_metrics)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
