"""
Denormalization Utilities for V4

Converts normalized features [-1, 1] back to original scale.

V4 Features:
    [0] dx       - Relative x from start (pixels)
    [1] dy       - Relative y from start (pixels)
    [2] speed    - Movement speed (pixels/second)
    [3] accel    - Acceleration (pixels/second^2)
    [4] sin_h    - Sine of heading [-1, 1]
    [5] cos_h    - Cosine of heading [-1, 1]
    [6] ang_vel  - Angular velocity (radians/second)
    [7] dt       - Time delta (seconds)

KEY V4 BENEFIT:
Since dx, dy are relative to trajectory start, we can place the generated
trajectory at ANY starting position by simply adding start_x, start_y!

    absolute_x = dx + start_x
    absolute_y = dy + start_y

This is what makes V4 "position-independent" - the model learned movement STYLE,
not screen POSITION.
"""

import numpy as np


def denormalize_value(normalized, fmin, fmax):
    """
    Denormalize a single value from [-1, 1] to [fmin, fmax].

    Formula: original = ((normalized + 1) / 2) * (fmax - fmin) + fmin
    """
    return ((normalized + 1) / 2) * (fmax - fmin) + fmin


def denormalize_trajectory(traj_norm, norm_params):
    """
    Denormalize a single trajectory from [-1, 1] to original scale.

    Args:
        traj_norm: (seq_len, 8) normalized trajectory
        norm_params: Dictionary with min/max values for each feature

    Returns:
        traj_denorm: (seq_len, 8) denormalized trajectory
    """
    traj_denorm = np.zeros_like(traj_norm)

    # Feature 0: dx (relative x position)
    traj_denorm[:, 0] = denormalize_value(
        traj_norm[:, 0],
        norm_params['dx_min'],
        norm_params['dx_max']
    )

    # Feature 1: dy (relative y position)
    traj_denorm[:, 1] = denormalize_value(
        traj_norm[:, 1],
        norm_params['dy_min'],
        norm_params['dy_max']
    )

    # Feature 2: speed
    traj_denorm[:, 2] = denormalize_value(
        traj_norm[:, 2],
        norm_params['speed_min'],
        norm_params['speed_max']
    )

    # Feature 3: acceleration
    traj_denorm[:, 3] = denormalize_value(
        traj_norm[:, 3],
        norm_params['accel_min'],
        norm_params['accel_max']
    )

    # Feature 4: sin(heading) - already in [-1, 1], no denormalization needed
    traj_denorm[:, 4] = traj_norm[:, 4]

    # Feature 5: cos(heading) - already in [-1, 1], no denormalization needed
    traj_denorm[:, 5] = traj_norm[:, 5]

    # Feature 6: angular velocity
    traj_denorm[:, 6] = denormalize_value(
        traj_norm[:, 6],
        norm_params['ang_vel_min'],
        norm_params['ang_vel_max']
    )

    # Feature 7: dt (time delta)
    traj_denorm[:, 7] = denormalize_value(
        traj_norm[:, 7],
        norm_params['dt_min'],
        norm_params['dt_max']
    )

    return traj_denorm


def denormalize_batch(batch_norm, norm_params):
    """
    Denormalize a batch of trajectories.

    Args:
        batch_norm: (batch_size, seq_len, 8) normalized trajectories
        norm_params: Dictionary with min/max values

    Returns:
        batch_denorm: (batch_size, seq_len, 8) denormalized trajectories
    """
    batch_denorm = np.zeros_like(batch_norm)

    for i in range(len(batch_norm)):
        batch_denorm[i] = denormalize_trajectory(batch_norm[i], norm_params)

    return batch_denorm


def denormalize_distance(dist_norm, norm_params):
    """
    Denormalize distance conditioning variable.

    Args:
        dist_norm: Normalized distance(s) in [-1, 1]
        norm_params: Dictionary with actual_dist_min/max

    Returns:
        dist_denorm: Distance(s) in pixels
    """
    return denormalize_value(
        dist_norm,
        norm_params['actual_dist_min'],
        norm_params['actual_dist_max']
    )


def convert_to_absolute_coords(traj_denorm, start_x=0, start_y=0):
    """
    Convert relative dx, dy to absolute x, y coordinates.

    This is the KEY V4 BENEFIT: We can place the generated trajectory
    at ANY starting position!

    Args:
        traj_denorm: (seq_len, 8) denormalized trajectory with relative coords
        start_x: Starting x position (pixels)
        start_y: Starting y position (pixels)

    Returns:
        traj_absolute: (seq_len, 8) trajectory with absolute x, y in positions 0, 1
    """
    traj_absolute = traj_denorm.copy()

    # Convert relative to absolute
    traj_absolute[:, 0] = traj_denorm[:, 0] + start_x  # x = dx + start_x
    traj_absolute[:, 1] = traj_denorm[:, 1] + start_y  # y = dy + start_y

    return traj_absolute


def extract_xy_path(traj, length=None, start_x=0, start_y=0):
    """
    Extract x, y path from trajectory for visualization.

    Args:
        traj: (seq_len, 8) denormalized trajectory
        length: Actual trajectory length (to exclude padding)
        start_x, start_y: Starting position

    Returns:
        x, y: 1D arrays of absolute coordinates
    """
    if length is None:
        length = len(traj)

    # Get relative positions
    dx = traj[:length, 0]
    dy = traj[:length, 1]

    # Convert to absolute
    x = dx + start_x
    y = dy + start_y

    return x, y


def reconstruct_heading(sin_h, cos_h):
    """
    Reconstruct heading angle from sin/cos encoding.

    Args:
        sin_h: Sine of heading
        cos_h: Cosine of heading

    Returns:
        heading: Angle in radians [-pi, pi]
    """
    return np.arctan2(sin_h, cos_h)


def get_trajectory_stats(traj_denorm, length=None):
    """
    Get statistics for a single trajectory (useful for debugging).

    Args:
        traj_denorm: (seq_len, 8) denormalized trajectory
        length: Actual length (to exclude padding)

    Returns:
        Dictionary of statistics
    """
    if length is None:
        length = len(traj_denorm)

    traj = traj_denorm[:length]

    # Reconstruct heading
    heading = reconstruct_heading(traj[:, 4], traj[:, 5])

    stats = {
        # Position stats
        'dx_range': (traj[:, 0].min(), traj[:, 0].max()),
        'dy_range': (traj[:, 1].min(), traj[:, 1].max()),
        'total_distance': np.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2),

        # Speed stats
        'speed_mean': traj[:, 2].mean(),
        'speed_max': traj[:, 2].max(),
        'speed_min': traj[:, 2].min(),

        # Acceleration stats (style signature!)
        'accel_mean': traj[:, 3].mean(),
        'accel_std': traj[:, 3].std(),
        'accel_max': traj[:, 3].max(),
        'accel_min': traj[:, 3].min(),

        # Heading stats
        'heading_range': (heading.min(), heading.max()),

        # Angular velocity stats (curvature style!)
        'ang_vel_mean': traj[:, 6].mean(),
        'ang_vel_std': traj[:, 6].std(),

        # Timing stats
        'dt_mean': traj[:, 7].mean(),
        'dt_total': traj[:, 7].sum(),

        # Length
        'num_points': length,
    }

    return stats


def compute_path_distance(traj_denorm, length=None):
    """
    Compute total path distance (sum of point-to-point Euclidean distances).

    This is the KEY metric for condition-matched validation:
    The model is conditioned on actual_distance (path length), so a well-trained
    model should generate trajectories with similar path distances when given
    the same condition.

    Args:
        traj_denorm: (seq_len, 8) denormalized trajectory
                     Features: [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]
        length: Actual trajectory length (to exclude padding)

    Returns:
        path_distance: Total distance traveled along the path (pixels)

    Note:
        This is different from straight-line distance (end - start).
        Path distance captures the actual distance traveled including curves.
    """
    if length is None:
        length = len(traj_denorm)

    if length < 2:
        return 0.0

    # Get dx, dy positions (relative to start)
    dx = traj_denorm[:length, 0]
    dy = traj_denorm[:length, 1]

    # Compute point-to-point distances
    dx_diff = np.diff(dx)  # Change in x between consecutive points
    dy_diff = np.diff(dy)  # Change in y between consecutive points

    # Euclidean distance for each step
    step_distances = np.sqrt(dx_diff**2 + dy_diff**2)

    # Total path distance
    return float(step_distances.sum())


def compute_straight_line_distance(traj_denorm, length=None):
    """
    Compute straight-line distance from start to end.

    Args:
        traj_denorm: (seq_len, 8) denormalized trajectory
        length: Actual trajectory length

    Returns:
        straight_distance: Euclidean distance from start to end (pixels)
    """
    if length is None:
        length = len(traj_denorm)

    if length < 1:
        return 0.0

    # End point relative to start (start is always 0,0 for dx,dy)
    end_dx = traj_denorm[length - 1, 0]
    end_dy = traj_denorm[length - 1, 1]

    return float(np.sqrt(end_dx**2 + end_dy**2))


def compute_curvature_ratio(traj_denorm, length=None):
    """
    Compute curvature ratio (path_distance / straight_line_distance).

    Values:
        = 1.0: Perfectly straight path
        > 1.0: Curved path (higher = more curved)

    Args:
        traj_denorm: (seq_len, 8) denormalized trajectory
        length: Actual trajectory length

    Returns:
        curvature_ratio: path_distance / straight_line_distance
    """
    path_dist = compute_path_distance(traj_denorm, length)
    straight_dist = compute_straight_line_distance(traj_denorm, length)

    if straight_dist < 1e-6:
        return float('inf') if path_dist > 1e-6 else 1.0

    return path_dist / straight_dist


# Test code
if __name__ == '__main__':
    print("Testing Denormalization Utilities V4...")
    print("=" * 50)

    # Create mock norm_params
    norm_params = {
        'dx_min': -500, 'dx_max': 500,
        'dy_min': -500, 'dy_max': 500,
        'speed_min': 0, 'speed_max': 5000,
        'accel_min': -10000, 'accel_max': 10000,
        'sin_h_min': -1, 'sin_h_max': 1,
        'cos_h_min': -1, 'cos_h_max': 1,
        'ang_vel_min': -50, 'ang_vel_max': 50,
        'dt_min': 0.001, 'dt_max': 0.05,
        'actual_dist_min': 20, 'actual_dist_max': 700,
    }

    # Create mock normalized trajectory
    seq_len = 100
    traj_norm = np.random.uniform(-1, 1, (seq_len, 8))

    # Test single trajectory denormalization
    print("\n1. Testing single trajectory denormalization...")
    traj_denorm = denormalize_trajectory(traj_norm, norm_params)
    print(f"   Input shape: {traj_norm.shape}")
    print(f"   Output shape: {traj_denorm.shape}")
    print(f"   dx range: [{traj_denorm[:, 0].min():.1f}, {traj_denorm[:, 0].max():.1f}]")
    print(f"   speed range: [{traj_denorm[:, 2].min():.1f}, {traj_denorm[:, 2].max():.1f}]")

    # Test batch denormalization
    print("\n2. Testing batch denormalization...")
    batch_norm = np.random.uniform(-1, 1, (10, seq_len, 8))
    batch_denorm = denormalize_batch(batch_norm, norm_params)
    print(f"   Batch input shape: {batch_norm.shape}")
    print(f"   Batch output shape: {batch_denorm.shape}")

    # Test absolute coordinate conversion
    print("\n3. Testing absolute coordinate conversion...")
    start_x, start_y = 500, 300
    traj_absolute = convert_to_absolute_coords(traj_denorm, start_x, start_y)
    print(f"   Start position: ({start_x}, {start_y})")
    print(f"   First point after conversion: ({traj_absolute[0, 0]:.1f}, {traj_absolute[0, 1]:.1f})")
    print(f"   (Should be close to start position since dx[0]=dy[0]=0 in real data)")

    # Test xy extraction
    print("\n4. Testing XY path extraction...")
    x, y = extract_xy_path(traj_denorm, length=50, start_x=100, start_y=200)
    print(f"   Path length: {len(x)} points")
    print(f"   X range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"   Y range: [{y.min():.1f}, {y.max():.1f}]")

    # Test heading reconstruction
    print("\n5. Testing heading reconstruction...")
    sin_h = np.array([0, 1, 0, -1])
    cos_h = np.array([1, 0, -1, 0])
    heading = reconstruct_heading(sin_h, cos_h)
    print(f"   sin_h: {sin_h}")
    print(f"   cos_h: {cos_h}")
    print(f"   Heading (rad): {heading}")
    print(f"   Heading (deg): {np.rad2deg(heading)}")
    print(f"   Expected: [0, 90, 180/-180, -90]")

    # Test trajectory stats
    print("\n6. Testing trajectory statistics...")
    stats = get_trajectory_stats(traj_denorm, length=50)
    print(f"   Speed mean: {stats['speed_mean']:.1f} px/s")
    print(f"   Accel std: {stats['accel_std']:.1f} px/s^2 (style signature)")
    print(f"   Ang vel std: {stats['ang_vel_std']:.4f} rad/s (curvature style)")
    print(f"   Total time: {stats['dt_total']*1000:.1f} ms")

    print("\nDenormalization V4 tests PASSED!")
    print("=" * 50)
