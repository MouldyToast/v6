"""
Phase 1: Data Preprocessing for V4 - Position-Independent Style Learning
TimeGAN v4.0 - COMPLETE FEATURE REDESIGN

==============================================================================
V4 KEY INNOVATION: POSITION-INDEPENDENT STYLE FEATURES
==============================================================================

V3 Problem: Used absolute screen coordinates (x, y) which meant:
  - Same movement at different screen positions = different training examples
  - Model learned WHERE movements happen, not HOW they happen
  - Generated trajectories were tied to training coordinate ranges

V4 Solution: Position-independent features that capture MOVEMENT STYLE:
  [dx, dy, speed, acceleration, sin(heading), cos(heading), angular_velocity, dt]

Feature Breakdown (8 features):
  1. dx: Relative x position from trajectory START (position-independent shape)
  2. dy: Relative y position from trajectory START (position-independent shape)
  3. speed: Movement magnitude in pixels/second (how fast you move)
  4. acceleration: d(speed)/dt (how you speed up/slow down - style signature!)
  5. sin(heading): Sine of direction angle (smooth direction encoding)
  6. cos(heading): Cosine of direction angle (no discontinuity at +/-180)
  7. angular_velocity: d(heading)/dt (turning rate - curvature style!)
  8. dt: Time delta between samples (timing patterns)

Why This Works:
  - Same movement shape at ANY screen position = SAME features
  - Model learns HOW you move, not WHERE you moved
  - Generated trajectories can start from ANY point
  - Acceleration captures your personal speed-up/slow-down patterns
  - Angular velocity captures your personal curving style

==============================================================================
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default directories (will be overridden by command line args)
DEFAULT_INPUT_DIR = 'raw_trajectories_adaptive'
DEFAULT_OUTPUT_DIR = 'processed_data_v4'

# Processing parameters
MAX_SEQ_LENGTH = 800  # For storage only; dynamic batching will re-pad
RANDOM_SEED = 42

# Train/val/test split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Feature count for V4
FEATURE_DIM = 8  # dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt


# ============================================================================
# STEP 1: LOAD RAW TRAJECTORIES
# ============================================================================

def load_trajectories(data_dir):
    """Load all JSON trajectory files from directory."""
    print(f"\n{'='*70}")
    print("STEP 1: Loading Raw Trajectories")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")

    trajectories = []
    json_files = list(Path(data_dir).glob('*.json'))

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Validate required fields
            required_fields = ['x', 'y', 't', 'ideal_distance', 'actual_distance']
            if not all(k in data for k in required_fields):
                print(f"  Skipping {json_file.name}: missing required fields")
                continue

            if not (len(data['x']) == len(data['y']) == len(data['t'])):
                print(f"  Skipping {json_file.name}: array length mismatch")
                continue

            # Need at least 3 points for acceleration/angular velocity
            if len(data['x']) < 3:
                print(f"  Skipping {json_file.name}: too few points ({len(data['x'])})")
                continue

            trajectories.append({
                'x': np.array(data['x'], dtype=np.float64),
                'y': np.array(data['y'], dtype=np.float64),
                't': np.array(data['t'], dtype=np.float64),
                'ideal_distance': float(data['ideal_distance']),
                'actual_distance': float(data['actual_distance']),
                'filename': json_file.name
            })

        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
            continue

    print(f"Successfully loaded {len(trajectories)} trajectories")

    if len(trajectories) == 0:
        raise ValueError(f"No valid trajectories found in {data_dir}")

    # Statistics
    point_counts = [len(traj['x']) for traj in trajectories]
    distances = [traj['ideal_distance'] for traj in trajectories]

    print(f"Point counts: min={min(point_counts)}, max={max(point_counts)}, "
          f"median={np.median(point_counts):.0f}")
    print(f"Distance range: {min(distances):.1f} to {max(distances):.1f} pixels")

    return trajectories


# ============================================================================
# STEP 1.5: FILTER BAD RECORDINGS
# ============================================================================

def filter_trajectories(trajectories, max_length=750, max_distance_ratio=2.5):
    """
    Filter out bad recordings that would harm training quality.

    Filter criteria:
    1. extracted_length > max_length: Would be truncated, breaking distance condition
    2. actual_distance / ideal_distance > max_distance_ratio: Erratic/distracted paths

    Args:
        trajectories: List of trajectory dicts from load_trajectories()
        max_length: Maximum allowed sequence length (default: 800)
        max_distance_ratio: Maximum actual/ideal distance ratio (default: 2.5)

    Returns:
        filtered_trajectories: Clean trajectories ready for processing
    """
    print(f"\n{'='*70}")
    print("STEP 1.5: Filtering Bad Recordings")
    print(f"{'='*70}")
    print(f"Filter criteria:")
    print(f"  1. Length > {max_length} points (would be truncated)")
    print(f"  2. Distance ratio > {max_distance_ratio}x (erratic/distracted path)")

    filtered_trajectories = []

    # Track filtered trajectories by reason
    filtered_too_long = []
    filtered_too_indirect = []
    filtered_both = []

    for traj in trajectories:
        extracted_length = len(traj['x'])
        distance_ratio = traj['actual_distance'] / traj['ideal_distance']

        too_long = extracted_length > max_length
        too_indirect = distance_ratio > max_distance_ratio

        if too_long and too_indirect:
            filtered_both.append({
                'filename': traj['filename'],
                'extracted_length': extracted_length,
                'distance_ratio': distance_ratio,
                'actual_distance': traj['actual_distance'],
                'ideal_distance': traj['ideal_distance']
            })
        elif too_long:
            filtered_too_long.append({
                'filename': traj['filename'],
                'extracted_length': extracted_length,
                'distance_ratio': distance_ratio,
                'actual_distance': traj['actual_distance'],
                'ideal_distance': traj['ideal_distance']
            })
        elif too_indirect:
            filtered_too_indirect.append({
                'filename': traj['filename'],
                'extracted_length': extracted_length,
                'distance_ratio': distance_ratio,
                'actual_distance': traj['actual_distance'],
                'ideal_distance': traj['ideal_distance']
            })
        else:
            filtered_trajectories.append(traj)

    # Report filtered trajectories
    total_filtered = len(filtered_too_long) + len(filtered_too_indirect) + len(filtered_both)

    print(f"\n--- FILTERING SUMMARY ---")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Filtered out: {total_filtered} ({100*total_filtered/len(trajectories):.1f}%)")
    print(f"Remaining: {len(filtered_trajectories)} ({100*len(filtered_trajectories)/len(trajectories):.1f}%)")

    # Log filtered files by reason
    if filtered_both:
        print(f"\n[FILTERED - BOTH REASONS] ({len(filtered_both)} files)")
        print(f"  (Too long AND too indirect)")
        for f in sorted(filtered_both, key=lambda x: x['extracted_length'], reverse=True):
            print(f"  - {f['filename']}")
            print(f"      length={f['extracted_length']}, ratio={f['distance_ratio']:.2f}x")
            print(f"      actual={f['actual_distance']:.1f}px, ideal={f['ideal_distance']:.1f}px")

    if filtered_too_long:
        print(f"\n[FILTERED - TOO LONG] ({len(filtered_too_long)} files)")
        print(f"  (extracted_length > {max_length})")
        for f in sorted(filtered_too_long, key=lambda x: x['extracted_length'], reverse=True):
            print(f"  - {f['filename']}")
            print(f"      length={f['extracted_length']}, ratio={f['distance_ratio']:.2f}x")

    if filtered_too_indirect:
        print(f"\n[FILTERED - TOO INDIRECT] ({len(filtered_too_indirect)} files)")
        print(f"  (distance ratio > {max_distance_ratio}x)")
        for f in sorted(filtered_too_indirect, key=lambda x: x['distance_ratio'], reverse=True):
            print(f"  - {f['filename']}")
            print(f"      ratio={f['distance_ratio']:.2f}x, length={f['extracted_length']}")
            print(f"      actual={f['actual_distance']:.1f}px, ideal={f['ideal_distance']:.1f}px")

    if total_filtered == 0:
        print(f"\n[NO FILES FILTERED] All trajectories passed quality checks!")

    # Statistics on remaining data
    remaining_lengths = [len(t['x']) for t in filtered_trajectories]
    remaining_ratios = [t['actual_distance'] / t['ideal_distance'] for t in filtered_trajectories]

    print(f"\n--- REMAINING DATA STATISTICS ---")
    print(f"Length: min={min(remaining_lengths)}, max={max(remaining_lengths)}, "
          f"median={np.median(remaining_lengths):.0f}")
    print(f"Distance ratio: min={min(remaining_ratios):.2f}x, max={max(remaining_ratios):.2f}x, "
          f"median={np.median(remaining_ratios):.2f}x")

    return filtered_trajectories


# ============================================================================
# STEP 2: COMPUTE V4 STYLE FEATURES
# ============================================================================

def compute_v4_features(traj):
    """
    Compute all 8 V4 style features for a single trajectory.

    Features: [dx, dy, speed, acceleration, sin(heading), cos(heading), angular_velocity, dt]

    Args:
        traj: Dictionary with 'x', 'y', 't' arrays

    Returns:
        features: (n_points, 8) array of features
    """
    x = traj['x']
    y = traj['y']
    t = traj['t']
    n = len(x)

    # =========================================================================
    # Feature 1-2: Relative positions (dx, dy) from trajectory START
    # =========================================================================
    # This makes the trajectory position-independent!
    dx = x - x[0]  # All positions relative to start
    dy = y - y[0]

    # =========================================================================
    # Feature 8: Time delta (dt)
    # =========================================================================
    # dt[i] = time since previous point (in seconds)
    dt_raw = np.diff(t) / 1000.0  # Convert ms to seconds
    dt_raw = np.maximum(dt_raw, 1e-6)  # Prevent division by zero
    # Pad first point with same dt as second
    dt = np.concatenate([[dt_raw[0]], dt_raw])

    # =========================================================================
    # Intermediate: Point-to-point deltas for velocity calculation
    # =========================================================================
    dx_diff = np.diff(x)  # Change in x between consecutive points
    dy_diff = np.diff(y)  # Change in y between consecutive points

    # =========================================================================
    # Feature 3: Speed (magnitude of velocity in pixels/second)
    # =========================================================================
    distance_per_step = np.sqrt(dx_diff**2 + dy_diff**2)
    speed_raw = distance_per_step / dt_raw  # pixels per second
    # Pad first point with same speed as second
    speed = np.concatenate([[speed_raw[0]], speed_raw])

    # =========================================================================
    # Feature 4: Acceleration (d(speed)/dt)
    # =========================================================================
    # How quickly you speed up or slow down - key style signature!
    d_speed = np.diff(speed_raw)  # Change in speed
    dt_for_accel = dt_raw[1:]  # Time between speed measurements
    dt_for_accel = np.maximum(dt_for_accel, 1e-6)  # Prevent division by zero
    accel_raw = d_speed / dt_for_accel
    # Pad first two points (need 2 points to compute first acceleration)
    accel = np.concatenate([[accel_raw[0]], [accel_raw[0]], accel_raw])

    # =========================================================================
    # Feature 5-6: Heading as sin/cos (direction of movement)
    # =========================================================================
    # Using atan2 gives angle in radians [-pi, pi]
    heading_raw = np.arctan2(dy_diff, dx_diff)
    # Pad first point with same heading as second
    heading = np.concatenate([[heading_raw[0]], heading_raw])

    sin_heading = np.sin(heading)
    cos_heading = np.cos(heading)

    # =========================================================================
    # Feature 7: Angular velocity (d(heading)/dt) - turning rate/curvature
    # =========================================================================
    # This captures how sharply you turn - very personal style feature!
    # Need to handle angle wrapping: diff between -179 and 179 is 2, not 358
    d_heading = np.diff(heading_raw)
    # Wrap angle differences to [-pi, pi]
    d_heading = np.arctan2(np.sin(d_heading), np.cos(d_heading))
    dt_for_ang = dt_raw[1:]
    dt_for_ang = np.maximum(dt_for_ang, 1e-6)
    ang_vel_raw = d_heading / dt_for_ang  # radians per second
    # Pad first two points
    ang_vel = np.concatenate([[ang_vel_raw[0]], [ang_vel_raw[0]], ang_vel_raw])

    # =========================================================================
    # Stack all features: (n_points, 8)
    # =========================================================================
    features = np.stack([
        dx,           # 0: Relative x from start
        dy,           # 1: Relative y from start
        speed,        # 2: Speed (px/s)
        accel,        # 3: Acceleration (px/s^2)
        sin_heading,  # 4: Sin of heading
        cos_heading,  # 5: Cos of heading
        ang_vel,      # 6: Angular velocity (rad/s)
        dt            # 7: Time delta (s)
    ], axis=1)

    return features


def compute_all_features(trajectories):
    """
    Compute V4 features for all trajectories.

    Returns:
        trajectories: Updated with 'features' key containing (n, 8) arrays
    """
    print(f"\n{'='*70}")
    print("STEP 2: Computing V4 Style Features")
    print(f"{'='*70}")
    print("Features: [dx, dy, speed, accel, sin(h), cos(h), ang_vel, dt]")
    print("  - dx, dy: Position relative to start (position-independent!)")
    print("  - speed: Movement magnitude (px/s)")
    print("  - accel: Acceleration (px/s^2) - style signature")
    print("  - sin(h), cos(h): Direction (smooth encoding)")
    print("  - ang_vel: Turning rate (rad/s) - curvature style")
    print("  - dt: Time delta (s)")

    for i, traj in enumerate(trajectories):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(trajectories)}...")

        traj['features'] = compute_v4_features(traj)

    # Collect statistics
    all_features = np.vstack([traj['features'] for traj in trajectories])

    print(f"\nFeature Statistics (raw, before normalization):")
    feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
    for i, name in enumerate(feature_names):
        col = all_features[:, i]
        print(f"  {name:8s}: min={col.min():12.4f}, max={col.max():12.4f}, "
              f"mean={col.mean():12.4f}, std={col.std():12.4f}")

    return trajectories


# ============================================================================
# STEP 2.5: CLIP OUTLIERS (CRITICAL FOR TRAINING)
# ============================================================================

def clip_outliers(trajectories, percentile_low=1, percentile_high=99):
    """
    Clip extreme outliers using percentile-based thresholds.

    WHY THIS IS CRITICAL:
    Without clipping, extreme values (e.g., acceleration of 10,000,000 px/s²)
    dominate the min-max normalization, compressing 99% of the data into a
    tiny range. This makes it nearly impossible for the model to learn
    nuanced style patterns.

    Features clipped:
    - speed: Clip extreme speeds (sensor noise, teleportation artifacts)
    - accel: Clip extreme accelerations (sudden stops, noise)
    - ang_vel: Clip extreme angular velocities (instant direction changes)

    Features NOT clipped:
    - dx, dy: Represent actual trajectory shape
    - sin_h, cos_h: Already bounded [-1, 1]
    - dt: Represents actual timing (clipping would lose information)

    Args:
        trajectories: List of trajectory dicts with 'features' key
        percentile_low: Lower percentile for clipping (default: 1)
        percentile_high: Upper percentile for clipping (default: 99)

    Returns:
        trajectories: Updated with clipped features
    """
    print(f"\n{'='*70}")
    print("STEP 2.5: Clipping Outliers (Percentile-Based)")
    print(f"{'='*70}")
    print(f"Clipping to [{percentile_low}th, {percentile_high}th] percentile")
    print("This prevents extreme values from compressing the data range.")

    # Collect all features for percentile calculation
    all_features = np.vstack([traj['features'] for traj in trajectories])

    # Feature indices
    SPEED_IDX = 2
    ACCEL_IDX = 3
    ANG_VEL_IDX = 6

    # Calculate percentile thresholds
    speed_all = all_features[:, SPEED_IDX]
    accel_all = all_features[:, ACCEL_IDX]
    ang_vel_all = all_features[:, ANG_VEL_IDX]

    speed_low, speed_high = np.percentile(speed_all, [percentile_low, percentile_high])
    accel_low, accel_high = np.percentile(accel_all, [percentile_low, percentile_high])
    ang_vel_low, ang_vel_high = np.percentile(ang_vel_all, [percentile_low, percentile_high])

    print(f"\nClipping thresholds:")
    print(f"  speed:   [{speed_low:12.2f}, {speed_high:12.2f}] px/s")
    print(f"           (was [{speed_all.min():12.2f}, {speed_all.max():12.2f}])")
    print(f"  accel:   [{accel_low:12.2f}, {accel_high:12.2f}] px/s²")
    print(f"           (was [{accel_all.min():12.2f}, {accel_all.max():12.2f}])")
    print(f"  ang_vel: [{ang_vel_low:12.4f}, {ang_vel_high:12.4f}] rad/s")
    print(f"           (was [{ang_vel_all.min():12.4f}, {ang_vel_all.max():12.4f}])")

    # Count how many values will be clipped
    speed_clipped = ((speed_all < speed_low) | (speed_all > speed_high)).sum()
    accel_clipped = ((accel_all < accel_low) | (accel_all > accel_high)).sum()
    ang_vel_clipped = ((ang_vel_all < ang_vel_low) | (ang_vel_all > ang_vel_high)).sum()
    total_points = len(speed_all)

    print(f"\nPoints to be clipped:")
    print(f"  speed:   {speed_clipped:6d} ({100*speed_clipped/total_points:.2f}%)")
    print(f"  accel:   {accel_clipped:6d} ({100*accel_clipped/total_points:.2f}%)")
    print(f"  ang_vel: {ang_vel_clipped:6d} ({100*ang_vel_clipped/total_points:.2f}%)")

    # Apply clipping to each trajectory
    for traj in trajectories:
        features = traj['features']

        # Clip speed
        features[:, SPEED_IDX] = np.clip(features[:, SPEED_IDX], speed_low, speed_high)

        # Clip acceleration
        features[:, ACCEL_IDX] = np.clip(features[:, ACCEL_IDX], accel_low, accel_high)

        # Clip angular velocity
        features[:, ANG_VEL_IDX] = np.clip(features[:, ANG_VEL_IDX], ang_vel_low, ang_vel_high)

        traj['features'] = features

    # Verify clipping
    all_features_clipped = np.vstack([traj['features'] for traj in trajectories])

    print(f"\nAfter clipping:")
    feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
    for i, name in enumerate(feature_names):
        col = all_features_clipped[:, i]
        print(f"  {name:8s}: min={col.min():12.4f}, max={col.max():12.4f}, "
              f"mean={col.mean():12.4f}, std={col.std():12.4f}")

    # Store clipping params for reference (useful for generation)
    clip_params = {
        'speed_low': float(speed_low), 'speed_high': float(speed_high),
        'accel_low': float(accel_low), 'accel_high': float(accel_high),
        'ang_vel_low': float(ang_vel_low), 'ang_vel_high': float(ang_vel_high),
        'percentile_low': percentile_low, 'percentile_high': percentile_high,
    }

    return trajectories, clip_params


# ============================================================================
# STEP 3: NORMALIZE FEATURES TO [-1, 1]
# ============================================================================

def normalize_features(trajectories):
    """
    Normalize all features to [-1, 1] range.

    Normalization strategy:
    - dx, dy, speed, accel, ang_vel, dt: Global min-max to [-1, 1]
    - sin_h, cos_h: Already in [-1, 1], but we'll track stats for verification

    Returns:
        trajectories: Updated with 'features_norm' key
        norm_params: Dictionary of normalization parameters
    """
    print(f"\n{'='*70}")
    print("STEP 3: Normalizing Features to [-1, 1]")
    print(f"{'='*70}")

    # Collect all features
    all_features = np.vstack([traj['features'] for traj in trajectories])

    feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
    norm_params = {}

    # Compute min/max for each feature
    for i, name in enumerate(feature_names):
        col = all_features[:, i]

        if name in ['sin_h', 'cos_h']:
            # sin/cos are already bounded [-1, 1], but store actual range
            norm_params[f'{name}_min'] = -1.0
            norm_params[f'{name}_max'] = 1.0
            print(f"  {name}: Already in [-1, 1] (sin/cos)")
        else:
            fmin, fmax = col.min(), col.max()
            # Add small epsilon to prevent division by zero if min==max
            if fmax - fmin < 1e-8:
                fmax = fmin + 1e-8
            norm_params[f'{name}_min'] = float(fmin)
            norm_params[f'{name}_max'] = float(fmax)
            print(f"  {name}: [{fmin:.6f}, {fmax:.6f}]")

    # Normalize each trajectory
    for traj in trajectories:
        features = traj['features']
        features_norm = np.zeros_like(features)

        for i, name in enumerate(feature_names):
            fmin = norm_params[f'{name}_min']
            fmax = norm_params[f'{name}_max']

            if name in ['sin_h', 'cos_h']:
                # Already in [-1, 1]
                features_norm[:, i] = features[:, i]
            else:
                # Min-max normalization to [-1, 1]
                features_norm[:, i] = 2 * ((features[:, i] - fmin) / (fmax - fmin)) - 1

        traj['features_norm'] = features_norm

    # Verify normalization
    all_norm = np.vstack([traj['features_norm'] for traj in trajectories])
    print(f"\nVerifying normalization:")
    for i, name in enumerate(feature_names):
        col = all_norm[:, i]
        in_range = (col >= -1.0 - 1e-6).all() and (col <= 1.0 + 1e-6).all()
        status = "OK" if in_range else "FAILED"
        print(f"  {name}: [{col.min():.4f}, {col.max():.4f}] - {status}")

    return trajectories, norm_params


# ============================================================================
# STEP 4: NORMALIZE DISTANCES (CONDITIONING VARIABLE)
# ============================================================================

def normalize_distances(trajectories, norm_params):
    """
    Normalize actual_distance (path length) to [-1, 1] for conditioning.
    """
    print(f"\n{'='*70}")
    print("STEP 4: Normalizing Distance Conditioning Variable")
    print(f"{'='*70}")

    # Use actual_distance (path length) for conditioning
    actual_distances = np.array([traj['actual_distance'] for traj in trajectories])
    ideal_distances = np.array([traj['ideal_distance'] for traj in trajectories])

    d_min, d_max = actual_distances.min(), actual_distances.max()

    print(f"Actual distance (path length): [{d_min:.1f}, {d_max:.1f}] px")
    print(f"Ideal distance (straight line): [{ideal_distances.min():.1f}, {ideal_distances.max():.1f}] px")
    print(f"Curvature ratio (actual/ideal): {actual_distances.mean() / ideal_distances.mean():.2f}x")

    # Normalize to [-1, 1]
    distances_norm = 2 * ((actual_distances - d_min) / (d_max - d_min)) - 1

    for i, traj in enumerate(trajectories):
        traj['distance_norm'] = distances_norm[i]

    # Store normalization params
    norm_params['actual_dist_min'] = float(d_min)
    norm_params['actual_dist_max'] = float(d_max)
    norm_params['ideal_dist_min'] = float(ideal_distances.min())
    norm_params['ideal_dist_max'] = float(ideal_distances.max())

    print(f"Normalized distance range: [{distances_norm.min():.4f}, {distances_norm.max():.4f}]")

    return trajectories, norm_params


# ============================================================================
# STEP 5: PAD SEQUENCES (POST-PADDING)
# ============================================================================

def pad_sequences(trajectories, max_length=800):
    """
    Post-pad all sequences to fixed length.
    Real data at start, zeros (padding) at end.
    """
    print(f"\n{'='*70}")
    print(f"STEP 5: Post-Padding Sequences to Length {max_length}")
    print(f"{'='*70}")
    print("POST-PADDING: [real, real, real, ..., 0, 0, 0]")
    print("Dynamic batching will re-pad at training time for efficiency")

    padded_data = []

    for i, traj in enumerate(trajectories):
        if (i + 1) % 500 == 0:
            print(f"  Padding {i+1}/{len(trajectories)}...")

        features_norm = traj['features_norm']
        orig_len = len(features_norm)

        if orig_len >= max_length:
            # Truncate if too long
            features_padded = features_norm[:max_length]
            orig_len = max_length
        else:
            # Post-pad with zeros
            pad_len = max_length - orig_len
            padding = np.zeros((pad_len, FEATURE_DIM))
            features_padded = np.vstack([features_norm, padding])

        padded_data.append({
            'features': features_padded,
            'original_length': orig_len,
            'distance_norm': traj['distance_norm'],
            'actual_distance': traj['actual_distance'],
            'ideal_distance': traj['ideal_distance'],
            'filename': traj['filename']
        })

    # Statistics
    orig_lengths = [p['original_length'] for p in padded_data]
    print(f"Original lengths: min={min(orig_lengths)}, max={max(orig_lengths)}, "
          f"median={np.median(orig_lengths):.0f}")

    avg_padding = max_length - np.mean(orig_lengths)
    padding_percent = (avg_padding / max_length) * 100
    print(f"Average padding: {avg_padding:.1f} zeros ({padding_percent:.1f}% of sequence)")
    print(f"With dynamic batching, actual padding will be ~10-20%")

    return padded_data


# ============================================================================
# STEP 6: CREATE TRAIN/VAL/TEST SPLITS
# ============================================================================

def create_splits(padded_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split data into train/val/test sets.
    """
    print(f"\n{'='*70}")
    print("STEP 6: Creating Train/Val/Test Splits")
    print(f"{'='*70}")
    print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Stack all features: (n_samples, max_len, 8)
    X = np.stack([p['features'] for p in padded_data])
    C = np.array([p['distance_norm'] for p in padded_data])
    L = np.array([p['original_length'] for p in padded_data])

    print(f"X shape: {X.shape} (feature_dim={X.shape[2]})")
    print(f"C shape: {C.shape}")
    print(f"L shape: {L.shape}")

    assert X.shape[2] == FEATURE_DIM, f"Expected feature_dim={FEATURE_DIM}, got {X.shape[2]}"

    # First split: train+val vs test
    X_temp, X_test, C_temp, C_test, L_temp, L_test = train_test_split(
        X, C, L, test_size=test_ratio, random_state=seed
    )

    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, C_train, C_val, L_train, L_val = train_test_split(
        X_temp, C_temp, L_temp, test_size=val_size_adjusted, random_state=seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, C_train, L_train, X_val, C_val, L_val, X_test, C_test, L_test


# ============================================================================
# STEP 7: SAVE PROCESSED DATA
# ============================================================================

def save_processed_data(output_dir, X_train, C_train, L_train, X_val, C_val, L_val,
                        X_test, C_test, L_test, norm_params):
    """Save all processed data to .npy files."""
    print(f"\n{'='*70}")
    print("STEP 7: Saving Processed Data")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Save data arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train.astype(np.float32))
    np.save(os.path.join(output_dir, 'C_train.npy'), C_train.astype(np.float32))
    np.save(os.path.join(output_dir, 'L_train.npy'), L_train.astype(np.int32))

    np.save(os.path.join(output_dir, 'X_val.npy'), X_val.astype(np.float32))
    np.save(os.path.join(output_dir, 'C_val.npy'), C_val.astype(np.float32))
    np.save(os.path.join(output_dir, 'L_val.npy'), L_val.astype(np.int32))

    np.save(os.path.join(output_dir, 'X_test.npy'), X_test.astype(np.float32))
    np.save(os.path.join(output_dir, 'C_test.npy'), C_test.astype(np.float32))
    np.save(os.path.join(output_dir, 'L_test.npy'), L_test.astype(np.int32))

    # Save normalization parameters
    np.save(os.path.join(output_dir, 'normalization_params.npy'), norm_params)

    # Save feature info for reference
    feature_info = {
        'feature_dim': FEATURE_DIM,
        'feature_names': ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt'],
        'feature_descriptions': [
            'Relative x from trajectory start (pixels)',
            'Relative y from trajectory start (pixels)',
            'Speed magnitude (pixels/second)',
            'Acceleration (pixels/second^2)',
            'Sine of heading angle',
            'Cosine of heading angle',
            'Angular velocity / turning rate (radians/second)',
            'Time delta between samples (seconds)'
        ],
        'version': 'V4',
        'position_independent': True
    }
    np.save(os.path.join(output_dir, 'feature_info.npy'), feature_info)

    print(f"\nSaved files:")
    print(f"  X_train.npy: {X_train.shape} (feature_dim={X_train.shape[2]})")
    print(f"  C_train.npy: {C_train.shape}")
    print(f"  L_train.npy: {L_train.shape}")
    print(f"  X_val.npy: {X_val.shape}")
    print(f"  X_test.npy: {X_test.shape}")
    print(f"  normalization_params.npy: {len(norm_params)} parameters")
    print(f"  feature_info.npy: V4 feature documentation")

    # Verify save
    X_test_loaded = np.load(os.path.join(output_dir, 'X_test.npy'))
    assert X_test_loaded.shape == X_test.shape, "Save/load verification failed"
    print("\nVerification: Data saved and loaded correctly")

    return


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete V4 preprocessing pipeline."""

    parser = argparse.ArgumentParser(
        description='Preprocess trajectories for V4 (position-independent style features)'
    )
    parser.add_argument(
        '--input_dir',
        default=DEFAULT_INPUT_DIR,
        help='Input directory with JSON trajectory files'
    )
    parser.add_argument(
        '--output_dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for processed .npy files'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=MAX_SEQ_LENGTH,
        help='Maximum sequence length for storage/padding (default: 800)'
    )
    parser.add_argument(
        '--max_filter_length',
        type=int,
        default=750,
        help='Maximum trajectory length before filtering (default: 750)'
    )
    parser.add_argument(
        '--max_distance_ratio',
        type=float,
        default=2.5,
        help='Maximum actual/ideal distance ratio before filtering (default: 2.5)'
    )
    parser.add_argument(
        '--no_filter',
        action='store_true',
        help='Disable trajectory filtering (not recommended)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TIMEGAN V4.0 - POSITION-INDEPENDENT STYLE PREPROCESSING")
    print("=" * 70)
    print(f"\nInput: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max sequence length (padding): {args.max_length}")
    print(f"Max filter length: {args.max_filter_length}")
    print(f"Max distance ratio: {args.max_distance_ratio}x")
    print(f"Filtering: {'DISABLED' if args.no_filter else 'ENABLED'}")
    print(f"Feature dimension: {FEATURE_DIM}")

    print("\n" + "=" * 70)
    print("V4 FEATURES (Position-Independent Style Learning)")
    print("=" * 70)
    print("""
    [dx, dy, speed, accel, sin(h), cos(h), ang_vel, dt]

    1. dx, dy      : Position relative to START (not absolute screen coords!)
    2. speed       : How fast you move (px/s)
    3. accel       : How you speed up/slow down (px/s^2) - STYLE SIGNATURE
    4. sin(h),cos(h): Direction of movement (smooth encoding)
    5. ang_vel     : Turning rate (rad/s) - CURVATURE STYLE
    6. dt          : Timing between samples (s)

    WHY THIS IS BETTER THAN V3:
    - Same movement at different screen positions = SAME features
    - Model learns HOW you move, not WHERE you moved
    - Generated trajectories can start from ANY point
    """)
    print("=" * 70)

    # Run pipeline
    trajectories = load_trajectories(args.input_dir)

    # Filter bad recordings (critical for training quality!)
    if not args.no_filter:
        trajectories = filter_trajectories(
            trajectories,
            max_length=args.max_filter_length,
            max_distance_ratio=args.max_distance_ratio
        )
    else:
        print("\n[WARNING] Filtering disabled! Training quality may be affected.")

    trajectories = compute_all_features(trajectories)
    trajectories, clip_params = clip_outliers(trajectories)  # Remove extreme outliers
    trajectories, norm_params = normalize_features(trajectories)

    # Merge clip params into norm params for reference
    norm_params['clip_params'] = clip_params
    trajectories, norm_params = normalize_distances(trajectories, norm_params)
    padded_data = pad_sequences(trajectories, args.max_length)

    X_train, C_train, L_train, X_val, C_val, L_val, X_test, C_test, L_test = create_splits(
        padded_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    save_processed_data(
        args.output_dir,
        X_train, C_train, L_train,
        X_val, C_val, L_val,
        X_test, C_test, L_test,
        norm_params
    )

    print("\n" + "=" * 70)
    print("V4 PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nProcessed data saved to: {args.output_dir}")
    print(f"Total trajectories: {len(X_train) + len(X_val) + len(X_test)}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Features: [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]")
    print(f"\nKey V4 improvements over V3:")
    print(f"  - Position-independent (dx, dy relative to start)")
    print(f"  - Acceleration captures speed-up/slow-down style")
    print(f"  - Angular velocity captures turning/curvature style")
    print(f"  - Sin/cos heading avoids angle discontinuity")
    print(f"  - Outlier clipping (1st-99th percentile) for better normalization")
    if not args.no_filter:
        print(f"  - Quality filtering (length <= {args.max_filter_length}, ratio <= {args.max_distance_ratio}x)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

