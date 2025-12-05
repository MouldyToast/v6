"""
Detect and trim initial pause/positioning movements from trajectories.

This script identifies where the "real" trajectory begins by detecting sustained movement,
and removes the initial pause or inter-recording positioning data.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader_v6 import load_v6_data


# ============================================================================
# CONFIGURATION - ADJUST THESE TO TUNE PAUSE DETECTION
# ============================================================================

# Speed threshold (normalized space: -1.0 = slowest, 0.0 = medium, +1.0 = fastest)
# Lower values (like -0.8) = more aggressive trimming (catches slower pauses)
# Higher values (like -0.3) = more conservative (only trims very slow pauses)
SPEED_THRESHOLD = -0.7  # Try: -0.8 for ~30 timesteps, -0.7 for ~40, -0.6 for ~50

# Number of consecutive points that must exceed threshold
# Lower values (3) = trigger earlier, more sensitive
# Higher values (10) = require more sustained movement, more conservative
WINDOW_SIZE = 3  # Try: 3 for early trigger, 5 for balanced, 10 for conservative

# Maximum timesteps to search for movement start (limits where detection can occur)
# If pause not found in first N timesteps, keeps entire trajectory
# Set to None to search entire trajectory
MAX_SEARCH_TIMESTEPS = 60  # Try: 40 for ~30ts, 50 for ~40ts, 60 for ~50ts, None for unlimited

# Minimum points that must remain after trimming (prevents over-trimming)
MIN_REMAINING_POINTS = 20

# ============================================================================


def detect_movement_start(x, y, t, speed_threshold=50, window_size=5, min_movement_points=10):
    """
    Detect where real movement begins by finding sustained speed above threshold.

    Args:
        x, y: Position arrays (pixels)
        t: Time array (milliseconds)
        speed_threshold: Speed threshold in pixels/second
        window_size: Number of consecutive points that must exceed threshold
        min_movement_points: Minimum remaining points after trim

    Returns:
        start_idx: Index where real movement begins (0 if no pause detected)
        stats: Dictionary with diagnostic information
    """
    if len(x) < window_size + min_movement_points:
        return 0, {'reason': 'trajectory_too_short', 'length': len(x)}

    # Compute instantaneous speed
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t) / 1000.0  # Convert to seconds
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero

    speed = np.sqrt(dx**2 + dy**2) / dt  # pixels/second

    # Find first sustained movement
    for i in range(len(speed) - window_size):
        # Check if next window_size points all exceed threshold
        if np.all(speed[i:i+window_size] > speed_threshold):
            # Make sure there are enough points remaining
            remaining = len(x) - i
            if remaining >= min_movement_points:
                # Calculate stats for the trimmed portion
                trimmed_distance = np.sqrt((x[i] - x[0])**2 + (y[i] - y[0])**2)
                trimmed_duration = (t[i] - t[0]) / 1000.0
                trimmed_avg_speed = speed[:i].mean() if i > 0 else 0

                return i, {
                    'reason': 'pause_detected',
                    'start_idx': i,
                    'trimmed_points': i,
                    'trimmed_distance': trimmed_distance,
                    'trimmed_duration': trimmed_duration,
                    'trimmed_avg_speed': trimmed_avg_speed,
                    'movement_avg_speed': speed[i:].mean(),
                    'speed_ratio': speed[i:].mean() / (trimmed_avg_speed + 1e-6),
                }

    # No sustained movement found - keep entire trajectory
    return 0, {
        'reason': 'no_pause_detected',
        'max_speed': speed.max(),
        'avg_speed': speed.mean(),
    }


def analyze_and_trim_trajectories(data_dir, speed_threshold=50, window_size=5):
    """
    Analyze all trajectories and show statistics about initial pauses.

    Args:
        data_dir: Path to processed_data_v4
        speed_threshold: Speed threshold in px/s for movement detection
        window_size: Window size for sustained movement check

    Returns:
        stats: Dictionary with overall statistics
    """
    print("="*70)
    print(" Initial Pause Detection Analysis")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Speed threshold (normalized): {SPEED_THRESHOLD}")
    print(f"  Window size: {WINDOW_SIZE} consecutive points")
    print(f"  Max search timesteps: {MAX_SEARCH_TIMESTEPS if MAX_SEARCH_TIMESTEPS else 'unlimited'}")
    print(f"  Min remaining points: {MIN_REMAINING_POINTS}")
    print(f"  Data directory: {data_dir}")
    print()

    # Load data
    print("Loading data...")
    data = load_v6_data(data_dir)

    # Analyze each split
    overall_stats = {}

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f" Analyzing {split.upper()} split")
        print(f"{'='*70}")

        X = data[f'X_{split}']
        L = data[f'L_{split}']

        pause_detected = 0
        no_pause = 0
        trimmed_points_total = 0
        trimmed_distances = []
        trimmed_durations = []
        speed_ratios = []

        trajectory_stats = []

        for i in range(len(X)):
            length = L[i]
            traj = X[i, :length, :]  # (T, 8)

            # Denormalize to approximate physical units
            # Note: These are normalized features, so we're working with normalized values
            # The detection will still work on normalized data patterns

            # Extract position and time (these are relative dx, dy from start)
            # We need to reconstruct absolute positions for speed calculation
            dx_rel = traj[:, 0]  # Relative to start
            dy_rel = traj[:, 1]

            # Since these are relative to start, we can treat them as absolute positions
            # (starting from origin)
            x = dx_rel
            y = dy_rel

            # For dt, we need to denormalize or work with the normalized values
            # The speed feature is already computed, so let's use that instead
            speed_normalized = traj[:, 2]

            # Detect movement using normalized speed (use global config parameters)
            start_idx = 0
            pause_found = False

            # Determine search range
            if MAX_SEARCH_TIMESTEPS is None:
                search_range = len(speed_normalized) - WINDOW_SIZE
            else:
                search_range = min(MAX_SEARCH_TIMESTEPS, len(speed_normalized) - WINDOW_SIZE)

            # Find first point where speed exceeds threshold for WINDOW_SIZE consecutive points
            for j in range(max(0, search_range)):
                if np.all(speed_normalized[j:j+WINDOW_SIZE] > SPEED_THRESHOLD):
                    # Check if enough points remain after trimming
                    remaining = length - j
                    if remaining >= MIN_REMAINING_POINTS:
                        start_idx = j
                        pause_found = True
                        break

            if pause_found and start_idx > 0:
                pause_detected += 1
                trimmed_points_total += start_idx

                # Calculate stats
                trimmed_distance = np.sqrt(dx_rel[start_idx]**2 + dy_rel[start_idx]**2)
                trimmed_distances.append(trimmed_distance)

                # For duration, use the dt values
                # dt is feature index 7
                dt_normalized = traj[:start_idx, 7]
                trimmed_durations.append(len(dt_normalized))

                trimmed_avg_speed = speed_normalized[:start_idx].mean()
                movement_avg_speed = speed_normalized[start_idx:].mean()
                speed_ratio = movement_avg_speed / (trimmed_avg_speed + 1e-6)
                speed_ratios.append(speed_ratio)

                trajectory_stats.append({
                    'index': i,
                    'original_length': length,
                    'start_idx': start_idx,
                    'trimmed_points': start_idx,
                    'trimmed_pct': 100 * start_idx / length,
                    'trimmed_distance': trimmed_distance,
                    'pause_avg_speed': trimmed_avg_speed,
                    'movement_avg_speed': movement_avg_speed,
                    'speed_ratio': speed_ratio,
                })
            else:
                no_pause += 1

        # Report statistics
        total = len(X)
        print(f"\nOverall Statistics:")
        print(f"  Total trajectories: {total}")
        print(f"  With initial pause: {pause_detected} ({100*pause_detected/total:.1f}%)")
        print(f"  Without pause: {no_pause} ({100*no_pause/total:.1f}%)")

        if pause_detected > 0:
            print(f"\nPause Statistics:")
            print(f"  Total points trimmed: {trimmed_points_total}")
            print(f"  Avg points per pause: {trimmed_points_total/pause_detected:.1f}")
            print(f"  Avg trimmed distance: {np.mean(trimmed_distances):.3f} (normalized)")
            print(f"  Avg trimmed duration: {np.mean(trimmed_durations):.1f} timesteps")
            print(f"  Avg speed ratio (movement/pause): {np.mean(speed_ratios):.2f}x")

            # Show examples of trajectories with longest pauses
            print(f"\nTop 5 trajectories with longest pauses:")
            sorted_stats = sorted(trajectory_stats, key=lambda x: x['trimmed_points'], reverse=True)
            for j, s in enumerate(sorted_stats[:5], 1):
                print(f"  {j}. Trajectory {s['index']}:")
                print(f"     Trimmed {s['trimmed_points']}/{s['original_length']} points ({s['trimmed_pct']:.1f}%)")
                print(f"     Speed ratio: {s['speed_ratio']:.2f}x")

        overall_stats[split] = {
            'total': total,
            'pause_detected': pause_detected,
            'no_pause': no_pause,
            'trimmed_points_total': trimmed_points_total,
            'trajectory_stats': trajectory_stats,
        }

    return overall_stats


def visualize_pause_examples(data_dir, n_examples=3):
    """
    Visualize examples of trajectories with and without initial pauses.
    """
    print(f"\n{'='*70}")
    print(" Generating Visualization")
    print(f"{'='*70}")

    data = load_v6_data(data_dir)
    X_train = data['X_train']
    L_train = data['L_train']

    # Find examples with pauses and without (use global config parameters)
    examples_with_pause = []
    examples_without_pause = []

    for i in range(len(X_train)):
        if len(examples_with_pause) >= n_examples and len(examples_without_pause) >= n_examples:
            break

        length = L_train[i]
        speed = X_train[i, :length, 2]

        # Check for pause using same logic as analysis
        start_idx = 0
        pause_found = False

        # Determine search range
        if MAX_SEARCH_TIMESTEPS is None:
            search_range = len(speed) - WINDOW_SIZE
        else:
            search_range = min(MAX_SEARCH_TIMESTEPS, len(speed) - WINDOW_SIZE)

        for j in range(max(0, search_range)):
            if np.all(speed[j:j+WINDOW_SIZE] > SPEED_THRESHOLD):
                remaining = length - j
                if remaining >= MIN_REMAINING_POINTS:
                    start_idx = j
                    pause_found = True
                    break

        if pause_found and start_idx > 5 and len(examples_with_pause) < n_examples:
            examples_with_pause.append((i, start_idx, length))
        elif not pause_found and len(examples_without_pause) < n_examples:
            examples_without_pause.append((i, 0, length))

    # Create visualization
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, start_idx, length) in enumerate(examples_with_pause):
        traj = X_train[idx, :length, :]

        # XY trajectory
        ax = axes[row, 0]
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, label='Full trajectory')
        if start_idx > 0:
            ax.plot(traj[:start_idx, 0], traj[:start_idx, 1], 'r-', linewidth=2, label='Pause (trimmed)')
            ax.plot(traj[start_idx:, 0], traj[start_idx:, 1], 'g-', linewidth=2, label='Movement')
            ax.scatter([traj[start_idx, 0]], [traj[start_idx, 1]], c='orange', s=100, marker='o',
                      label=f'Start (t={start_idx})', zorder=5)
        ax.set_xlabel('dx (normalized)')
        ax.set_ylabel('dy (normalized)')
        ax.set_title(f'Trajectory {idx} - XY Path')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Speed over time
        ax = axes[row, 1]
        timesteps = np.arange(length)
        ax.plot(timesteps, traj[:, 2], 'b-', alpha=0.5, label='Speed')
        if start_idx > 0:
            ax.axvline(start_idx, color='orange', linestyle='--', linewidth=2, label='Movement start')
            ax.axhspan(-1, SPEED_THRESHOLD, color='red', alpha=0.1, label=f'Pause threshold ({SPEED_THRESHOLD})')
        # Show max search range if limited
        if MAX_SEARCH_TIMESTEPS is not None:
            ax.axvline(MAX_SEARCH_TIMESTEPS, color='gray', linestyle=':', alpha=0.5, label=f'Max search ({MAX_SEARCH_TIMESTEPS})')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Speed (normalized)')
        ax.set_title(f'Speed over Time (trimmed {start_idx} pts)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Feature comparison (before/after trim)
        ax = axes[row, 2]
        feature_names = ['dx', 'dy', 'speed', 'accel', 'ang_vel', 'dt']
        feature_indices = [0, 1, 2, 3, 6, 7]

        pause_means = [traj[:start_idx, i].mean() if start_idx > 0 else 0 for i in feature_indices]
        movement_means = [traj[start_idx:, i].mean() for i in feature_indices]

        x_pos = np.arange(len(feature_names))
        width = 0.35

        ax.bar(x_pos - width/2, pause_means, width, label='Pause', color='red', alpha=0.7)
        ax.bar(x_pos + width/2, movement_means, width, label='Movement', color='green', alpha=0.7)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Value (normalized)')
        ax.set_title('Feature Means: Pause vs Movement')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('initial_pause_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: initial_pause_analysis.png")

    return fig


if __name__ == "__main__":
    DATA_DIR = "processed_data_v4"

    # Run analysis
    stats = analyze_and_trim_trajectories(DATA_DIR, speed_threshold=50, window_size=5)

    # Create visualizations
    visualize_pause_examples(DATA_DIR, n_examples=3)

    print(f"\n{'='*70}")
    print(" Analysis Complete!")
    print(f"{'='*70}")
    print()
    print("Summary:")
    for split in ['train', 'val', 'test']:
        s = stats[split]
        pct = 100 * s['pause_detected'] / s['total']
        print(f"  {split:5s}: {s['pause_detected']:4d}/{s['total']:4d} trajectories ({pct:5.1f}%) have initial pause")

    print()
    print("Next steps:")
    print("1. Review 'initial_pause_analysis.png' to verify pause detection")
    print("2. If detection looks good, modify preprocess_v4.py to trim pauses")
    print("3. Reprocess the raw data with trimming enabled")
    print("4. Retrain the model on cleaned data")
