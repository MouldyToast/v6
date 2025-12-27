"""
Preprocessing for V8 Trajectory Diffusion

V8 Key Changes from V6/V7:
- Output 2D positions (x, y) instead of 8D features
- No goal conditions (C_*.npy files)
- Store endpoints separately for evaluation

Output format:
    X: (n_samples, max_seq_len, 2) - positions relative to start, normalized
    L: (n_samples,) - sequence lengths
    endpoints: (n_samples, 2) - original endpoints for evaluation

Usage:
    python preprocess_V8.py --input trajectories/ --output processed_data_v8/
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

DEFAULT_INPUT_DIR = 'trajectories'
DEFAULT_OUTPUT_DIR = 'processed_data_v8'

MAX_SEQ_LENGTH = 200
RANDOM_SEED = 42

# Train/val/test split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# V8: 2D features (x, y positions)
FEATURE_DIM = 2


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

            required_fields = ['x', 'y', 't', 'ideal_distance', 'actual_distance']
            if not all(k in data for k in required_fields):
                print(f"  Skipping {json_file.name}: missing required fields")
                continue

            if not (len(data['x']) == len(data['y']) == len(data['t'])):
                print(f"  Skipping {json_file.name}: array length mismatch")
                continue

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

    point_counts = [len(traj['x']) for traj in trajectories]
    distances = [traj['ideal_distance'] for traj in trajectories]

    print(f"Point counts: min={min(point_counts)}, max={max(point_counts)}, "
          f"median={np.median(point_counts):.0f}")
    print(f"Distance range: {min(distances):.1f} to {max(distances):.1f} pixels")

    return trajectories


# ============================================================================
# STEP 2: FILTER BAD RECORDINGS
# ============================================================================

def filter_trajectories(trajectories, max_length=195, max_distance_ratio=3.0, min_distance=20):
    """
    Filter out bad recordings.

    Filter criteria:
    1. length > max_length: Would be truncated
    2. actual_distance / ideal_distance > max_distance_ratio: Erratic paths
    3. ideal_distance < min_distance: Too short

    Returns:
        filtered_trajectories: Clean trajectories
    """
    print(f"\n{'='*70}")
    print("STEP 2: Filtering Bad Recordings")
    print(f"{'='*70}")

    filtered = []
    stats = {'too_long': 0, 'erratic': 0, 'too_short': 0}

    for traj in trajectories:
        length = len(traj['x'])
        distance_ratio = traj['actual_distance'] / max(traj['ideal_distance'], 1e-6)

        if length > max_length:
            stats['too_long'] += 1
            continue

        if distance_ratio > max_distance_ratio:
            stats['erratic'] += 1
            continue

        if traj['ideal_distance'] < min_distance:
            stats['too_short'] += 1
            continue

        filtered.append(traj)

    print(f"Filtered: {len(trajectories)} -> {len(filtered)}")
    print(f"  Removed (too long > {max_length}): {stats['too_long']}")
    print(f"  Removed (erratic ratio > {max_distance_ratio}): {stats['erratic']}")
    print(f"  Removed (too short < {min_distance}px): {stats['too_short']}")

    return filtered


# ============================================================================
# STEP 3: COMPUTE POSITION SCALE
# ============================================================================

def compute_position_scale(trajectories):
    """
    Compute normalization scale from training data.

    Uses max displacement across all trajectories.
    """
    print(f"\n{'='*70}")
    print("STEP 3: Computing Position Scale")
    print(f"{'='*70}")

    max_displacement = 0
    for traj in trajectories:
        x, y = traj['x'], traj['y']
        x_rel = x - x[0]
        y_rel = y - y[0]
        displacement = np.sqrt(x_rel**2 + y_rel**2).max()
        max_displacement = max(max_displacement, displacement)

    # Add 10% margin
    position_scale = max_displacement * 1.1

    print(f"Max displacement: {max_displacement:.2f} px")
    print(f"Position scale (with margin): {position_scale:.2f} px")

    return position_scale


# ============================================================================
# STEP 4: EXTRACT V8 FEATURES (2D POSITIONS)
# ============================================================================

def extract_features_v8(traj, position_scale):
    """
    Extract V8 features: 2D positions relative to start, normalized.

    Args:
        traj: Trajectory dict with 'x', 'y' arrays
        position_scale: Normalization scale

    Returns:
        features: (L, 2) array of normalized positions
        endpoint: (2,) original endpoint before normalization
    """
    x = traj['x']
    y = traj['y']

    # Make relative to start
    x_rel = x - x[0]
    y_rel = y - y[0]

    # Store original endpoint for evaluation
    endpoint = np.array([x_rel[-1], y_rel[-1]])

    # Normalize
    x_norm = x_rel / position_scale
    y_norm = y_rel / position_scale

    # Stack into (L, 2)
    features = np.stack([x_norm, y_norm], axis=1)

    return features, endpoint


def process_all_trajectories(trajectories, position_scale):
    """Process all trajectories into V8 format."""
    print(f"\n{'='*70}")
    print("STEP 4: Extracting V8 Features (2D Positions)")
    print(f"{'='*70}")

    all_features = []
    all_endpoints = []
    all_lengths = []

    for traj in trajectories:
        features, endpoint = extract_features_v8(traj, position_scale)
        all_features.append(features)
        all_endpoints.append(endpoint)
        all_lengths.append(len(features))

    print(f"Processed {len(all_features)} trajectories")
    print(f"Length range: [{min(all_lengths)}, {max(all_lengths)}]")

    return all_features, all_endpoints, all_lengths


# ============================================================================
# STEP 5: PAD AND SPLIT
# ============================================================================

def pad_trajectories(features_list, max_len=MAX_SEQ_LENGTH):
    """Pad all trajectories to fixed length."""
    print(f"\n{'='*70}")
    print("STEP 5: Padding Trajectories")
    print(f"{'='*70}")

    n_samples = len(features_list)
    X = np.zeros((n_samples, max_len, FEATURE_DIM), dtype=np.float32)

    for i, features in enumerate(features_list):
        L = min(len(features), max_len)
        X[i, :L] = features[:L]

    print(f"Padded to shape: {X.shape}")
    return X


def split_data(X, endpoints, lengths, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Split data into train/val/test sets."""
    print(f"\n{'='*70}")
    print("STEP 6: Splitting Data")
    print(f"{'='*70}")

    endpoints = np.array(endpoints, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    n = len(X)
    indices = np.arange(n)

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=RANDOM_SEED
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=RANDOM_SEED
    )

    splits = {
        'train': (X[train_idx], endpoints[train_idx], lengths[train_idx]),
        'val': (X[val_idx], endpoints[val_idx], lengths[val_idx]),
        'test': (X[test_idx], endpoints[test_idx], lengths[test_idx]),
    }

    for name, (x, e, l) in splits.items():
        print(f"  {name}: {len(x)} samples")

    return splits


# ============================================================================
# STEP 7: SAVE
# ============================================================================

def save_data(splits, position_scale, output_dir):
    """Save processed data to disk."""
    print(f"\n{'='*70}")
    print("STEP 7: Saving Data")
    print(f"{'='*70}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, (X, endpoints, lengths) in splits.items():
        np.save(output_path / f'X_{split_name}.npy', X)
        np.save(output_path / f'endpoints_{split_name}.npy', endpoints)
        np.save(output_path / f'L_{split_name}.npy', lengths)
        print(f"  Saved {split_name}: X={X.shape}, endpoints={endpoints.shape}, L={lengths.shape}")

    # Save normalization params
    norm_params = {
        'position_scale': position_scale,
        'feature_dim': FEATURE_DIM,
        'max_seq_len': MAX_SEQ_LENGTH,
    }
    np.save(output_path / 'normalization_params.npy', norm_params)
    print(f"  Saved normalization_params.npy")

    print(f"\nOutput directory: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Preprocess trajectories for V8 diffusion')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
                        help='Input directory with trajectory JSON files')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for processed data')
    parser.add_argument('--max_length', type=int, default=195,
                        help='Maximum trajectory length (filter longer)')
    parser.add_argument('--max_distance_ratio', type=float, default=3.0,
                        help='Maximum actual/ideal distance ratio')
    parser.add_argument('--min_distance', type=float, default=20.0,
                        help='Minimum ideal distance in pixels')
    args = parser.parse_args()

    print("=" * 70)
    print("V8 TRAJECTORY PREPROCESSING")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Features: 2D positions (x, y)")
    print("=" * 70)

    # Step 1: Load
    trajectories = load_trajectories(args.input)

    # Step 2: Filter
    trajectories = filter_trajectories(
        trajectories,
        max_length=args.max_length,
        max_distance_ratio=args.max_distance_ratio,
        min_distance=args.min_distance
    )

    # Step 3: Compute scale
    position_scale = compute_position_scale(trajectories)

    # Step 4: Extract features
    features_list, endpoints, lengths = process_all_trajectories(trajectories, position_scale)

    # Step 5: Pad
    X = pad_trajectories(features_list)

    # Step 6: Split
    splits = split_data(X, endpoints, lengths)

    # Step 7: Save
    save_data(splits, position_scale, args.output)

    print("\n" + "=" * 70)
    print("V8 PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
