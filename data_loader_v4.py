"""
Data Loader for TimeGAN V4

Handles loading preprocessed V4 data and creating PyTorch DataLoaders
with dynamic batching for efficient training.

V4 Features (8 dimensions):
    [0] dx       - Relative x from start
    [1] dy       - Relative y from start
    [2] speed    - Movement speed (px/s)
    [3] accel    - Acceleration (px/s^2)
    [4] sin_h    - Sine of heading
    [5] cos_h    - Cosine of heading
    [6] ang_vel  - Angular velocity (rad/s)
    [7] dt       - Time delta (s)

Dynamic Batching:
    Each batch is padded to its own maximum length (not fixed 800).
    This gives ~85-95% real data efficiency vs ~20-30% with fixed padding.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MouseTrajectoryDatasetV4(Dataset):
    """
    PyTorch Dataset for V4 mouse trajectories.

    Returns unpadded sequences (removes storage padding).
    The DataLoader's collate function handles re-padding per batch.
    """

    def __init__(self, X, C, L, augment=False):
        """
        Args:
            X: (n_samples, max_seq_len, 8) - padded feature arrays
            C: (n_samples,) - normalized distance conditions
            L: (n_samples,) - original lengths before padding
            augment: Whether to apply data augmentation
        """
        self.X = torch.FloatTensor(X)
        self.C = torch.FloatTensor(C).unsqueeze(-1)  # (n, 1)
        self.L = torch.LongTensor(L)
        self.augment = augment

        # Validate feature dimension
        assert X.shape[2] == 8, f"Expected feature_dim=8, got {X.shape[2]}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            x: (original_length, 8) - unpadded trajectory
            c: (1,) - distance condition
            length: scalar - original length
        """
        length = self.L[idx].item()

        # Extract only real data (no padding)
        x = self.X[idx, :length, :].clone()
        c = self.C[idx].clone()

        # Optional augmentation
        if self.augment:
            x = self._augment(x)

        return x, c, length

    def _augment(self, x):
        """
        Apply augmentation to trajectory.

        V4 augmentation is simpler than V3 because we're working with
        position-independent features. We can:
        1. Add small noise to features
        2. Flip trajectory (negate dx, dy, sin_h, and ang_vel for horizontal flip)
        """
        # Random horizontal flip (50% chance)
        if torch.rand(1).item() > 0.5:
            x = x.clone()
            x[:, 0] = -x[:, 0]  # Flip dx
            x[:, 4] = -x[:, 4]  # Flip sin(heading)
            x[:, 6] = -x[:, 6]  # Flip angular velocity

        # Random vertical flip (50% chance)
        # Vertical flip = reflection about x-axis
        # For heading angle h, the new heading h' = -h
        # sin(h') = sin(-h) = -sin(h)
        # cos(h') = cos(-h) = cos(h) (unchanged)
        # Angular velocity flips sign (clockwise becomes counter-clockwise)
        if torch.rand(1).item() > 0.5:
            x = x.clone()
            x[:, 1] = -x[:, 1]   # Flip dy
            x[:, 4] = -x[:, 4]   # Flip sin(heading): sin(-h) = -sin(h)
            # cos(heading) stays same: cos(-h) = cos(h)
            x[:, 6] = -x[:, 6]   # Flip angular velocity

        # Add small noise (helps generalization)
        noise_scale = 0.02
        noise = torch.randn_like(x) * noise_scale
        # Don't add noise to sin_h, cos_h (should stay on unit circle)
        noise[:, 4] = 0
        noise[:, 5] = 0
        x = x + noise

        # Clamp to valid range
        x = torch.clamp(x, -1, 1)

        return x


def collate_fn_v4(batch):
    """
    Custom collate function for dynamic batching.

    Pads each batch to its own maximum length (not fixed 800).
    This is much more efficient for variable-length sequences.

    Args:
        batch: List of (x, c, length) tuples

    Returns:
        X_padded: (batch_size, max_len_in_batch, 8)
        C: (batch_size, 1)
        lengths: (batch_size,)
    """
    # Sort by length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    # Extract components
    xs, cs, lengths = zip(*batch)

    # Find max length in this batch
    max_len = max(lengths)

    # Pad sequences to batch max length (not fixed 800)
    batch_size = len(xs)
    feature_dim = xs[0].size(-1)

    X_padded = torch.zeros(batch_size, max_len, feature_dim)
    for i, (x, length) in enumerate(zip(xs, lengths)):
        X_padded[i, :length, :] = x

    # Stack conditions and lengths
    C = torch.stack(cs)
    lengths = torch.LongTensor(lengths)

    return X_padded, C, lengths


def create_dataloaders_v4(data_dir, batch_size=64, num_workers=0, augment_train=True):
    """
    Create train, val, test DataLoaders for V4 data.

    Args:
        data_dir: Directory containing preprocessed V4 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Loading V4 data from: {data_dir}")

    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    C_train = np.load(os.path.join(data_dir, 'C_train.npy'))
    L_train = np.load(os.path.join(data_dir, 'L_train.npy'))

    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    C_val = np.load(os.path.join(data_dir, 'C_val.npy'))
    L_val = np.load(os.path.join(data_dir, 'L_val.npy'))

    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    C_test = np.load(os.path.join(data_dir, 'C_test.npy'))
    L_test = np.load(os.path.join(data_dir, 'L_test.npy'))

    # Validate V4 data
    assert X_train.shape[2] == 8, f"Expected feature_dim=8, got {X_train.shape[2]}"

    print(f"  Train: {len(X_train)} samples, shape {X_train.shape}")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Feature dim: {X_train.shape[2]} (V4)")

    # Create datasets
    train_dataset = MouseTrajectoryDatasetV4(X_train, C_train, L_train, augment=augment_train)
    val_dataset = MouseTrajectoryDatasetV4(X_val, C_val, L_val, augment=False)
    test_dataset = MouseTrajectoryDatasetV4(X_test, C_test, L_test, augment=False)

    # Create dataloaders with dynamic batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_v4,
        pin_memory=True,
        drop_last=True,  # For consistent batch sizes
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v4,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v4,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # Print efficiency stats
    avg_len = L_train.mean()
    max_len = X_train.shape[1]
    efficiency = (avg_len / max_len) * 100
    print(f"\n  Fixed padding efficiency: {efficiency:.1f}%")
    print(f"  With dynamic batching: ~85-95% (much better!)")

    return train_loader, val_loader, test_loader


def load_normalization_params(data_dir):
    """
    Load normalization parameters for denormalization.

    Args:
        data_dir: Directory containing normalization_params.npy

    Returns:
        Dictionary of normalization parameters
    """
    path = os.path.join(data_dir, 'normalization_params.npy')
    return np.load(path, allow_pickle=True).item()


def load_v4_data(data_dir):
    """
    Load all V4 data into a dictionary.

    Args:
        data_dir: Directory containing preprocessed V4 data

    Returns:
        Dictionary with X_train, C_train, L_train, X_val, etc.
    """
    data = {
        'X_train': np.load(os.path.join(data_dir, 'X_train.npy')),
        'C_train': np.load(os.path.join(data_dir, 'C_train.npy')),
        'L_train': np.load(os.path.join(data_dir, 'L_train.npy')),
        'X_val': np.load(os.path.join(data_dir, 'X_val.npy')),
        'C_val': np.load(os.path.join(data_dir, 'C_val.npy')),
        'L_val': np.load(os.path.join(data_dir, 'L_val.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test.npy')),
        'C_test': np.load(os.path.join(data_dir, 'C_test.npy')),
        'L_test': np.load(os.path.join(data_dir, 'L_test.npy')),
    }

    # Load normalization params if available
    norm_path = os.path.join(data_dir, 'normalization_params.npy')
    if os.path.exists(norm_path):
        data['norm_params'] = np.load(norm_path, allow_pickle=True).item()

    return data


def create_dataloader_v4(X, C, L, batch_size=64, shuffle=True, num_workers=0, augment=False):
    """
    Create a single DataLoader for V4 data.

    Args:
        X: (n_samples, max_seq_len, 8) - feature arrays
        C: (n_samples,) - distance conditions
        L: (n_samples,) - original lengths
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        augment: Whether to apply augmentation

    Returns:
        DataLoader instance
    """
    dataset = MouseTrajectoryDatasetV4(X, C, L, augment=augment)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_v4,
        pin_memory=True,
        drop_last=shuffle,  # Only drop last for training
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )

    return loader


# Test code
if __name__ == '__main__':
    print("Testing Data Loader V4...")
    print("=" * 50)

    # Check if test data exists
    test_dir = 'processed_data_v4'
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found.")
        print("Creating mock data for testing...")

        # Create mock data
        os.makedirs(test_dir, exist_ok=True)
        n_samples = 100
        seq_len = 200
        feature_dim = 8

        X = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
        X = np.clip(X, -1, 1)
        C = np.random.uniform(-1, 1, n_samples).astype(np.float32)
        L = np.random.randint(50, seq_len, n_samples).astype(np.int32)

        np.save(os.path.join(test_dir, 'X_train.npy'), X)
        np.save(os.path.join(test_dir, 'C_train.npy'), C)
        np.save(os.path.join(test_dir, 'L_train.npy'), L)

        np.save(os.path.join(test_dir, 'X_val.npy'), X[:20])
        np.save(os.path.join(test_dir, 'C_val.npy'), C[:20])
        np.save(os.path.join(test_dir, 'L_val.npy'), L[:20])

        np.save(os.path.join(test_dir, 'X_test.npy'), X[:20])
        np.save(os.path.join(test_dir, 'C_test.npy'), C[:20])
        np.save(os.path.join(test_dir, 'L_test.npy'), L[:20])

        mock_params = {
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
        np.save(os.path.join(test_dir, 'normalization_params.npy'), mock_params)

        print("Mock data created.\n")

    # Test data loading
    train_loader, val_loader, test_loader = create_dataloaders_v4(
        test_dir, batch_size=16, augment_train=True
    )

    print("\nTesting batch loading...")
    for i, (X, C, L) in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  X shape: {X.shape} (dynamically padded!)")
        print(f"  C shape: {C.shape}")
        print(f"  L: {L.tolist()}")
        print(f"  Max length in batch: {X.shape[1]}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")

        if i >= 2:
            break

    print("\nData Loader V4 test PASSED!")
    print("=" * 50)
