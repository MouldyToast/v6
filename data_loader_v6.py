"""
Data Loader for TimeGAN V6 - Optimized for Two-Stage Training

This module provides V6-specific data loading optimizations:
1. Condition-Stratified Sampling: Uniform condition distribution for WGAN-GP (Stage 2)
2. Length-Aware Batching: Group similar-length sequences for efficient autoencoder training (Stage 1)
3. Quality Filtering: Remove outlier trajectories for cleaner latent space learning
4. Stage-Specific Loaders: Different strategies for Stage 1 vs Stage 2

Uses V4 preprocessed data format:
    V4 Features (8 dimensions):
        [0] dx       - Relative x from start
        [1] dy       - Relative y from start
        [2] speed    - Movement speed (px/s)
        [3] accel    - Acceleration (px/s^2)
        [4] sin_h    - Sine of heading
        [5] cos_h    - Cosine of heading
        [6] ang_vel  - Angular velocity (rad/s)
        [7] dt       - Time delta (s)

Usage:
    from data_loader_v6 import create_dataloaders_v6, create_stage_loaders

    # Full pipeline
    train_loader, val_loader, test_loader = create_dataloaders_v6(data_dir)

    # Stage-specific loaders
    stage1_loader, stage2_loader = create_stage_loaders(data_dir)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from typing import Tuple, List, Dict, Optional, Iterator
from collections import defaultdict
import warnings


# =============================================================================
# V6-Optimized Dataset
# =============================================================================

class MouseTrajectoryDatasetV6(Dataset):
    """
    PyTorch Dataset for V6 mouse trajectories with quality filtering.

    Enhancements over V4:
    - Quality filtering (removes outliers)
    - Trajectory validation
    - Condition bin assignment for stratified sampling
    - Length bin assignment for length-aware batching
    """

    def __init__(self, X: np.ndarray, C: np.ndarray, L: np.ndarray,
                 augment: bool = False,
                 min_length: int = 10,
                 max_length: int = None,
                 quality_filter: bool = True,
                 quality_threshold: float = 3.0,
                 n_condition_bins: int = 10,
                 n_length_bins: int = 5):
        """
        Args:
            X: (n_samples, max_seq_len, 8) - padded feature arrays
            C: (n_samples,) - normalized distance conditions
            L: (n_samples,) - original lengths before padding
            augment: Whether to apply data augmentation
            min_length: Minimum sequence length (shorter sequences filtered)
            max_length: Maximum sequence length (longer sequences truncated)
            quality_filter: Whether to filter outlier trajectories
            quality_threshold: Z-score threshold for outlier detection
            n_condition_bins: Number of bins for condition stratification
            n_length_bins: Number of bins for length grouping
        """
        # Validate input
        assert X.ndim == 3, f"X must be 3D, got shape {X.shape}"
        assert X.shape[2] == 8, f"Expected feature_dim=8, got {X.shape[2]}"
        assert len(X) == len(C) == len(L), "X, C, L must have same length"

        self.augment = augment
        self.min_length = min_length
        self.max_length = max_length or X.shape[1]
        self.n_condition_bins = n_condition_bins
        self.n_length_bins = n_length_bins

        # Filter and store data
        X_filtered, C_filtered, L_filtered, filter_stats = self._filter_data(
            X, C, L, quality_filter, quality_threshold
        )

        self.X = torch.FloatTensor(X_filtered)
        self.C = torch.FloatTensor(C_filtered).unsqueeze(-1)  # (n, 1)
        self.L = torch.LongTensor(L_filtered)
        self.filter_stats = filter_stats

        # Compute bin assignments
        self.condition_bins = self._compute_condition_bins(C_filtered)
        self.length_bins = self._compute_length_bins(L_filtered)

        # Build indices for each bin (for stratified sampling)
        self.condition_bin_indices = self._build_bin_indices(self.condition_bins)
        self.length_bin_indices = self._build_bin_indices(self.length_bins)

    def _filter_data(self, X: np.ndarray, C: np.ndarray, L: np.ndarray,
                     quality_filter: bool, quality_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Filter data based on quality criteria.

        Returns:
            X_filtered, C_filtered, L_filtered, filter_stats
        """
        n_original = len(X)
        valid_mask = np.ones(n_original, dtype=bool)
        stats = {'original': n_original}

        # Filter by length
        length_valid = (L >= self.min_length) & (L <= self.max_length)
        valid_mask &= length_valid
        stats['removed_length'] = n_original - length_valid.sum()

        # Quality filter: remove trajectories with extreme feature values
        if quality_filter:
            # Compute per-trajectory statistics (mean of absolute values)
            traj_stats = []
            for i in range(n_original):
                if valid_mask[i]:
                    length = L[i]
                    traj = X[i, :length, :]
                    # Mean absolute value per feature
                    mean_abs = np.abs(traj).mean(axis=0)
                    traj_stats.append(mean_abs)
                else:
                    traj_stats.append(np.zeros(8))

            traj_stats = np.array(traj_stats)

            # Compute global statistics for valid trajectories
            valid_stats = traj_stats[valid_mask]
            if len(valid_stats) > 0:
                global_mean = valid_stats.mean(axis=0)
                global_std = valid_stats.std(axis=0) + 1e-8

                # Z-score for each trajectory
                z_scores = np.abs((traj_stats - global_mean) / global_std)
                max_z = z_scores.max(axis=1)

                # Filter outliers
                quality_valid = max_z < quality_threshold
                quality_valid[~valid_mask] = True  # Don't double-count already filtered

                stats['removed_quality'] = valid_mask.sum() - (valid_mask & quality_valid).sum()
                valid_mask &= quality_valid
        else:
            stats['removed_quality'] = 0

        stats['remaining'] = valid_mask.sum()
        stats['removed_total'] = n_original - valid_mask.sum()

        # Apply filter
        X_filtered = X[valid_mask]
        C_filtered = C[valid_mask]
        L_filtered = L[valid_mask]

        # Clamp lengths to max_length
        L_filtered = np.minimum(L_filtered, self.max_length)

        return X_filtered, C_filtered, L_filtered, stats

    def _compute_condition_bins(self, C: np.ndarray) -> np.ndarray:
        """
        Assign each sample to a condition bin for stratified sampling.

        Uses quantile-based binning to handle non-uniform distributions.
        """
        # Use quantile-based binning for robustness
        percentiles = np.linspace(0, 100, self.n_condition_bins + 1)
        bin_edges = np.percentile(C, percentiles)

        # Handle edge cases where all values are same
        if bin_edges[0] == bin_edges[-1]:
            return np.zeros(len(C), dtype=np.int64)

        # Digitize (bin 0 to n_bins-1)
        bins = np.digitize(C, bin_edges[1:-1])

        return bins.astype(np.int64)

    def _compute_length_bins(self, L: np.ndarray) -> np.ndarray:
        """
        Assign each sample to a length bin for length-aware batching.
        """
        percentiles = np.linspace(0, 100, self.n_length_bins + 1)
        bin_edges = np.percentile(L, percentiles)

        if bin_edges[0] == bin_edges[-1]:
            return np.zeros(len(L), dtype=np.int64)

        bins = np.digitize(L, bin_edges[1:-1])

        return bins.astype(np.int64)

    def _build_bin_indices(self, bins: np.ndarray) -> Dict[int, List[int]]:
        """Build mapping from bin ID to list of sample indices."""
        indices = defaultdict(list)
        for i, bin_id in enumerate(bins):
            indices[int(bin_id)].append(i)
        return dict(indices)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        Returns:
            x: (original_length, 8) - unpadded trajectory
            c: (1,) - distance condition
            length: int - original length
            condition_bin: int - condition bin assignment
            length_bin: int - length bin assignment
        """
        length = self.L[idx].item()

        # Extract only real data (no padding)
        x = self.X[idx, :length, :].clone()
        c = self.C[idx].clone()

        # Optional augmentation
        if self.augment:
            x = self._augment(x)

        return x, c, length, self.condition_bins[idx], self.length_bins[idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to trajectory.

        V6 augmentation (same as V4 but with additional temporal jitter option).
        """
        # Random horizontal flip (50% chance)
        if torch.rand(1).item() > 0.5:
            x = x.clone()
            x[:, 0] = -x[:, 0]  # Flip dx
            x[:, 4] = -x[:, 4]  # Flip sin(heading)
            x[:, 6] = -x[:, 6]  # Flip angular velocity

        # Random vertical flip (50% chance)
        if torch.rand(1).item() > 0.5:
            x = x.clone()
            x[:, 1] = -x[:, 1]   # Flip dy
            x[:, 4] = -x[:, 4]   # Flip sin(heading)
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

    def get_condition_weights(self) -> torch.Tensor:
        """
        Get sampling weights for uniform condition distribution.

        Returns weights inversely proportional to bin frequency.
        """
        bin_counts = np.bincount(self.condition_bins, minlength=self.n_condition_bins)
        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)

        # Weight = 1 / (bin_count * n_bins) to normalize
        bin_weights = 1.0 / (bin_counts * self.n_condition_bins)

        # Assign weight to each sample based on its bin
        weights = torch.FloatTensor([bin_weights[bin_id] for bin_id in self.condition_bins])

        return weights

    def get_length_order(self) -> np.ndarray:
        """
        Get indices sorted by length (for length-aware batching).

        Returns indices sorted in descending order of length.
        """
        return np.argsort(-self.L.numpy())


# =============================================================================
# Stratified Sampler for Stage 2 (WGAN-GP)
# =============================================================================

class ConditionStratifiedSampler(Sampler):
    """
    Sampler that ensures uniform condition distribution in each batch.

    Samples uniformly from each condition bin, ensuring WGAN-GP sees
    diverse conditions in each batch.
    """

    def __init__(self, dataset: MouseTrajectoryDatasetV6, batch_size: int,
                 drop_last: bool = True):
        """
        Args:
            dataset: MouseTrajectoryDatasetV6 instance
            batch_size: Batch size
            drop_last: Whether to drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.bin_indices = dataset.condition_bin_indices
        self.n_bins = len(self.bin_indices)

        # Calculate samples per bin per batch
        self.samples_per_bin = max(1, batch_size // self.n_bins)
        self.effective_batch_size = self.samples_per_bin * self.n_bins

    def __iter__(self) -> Iterator[int]:
        # Shuffle indices within each bin
        bin_indices_shuffled = {}
        for bin_id, indices in self.bin_indices.items():
            shuffled = np.random.permutation(indices)
            bin_indices_shuffled[bin_id] = list(shuffled)

        # Track position in each bin
        bin_positions = {bin_id: 0 for bin_id in self.bin_indices}

        # Generate batches
        n_batches = len(self) if self.drop_last else (len(self.dataset) + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            batch = []

            for bin_id in sorted(self.bin_indices.keys()):
                indices = bin_indices_shuffled[bin_id]
                pos = bin_positions[bin_id]

                # Get samples from this bin
                for _ in range(self.samples_per_bin):
                    if pos >= len(indices):
                        # Reshuffle and reset
                        indices = list(np.random.permutation(self.bin_indices[bin_id]))
                        bin_indices_shuffled[bin_id] = indices
                        pos = 0

                    batch.append(indices[pos])
                    pos += 1

                bin_positions[bin_id] = pos

            # Shuffle the batch
            np.random.shuffle(batch)

            yield from batch

    def __len__(self) -> int:
        if self.drop_last:
            return (len(self.dataset) // self.effective_batch_size) * self.effective_batch_size
        return len(self.dataset)


# =============================================================================
# Length-Aware Sampler for Stage 1 (Autoencoder)
# =============================================================================

class LengthAwareSampler(Sampler):
    """
    Sampler that groups sequences of similar lengths together.

    Reduces padding waste during autoencoder training by ensuring
    sequences in each batch have similar lengths.
    """

    def __init__(self, dataset: MouseTrajectoryDatasetV6, batch_size: int,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Args:
            dataset: MouseTrajectoryDatasetV6 instance
            batch_size: Batch size
            shuffle: Whether to shuffle within length bins
            drop_last: Whether to drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.length_bins = dataset.length_bin_indices

    def __iter__(self) -> Iterator[int]:
        # Get indices grouped by length bin
        all_indices = []

        for bin_id in sorted(self.length_bins.keys()):
            indices = self.length_bins[bin_id].copy()

            if self.shuffle:
                np.random.shuffle(indices)

            all_indices.extend(indices)

        # Shuffle bins order (but keep items within bins together)
        if self.shuffle:
            # Group into batches first
            batches = []
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

            # Shuffle batch order
            np.random.shuffle(batches)

            # Flatten
            for batch in batches:
                yield from batch
        else:
            yield from all_indices

    def __len__(self) -> int:
        if self.drop_last:
            return (len(self.dataset) // self.batch_size) * self.batch_size
        return len(self.dataset)


# =============================================================================
# Collate Functions
# =============================================================================

def collate_fn_v6(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for V6 data with dynamic batching.

    Pads each batch to its own maximum length (not fixed padding).

    Args:
        batch: List of (x, c, length, condition_bin, length_bin) tuples

    Returns:
        X_padded: (batch_size, max_len_in_batch, 8)
        C: (batch_size, 1)
        lengths: (batch_size,)
    """
    # Sort by length (descending) for pack_padded_sequence compatibility
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    # Extract components
    xs, cs, lengths, _, _ = zip(*batch)

    # Find max length in this batch
    max_len = max(lengths)

    # Pad sequences
    batch_size = len(xs)
    feature_dim = xs[0].size(-1)

    X_padded = torch.zeros(batch_size, max_len, feature_dim)
    for i, (x, length) in enumerate(zip(xs, lengths)):
        X_padded[i, :length, :] = x

    # Stack conditions and lengths
    C = torch.stack(cs)
    lengths = torch.LongTensor(lengths)

    return X_padded, C, lengths


def collate_fn_v6_full(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extended collate function that also returns bin assignments.

    Useful for analysis and debugging.

    Returns:
        X_padded, C, lengths, condition_bins, length_bins
    """
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    xs, cs, lengths, condition_bins, length_bins = zip(*batch)

    max_len = max(lengths)
    batch_size = len(xs)
    feature_dim = xs[0].size(-1)

    X_padded = torch.zeros(batch_size, max_len, feature_dim)
    for i, (x, length) in enumerate(zip(xs, lengths)):
        X_padded[i, :length, :] = x

    C = torch.stack(cs)
    lengths = torch.LongTensor(lengths)
    condition_bins = torch.LongTensor(condition_bins)
    length_bins = torch.LongTensor(length_bins)

    return X_padded, C, lengths, condition_bins, length_bins


# =============================================================================
# DataLoader Factory Functions
# =============================================================================

def create_dataloaders_v6(data_dir: str, batch_size: int = 64,
                          num_workers: int = 0, augment_train: bool = True,
                          quality_filter: bool = True,
                          min_length: int = 10,
                          max_length: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders for V6 data.

    This is the standard loader - for stage-specific optimizations,
    use create_stage_loaders().

    Args:
        data_dir: Directory containing preprocessed V4 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data
        quality_filter: Whether to filter outlier trajectories
        min_length: Minimum sequence length
        max_length: Maximum sequence length (None = use data max)

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Loading V6 data from: {data_dir}")

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

    # Validate V4 format
    assert X_train.shape[2] == 8, f"Expected feature_dim=8, got {X_train.shape[2]}"

    # Create datasets
    train_dataset = MouseTrajectoryDatasetV6(
        X_train, C_train, L_train,
        augment=augment_train,
        min_length=min_length,
        max_length=max_length,
        quality_filter=quality_filter
    )

    val_dataset = MouseTrajectoryDatasetV6(
        X_val, C_val, L_val,
        augment=False,
        min_length=min_length,
        max_length=max_length,
        quality_filter=False  # Don't filter validation
    )

    test_dataset = MouseTrajectoryDatasetV6(
        X_test, C_test, L_test,
        augment=False,
        min_length=min_length,
        max_length=max_length,
        quality_filter=False  # Don't filter test
    )

    # Print statistics
    print(f"\n  Train: {len(train_dataset)} samples (filtered from {train_dataset.filter_stats['original']})")
    if train_dataset.filter_stats['removed_total'] > 0:
        print(f"    - Removed by length: {train_dataset.filter_stats['removed_length']}")
        print(f"    - Removed by quality: {train_dataset.filter_stats['removed_quality']}")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Feature dim: 8 (V6 compatible)")

    # Create dataloaders with standard sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_v6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v6,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_v6,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


def create_stage1_loader(data_dir: str, batch_size: int = 64,
                         num_workers: int = 0, augment: bool = True,
                         quality_filter: bool = True,
                         min_length: int = 10,
                         max_length: int = None) -> DataLoader:
    """
    Create DataLoader optimized for Stage 1 (Autoencoder) training.

    Uses length-aware batching to reduce padding waste.

    Args:
        data_dir: Directory containing preprocessed V4 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to augment data
        quality_filter: Whether to filter outlier trajectories
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        stage1_loader: DataLoader with length-aware batching
    """
    print("Creating Stage 1 (Autoencoder) DataLoader...")
    print("  Optimization: Length-aware batching (reduced padding)")

    # Load training data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    C_train = np.load(os.path.join(data_dir, 'C_train.npy'))
    L_train = np.load(os.path.join(data_dir, 'L_train.npy'))

    # Create dataset
    dataset = MouseTrajectoryDatasetV6(
        X_train, C_train, L_train,
        augment=augment,
        min_length=min_length,
        max_length=max_length,
        quality_filter=quality_filter
    )

    print(f"  Samples: {len(dataset)} (filtered from {dataset.filter_stats['original']})")
    print(f"  Length bins: {dataset.n_length_bins}")

    # Create length-aware sampler
    sampler = LengthAwareSampler(dataset, batch_size, shuffle=True, drop_last=True)

    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_v6,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # Compute efficiency
    _print_efficiency_stats(dataset, batch_size, "Stage 1")

    return loader


def create_stage2_loader(data_dir: str, batch_size: int = 64,
                         num_workers: int = 0, augment: bool = False,
                         quality_filter: bool = True,
                         min_length: int = 10,
                         max_length: int = None) -> DataLoader:
    """
    Create DataLoader optimized for Stage 2 (WGAN-GP) training.

    Uses condition-stratified sampling for uniform condition distribution.

    Args:
        data_dir: Directory containing preprocessed V4 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to augment data (usually False for GAN)
        quality_filter: Whether to filter outlier trajectories
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        stage2_loader: DataLoader with condition-stratified sampling
    """
    print("Creating Stage 2 (WGAN-GP) DataLoader...")
    print("  Optimization: Condition-stratified sampling (uniform conditions)")

    # Load training data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    C_train = np.load(os.path.join(data_dir, 'C_train.npy'))
    L_train = np.load(os.path.join(data_dir, 'L_train.npy'))

    # Create dataset
    dataset = MouseTrajectoryDatasetV6(
        X_train, C_train, L_train,
        augment=augment,
        min_length=min_length,
        max_length=max_length,
        quality_filter=quality_filter
    )

    print(f"  Samples: {len(dataset)} (filtered from {dataset.filter_stats['original']})")
    print(f"  Condition bins: {dataset.n_condition_bins}")

    # Print condition distribution
    _print_condition_distribution(dataset)

    # Create condition-stratified sampler
    sampler = ConditionStratifiedSampler(dataset, batch_size, drop_last=True)

    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_v6,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    return loader


def create_stage_loaders(data_dir: str, batch_size: int = 64,
                         num_workers: int = 0,
                         quality_filter: bool = True,
                         min_length: int = 10,
                         max_length: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create both stage-specific DataLoaders.

    Args:
        data_dir: Directory containing preprocessed V4 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        quality_filter: Whether to filter outlier trajectories
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        stage1_loader: Length-aware batching for autoencoder
        stage2_loader: Condition-stratified for WGAN-GP
    """
    stage1_loader = create_stage1_loader(
        data_dir, batch_size, num_workers,
        augment=True, quality_filter=quality_filter,
        min_length=min_length, max_length=max_length
    )

    stage2_loader = create_stage2_loader(
        data_dir, batch_size, num_workers,
        augment=False, quality_filter=quality_filter,
        min_length=min_length, max_length=max_length
    )

    return stage1_loader, stage2_loader


# =============================================================================
# Utility Functions
# =============================================================================

def _print_efficiency_stats(dataset: MouseTrajectoryDatasetV6, batch_size: int, stage: str):
    """Print padding efficiency statistics."""
    lengths = dataset.L.numpy()

    # Fixed padding efficiency (worst case)
    max_len = lengths.max()
    fixed_efficiency = (lengths.sum() / (len(lengths) * max_len)) * 100

    # Estimate length-aware efficiency
    # Group by length bins and compute efficiency per bin
    length_aware_total = 0
    length_aware_count = 0

    for bin_id, indices in dataset.length_bin_indices.items():
        bin_lengths = lengths[indices]
        bin_max = bin_lengths.max()
        bin_total = bin_lengths.sum()
        length_aware_total += bin_total
        length_aware_count += len(indices) * bin_max

    length_aware_efficiency = (length_aware_total / length_aware_count) * 100 if length_aware_count > 0 else 0

    print(f"\n  {stage} Efficiency:")
    print(f"    Fixed padding: {fixed_efficiency:.1f}%")
    print(f"    Length-aware:  {length_aware_efficiency:.1f}%")
    print(f"    Improvement:   {length_aware_efficiency - fixed_efficiency:.1f}%")


def _print_condition_distribution(dataset: MouseTrajectoryDatasetV6):
    """Print condition distribution across bins."""
    print("\n  Condition distribution:")

    total = len(dataset)
    for bin_id in sorted(dataset.condition_bin_indices.keys()):
        count = len(dataset.condition_bin_indices[bin_id])
        pct = (count / total) * 100
        bar = "#" * int(pct / 2)
        print(f"    Bin {bin_id}: {count:5d} ({pct:5.1f}%) {bar}")


def load_normalization_params(data_dir: str) -> Dict:
    """
    Load normalization parameters for denormalization.

    Args:
        data_dir: Directory containing normalization_params.npy

    Returns:
        Dictionary of normalization parameters
    """
    path = os.path.join(data_dir, 'normalization_params.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    else:
        warnings.warn(f"Normalization params not found at {path}")
        return {}


def load_v6_data(data_dir: str) -> Dict:
    """
    Load all V6-compatible data into a dictionary.

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
    data['norm_params'] = load_normalization_params(data_dir)

    return data


def validate_v6_data(data_dir: str) -> bool:
    """
    Validate that data directory contains V6-compatible data.

    Args:
        data_dir: Directory to validate

    Returns:
        True if valid, raises exception otherwise
    """
    required_files = [
        'X_train.npy', 'C_train.npy', 'L_train.npy',
        'X_val.npy', 'C_val.npy', 'L_val.npy',
        'X_test.npy', 'C_test.npy', 'L_test.npy',
    ]

    for fname in required_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Check feature dimension
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    if X_train.shape[2] != 8:
        raise ValueError(f"Expected feature_dim=8, got {X_train.shape[2]}")

    # Check value ranges
    if X_train.min() < -1.5 or X_train.max() > 1.5:
        warnings.warn(f"Data range [{X_train.min():.2f}, {X_train.max():.2f}] exceeds expected [-1, 1]")

    print(f"V6 data validation passed: {data_dir}")
    return True


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing Data Loader V6...")
    print("=" * 60)

    # Create test directory with mock data
    test_dir = 'test_data_v6'
    os.makedirs(test_dir, exist_ok=True)

    # Create mock data with varying conditions and lengths
    np.random.seed(42)
    n_samples = 200
    seq_len = 150
    feature_dim = 8

    # Create data with non-uniform condition distribution
    # (more samples at low distances)
    C = np.random.beta(2, 5, n_samples)  # Skewed distribution
    C = C * 2 - 1  # Map to [-1, 1]

    # Create lengths correlated with condition (longer for larger distances)
    base_lengths = 30 + (C + 1) / 2 * 80  # 30-110 based on condition
    noise = np.random.randint(-20, 20, n_samples)
    L = np.clip(base_lengths + noise, 20, seq_len).astype(np.int32)

    # Create features
    X = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    X = np.clip(X, -1, 1)

    # Add some outliers
    X[0, :, :] = 0.99  # Constant trajectory
    X[1, :, :] = np.random.randn(seq_len, feature_dim) * 3  # High variance

    # Save mock data
    split_idx = int(n_samples * 0.8)
    val_idx = int(n_samples * 0.9)

    np.save(os.path.join(test_dir, 'X_train.npy'), X[:split_idx])
    np.save(os.path.join(test_dir, 'C_train.npy'), C[:split_idx])
    np.save(os.path.join(test_dir, 'L_train.npy'), L[:split_idx])

    np.save(os.path.join(test_dir, 'X_val.npy'), X[split_idx:val_idx])
    np.save(os.path.join(test_dir, 'C_val.npy'), C[split_idx:val_idx])
    np.save(os.path.join(test_dir, 'L_val.npy'), L[split_idx:val_idx])

    np.save(os.path.join(test_dir, 'X_test.npy'), X[val_idx:])
    np.save(os.path.join(test_dir, 'C_test.npy'), C[val_idx:])
    np.save(os.path.join(test_dir, 'L_test.npy'), L[val_idx:])

    print("Mock data created.\n")

    # Test validation
    print("=== Testing Validation ===")
    validate_v6_data(test_dir)
    print("Validation: PASS\n")

    # Test standard loaders
    print("=== Testing Standard DataLoaders ===")
    train_loader, val_loader, test_loader = create_dataloaders_v6(
        test_dir, batch_size=16, quality_filter=True
    )

    print("\nTesting batch iteration...")
    for i, (X_batch, C_batch, L_batch) in enumerate(train_loader):
        print(f"  Batch {i+1}: X={X_batch.shape}, C={C_batch.shape}, L={L_batch.tolist()[:5]}...")
        if i >= 2:
            break
    print("Standard loaders: PASS\n")

    # Test stage-specific loaders
    print("=== Testing Stage-Specific Loaders ===")

    # Stage 1
    stage1_loader = create_stage1_loader(test_dir, batch_size=16)
    print("\nStage 1 (length-aware) batch iteration...")
    for i, (X_batch, C_batch, L_batch) in enumerate(stage1_loader):
        lengths = L_batch.tolist()
        max_len = X_batch.size(1)
        efficiency = sum(lengths) / (len(lengths) * max_len) * 100
        print(f"  Batch {i+1}: max_len={max_len}, lengths={lengths[:5]}..., efficiency={efficiency:.1f}%")
        if i >= 2:
            break
    print("Stage 1 loader: PASS\n")

    # Stage 2
    stage2_loader = create_stage2_loader(test_dir, batch_size=16)
    print("\nStage 2 (condition-stratified) batch iteration...")
    for i, (X_batch, C_batch, L_batch) in enumerate(stage2_loader):
        conditions = C_batch.squeeze().tolist()
        print(f"  Batch {i+1}: conditions={[f'{c:.2f}' for c in conditions[:5]]}...")
        if i >= 2:
            break
    print("Stage 2 loader: PASS\n")

    # Test both loaders together
    print("=== Testing Combined Stage Loaders ===")
    stage1, stage2 = create_stage_loaders(test_dir, batch_size=16)
    print(f"Stage 1 loader batches: {len(stage1)}")
    print(f"Stage 2 loader batches: {len(stage2)}")
    print("Combined loaders: PASS\n")

    # Test dataset features
    print("=== Testing Dataset Features ===")
    dataset = MouseTrajectoryDatasetV6(
        X[:split_idx], C[:split_idx], L[:split_idx],
        quality_filter=True
    )

    print(f"Filter stats: {dataset.filter_stats}")
    print(f"Condition bins: {len(dataset.condition_bin_indices)}")
    print(f"Length bins: {len(dataset.length_bin_indices)}")

    weights = dataset.get_condition_weights()
    print(f"Condition weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print("Dataset features: PASS\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("Test directory cleaned up.\n")

    print("=" * 60)
    print("ALL DATA LOADER V6 TESTS PASSED")
    print("=" * 60)
