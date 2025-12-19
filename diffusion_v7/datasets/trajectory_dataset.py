"""
Trajectory Dataset for Diffusion Training

Loads V6 preprocessed data and provides it in format for diffusion training.

Data format (from preprocess_V6.py):
    X: (n_samples, max_seq_len, 8) - trajectories [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]
    C: (n_samples, 3) - goal conditions [distance_norm, cos(angle), sin(angle)]
    L: (n_samples,) - original sequence lengths before padding

Outputs batches in format:
    {
        'motion': (batch, seq_len, 8) - padded trajectories
        'condition': (batch, 3) - goal conditions
        'length': (batch,) - original lengths
    }
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional


class TrajectoryDataset(Dataset):
    """
    Dataset adapter for V6 preprocessed trajectory data.

    Loads .npy files and provides them in diffusion training format.

    Args:
        data_dir: Directory containing preprocessed V6 data
        split: 'train', 'val', or 'test'
        max_seq_len: Maximum sequence length (for validation)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_seq_len: int = 200
    ):
        super().__init__()

        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"

        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len

        # Load data
        self.X, self.C, self.L = self._load_data()

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self)}")
        print(f"  X shape: {self.X.shape} (trajectories)")
        print(f"  C shape: {self.C.shape} (conditions)")
        print(f"  L shape: {self.L.shape} (lengths)")
        print(f"  Length range: [{self.L.min()}, {self.L.max()}]")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load X, C, L from .npy files."""
        X_path = os.path.join(self.data_dir, f'X_{self.split}.npy')
        C_path = os.path.join(self.data_dir, f'C_{self.split}.npy')
        L_path = os.path.join(self.data_dir, f'L_{self.split}.npy')

        # Check files exist
        for path in [X_path, C_path, L_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

        # Load
        X = np.load(X_path)  # (n, max_seq_len, 8)
        C = np.load(C_path)  # (n, 3)
        L = np.load(L_path)  # (n,)

        # Validate shapes
        n_samples = len(X)
        assert X.ndim == 3, f"Expected X.ndim=3, got {X.ndim}"
        assert X.shape[2] == 8, f"Expected X.shape[2]=8 (features), got {X.shape[2]}"
        assert C.ndim == 2, f"Expected C.ndim=2, got {C.ndim}"
        assert C.shape[1] == 3, f"Expected C.shape[1]=3 (goal condition), got {C.shape[1]}"
        assert L.ndim == 1, f"Expected L.ndim=1, got {L.ndim}"
        assert len(X) == len(C) == len(L), "X, C, L must have same length"

        # Validate sequence lengths
        max_len = X.shape[1]
        assert np.all(L <= max_len), f"Some lengths exceed max_seq_len={max_len}"
        assert np.all(L > 0), "All lengths must be > 0"

        return X, C, L

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict with keys:
                'motion': (seq_len, 8) - trajectory features
                'condition': (3,) - goal condition
                'length': scalar - original sequence length
        """
        return {
            'motion': torch.FloatTensor(self.X[idx]),        # (max_seq_len, 8)
            'condition': torch.FloatTensor(self.C[idx]),     # (3,)
            'length': torch.LongTensor([self.L[idx]])[0]     # scalar
        }

    def get_normalization_params(self) -> Dict:
        """Load normalization parameters if available."""
        norm_path = os.path.join(self.data_dir, 'normalization_params.npy')
        if os.path.exists(norm_path):
            return np.load(norm_path, allow_pickle=True).item()
        return None


def create_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 64,
    shuffle: bool = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_len: int = 200
) -> DataLoader:
    """
    Create a DataLoader for trajectory data.

    Args:
        data_dir: Directory with preprocessed V6 data
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for train, False for val/test)
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU
        max_seq_len: Maximum sequence length

    Returns:
        DataLoader instance
    """
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')

    # Create dataset
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        split=split,
        max_seq_len=max_seq_len
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')  # Drop incomplete batches for training
    )

    return dataloader


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_len: int = 200
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.

    Args:
        data_dir: Directory with preprocessed V6 data
        batch_size: Batch size
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU
        max_seq_len: Maximum sequence length

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_seq_len=max_seq_len
    )

    val_loader = create_dataloader(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_seq_len=max_seq_len
    )

    test_loader = create_dataloader(
        data_dir=data_dir,
        split='test',
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_seq_len=max_seq_len
    )

    return train_loader, val_loader, test_loader


def create_single_batch_dataset(
    data_dir: str,
    batch_size: int = 64,
    split: str = 'train'
) -> Tuple[Dict[str, torch.Tensor], TrajectoryDataset]:
    """
    Create a single batch for overfitting test (Phase 2.1).

    Loads the first batch_size samples from the dataset and returns them
    as a fixed batch. Used for smoke testing to verify the model can
    overfit on a small amount of data.

    Args:
        data_dir: Directory with preprocessed V6 data
        batch_size: Number of samples in the batch
        split: Which split to use

    Returns:
        batch: Single batch dict with keys 'motion', 'condition', 'length'
        dataset: The full dataset (for reference)
    """
    dataset = TrajectoryDataset(data_dir=data_dir, split=split)

    # Get first batch_size samples
    indices = list(range(min(batch_size, len(dataset))))

    batch = {
        'motion': torch.stack([dataset[i]['motion'] for i in indices]),
        'condition': torch.stack([dataset[i]['condition'] for i in indices]),
        'length': torch.stack([dataset[i]['length'] for i in indices])
    }

    print(f"\nCreated single batch for overfitting test:")
    print(f"  Batch size: {len(indices)}")
    print(f"  Motion: {batch['motion'].shape}")
    print(f"  Condition: {batch['condition'].shape}")
    print(f"  Length: {batch['length'].shape}")
    print(f"  Length range: [{batch['length'].min()}, {batch['length'].max()}]")

    return batch, dataset


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing TrajectoryDataset...")
    print("=" * 70)

    # NOTE: This will fail without actual data files
    # Uncomment if you have processed_data_v6/ available

    # Test parameters
    data_dir = 'processed_data_v6'
    batch_size = 16

    try:
        # Test dataset creation
        print("\n=== Test 1: Dataset Creation ===")
        dataset = TrajectoryDataset(data_dir=data_dir, split='train')

        print(f"Dataset size: {len(dataset)}")

        # Test __getitem__
        print("\n=== Test 2: Get Item ===")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  motion: {sample['motion'].shape}")
        print(f"  condition: {sample['condition'].shape}")
        print(f"  length: {sample['length']}")

        # Test dataloader
        print("\n=== Test 3: DataLoader ===")
        dataloader = create_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # 0 for testing
        )

        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"  motion: {batch['motion'].shape}")
        print(f"  condition: {batch['condition'].shape}")
        print(f"  length: {batch['length'].shape}")

        # Test single batch creation
        print("\n=== Test 4: Single Batch (Overfitting Test) ===")
        single_batch, _ = create_single_batch_dataset(
            data_dir=data_dir,
            batch_size=64
        )

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nSkipping tests: {e}")
        print("This is expected if processed_data_v6/ is not available")
        print("The dataset will work when data is present")
