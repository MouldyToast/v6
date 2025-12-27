"""
Trajectory Dataset for Diffusion Training V8

V8 Key Changes from V7:
- Loads 2D positions (x, y) instead of 8D features
- No condition loading (no C_*.npy files)
- Returns only 'motion' and 'length' in batch

Data format (from preprocess_V8.py):
    X: (n_samples, max_seq_len, 2) - positions [x, y]
    L: (n_samples,) - original sequence lengths before padding

Outputs batches in format:
    {
        'motion': (batch, seq_len, 2) - padded trajectories
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
    Dataset adapter for V8 preprocessed trajectory data.

    V8: Loads 2D positions only, no conditions.

    Args:
        data_dir: Directory containing preprocessed V8 data
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

        # Load data (V8: no conditions)
        self.X, self.L = self._load_data()

        print(f"Loaded {split} dataset (V8):")
        print(f"  Samples: {len(self)}")
        print(f"  X shape: {self.X.shape} (2D positions)")
        print(f"  L shape: {self.L.shape} (lengths)")
        print(f"  Length range: [{self.L.min()}, {self.L.max()}]")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load X, L from .npy files (no C in V8)."""
        X_path = os.path.join(self.data_dir, f'X_{self.split}.npy')
        L_path = os.path.join(self.data_dir, f'L_{self.split}.npy')

        # Check files exist
        for path in [X_path, L_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

        # Load
        X = np.load(X_path)  # (n, max_seq_len, 2) - V8: 2D positions
        L = np.load(L_path)  # (n,)

        # Validate shapes
        n_samples = len(X)
        assert X.ndim == 3, f"Expected X.ndim=3, got {X.ndim}"
        assert X.shape[2] == 2, f"V8 expects X.shape[2]=2 (x, y positions), got {X.shape[2]}"
        assert L.ndim == 1, f"Expected L.ndim=1, got {L.ndim}"
        assert len(X) == len(L), "X and L must have same length"

        # Validate sequence lengths
        max_len = X.shape[1]
        assert np.all(L <= max_len), f"Some lengths exceed max_seq_len={max_len}"
        assert np.all(L > 0), "All lengths must be > 0"

        return X, L

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        V8: No 'condition' key in output.

        Returns:
            dict with keys:
                'motion': (seq_len, 2) - trajectory positions
                'length': scalar - original sequence length
        """
        return {
            'motion': torch.FloatTensor(self.X[idx]),        # (max_seq_len, 2)
            'length': torch.LongTensor([self.L[idx]])[0]     # scalar
            # NOTE: No 'condition' in V8
        }

    def get_normalization_params(self) -> Optional[Dict]:
        """Load normalization parameters if available."""
        norm_path = os.path.join(self.data_dir, 'normalization_params.npy')
        if os.path.exists(norm_path):
            return np.load(norm_path, allow_pickle=True).item()
        return None

    def get_endpoints(self) -> Optional[np.ndarray]:
        """Load endpoints for evaluation (if available)."""
        endpoints_path = os.path.join(self.data_dir, f'endpoints_{self.split}.npy')
        if os.path.exists(endpoints_path):
            return np.load(endpoints_path)
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
    Create a DataLoader for V8 trajectory data.

    Args:
        data_dir: Directory with preprocessed V8 data
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for train, False for val/test)
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU
        max_seq_len: Maximum sequence length

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = TrajectoryDataset(
        data_dir=data_dir,
        split=split,
        max_seq_len=max_seq_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
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
        data_dir: Directory with preprocessed V8 data
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
    Create a single batch for overfitting test.

    Args:
        data_dir: Directory with preprocessed V8 data
        batch_size: Number of samples in the batch
        split: Which split to use

    Returns:
        batch: Single batch dict with keys 'motion', 'length' (no 'condition' in V8)
        dataset: The full dataset (for reference)
    """
    dataset = TrajectoryDataset(data_dir=data_dir, split=split)

    indices = list(range(min(batch_size, len(dataset))))

    batch = {
        'motion': torch.stack([dataset[i]['motion'] for i in indices]),
        'length': torch.stack([dataset[i]['length'] for i in indices])
        # NOTE: No 'condition' in V8
    }

    print(f"\nCreated single batch for overfitting test (V8):")
    print(f"  Batch size: {len(indices)}")
    print(f"  Motion: {batch['motion'].shape}")
    print(f"  Length: {batch['length'].shape}")
    print(f"  Length range: [{batch['length'].min()}, {batch['length'].max()}]")

    return batch, dataset


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing TrajectoryDataset V8...")
    print("=" * 70)

    data_dir = 'processed_data_v8'
    batch_size = 16

    try:
        print("\n=== Test 1: Dataset Creation ===")
        dataset = TrajectoryDataset(data_dir=data_dir, split='train')

        print(f"Dataset size: {len(dataset)}")

        print("\n=== Test 2: Get Item ===")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  motion: {sample['motion'].shape}")
        print(f"  length: {sample['length']}")
        print(f"  (no 'condition' in V8)")

        print("\n=== Test 3: DataLoader ===")
        dataloader = create_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"  motion: {batch['motion'].shape}")
        print(f"  length: {batch['length'].shape}")

        print("\n=== Test 4: Single Batch (Overfitting Test) ===")
        single_batch, _ = create_single_batch_dataset(
            data_dir=data_dir,
            batch_size=64
        )

        print("\n" + "=" * 70)
        print("ALL V8 TESTS PASSED")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nSkipping tests: {e}")
        print("This is expected if processed_data_v8/ is not available")
        print("The dataset will work when data is present")
