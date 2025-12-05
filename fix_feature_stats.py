"""
Post-process generated trajectories to match real feature statistics.
This is a quick fix for the low-variance decoder problem.
"""

import numpy as np
import torch
from timegan_v6 import TimeGANV6, get_default_config
from data_loader_v6 import load_v6_data

def match_feature_statistics(X_fake, X_real, feature_indices=None):
    """
    Adjust generated features to match real statistics (mean, std).

    Args:
        X_fake: Generated trajectories (N, T, 8)
        X_real: Real trajectories (N, T, 8)
        feature_indices: Which features to adjust (default: [3, 6, 7] = accel, ang_vel, dt)

    Returns:
        X_adjusted: Trajectories with corrected statistics
    """
    if feature_indices is None:
        # Fix the problematic temporal features
        feature_indices = [3, 6, 7]  # accel, ang_vel, dt

    X_adjusted = X_fake.copy()

    for i in feature_indices:
        # Compute statistics
        real_mean = X_real[:, :, i].mean()
        real_std = X_real[:, :, i].std()
        fake_mean = X_fake[:, :, i].mean()
        fake_std = X_fake[:, :, i].std()

        # Standardize then rescale
        X_adjusted[:, :, i] = (X_fake[:, :, i] - fake_mean) / (fake_std + 1e-8)
        X_adjusted[:, :, i] = X_adjusted[:, :, i] * real_std + real_mean

        # Clip to valid range
        X_adjusted[:, :, i] = np.clip(X_adjusted[:, :, i], -1, 1)

    return X_adjusted


# Demo
CHECKPOINT_PATH = "./checkpoints/v6/run_20251205_100119/final.pt"
DATA_DIR = "processed_data_v4"

print("Loading model and data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = get_default_config()
config.summary_dim = 192
config.pool_type = "hybrid"
config.latent_dim = 64
config.expand_type = "mlp"

model = TimeGANV6(config)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

data = load_v6_data(DATA_DIR)
X_test = data['X_test'][:1000]
C_test = data['C_test'][:1000]

print("Generating...")
with torch.no_grad():
    C_gen = torch.from_numpy(C_test).float().unsqueeze(1).to(device)
    X_fake = model.generate(1000, C_gen, seq_len=100)
    X_fake_np = X_fake.cpu().numpy()

print("Adjusting feature statistics...")
X_adjusted = match_feature_statistics(X_fake_np, X_test, feature_indices=[2, 3, 6, 7])

# Compare
FEATURE_NAMES = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
print("\n" + "="*80)
print("Before vs After Adjustment")
print("="*80)
print(f"{'Feature':<10} {'Real Mean':<12} {'Before Mean':<12} {'After Mean':<12} {'Real Std':<10} {'After Std':<10}")
print("-"*80)

for i, name in enumerate(FEATURE_NAMES):
    real_mean = X_test[:, :, i].mean()
    real_std = X_test[:, :, i].std()
    before_mean = X_fake_np[:, :, i].mean()
    after_mean = X_adjusted[:, :, i].mean()
    after_std = X_adjusted[:, :, i].std()

    marker = "✓" if i in [2, 3, 6, 7] else " "
    print(f"{marker} {name:<9} {real_mean:>11.4f} {before_mean:>11.4f} {after_mean:>11.4f} {real_std:>9.4f} {after_std:>9.4f}")

print("\nSaving adjusted samples...")
np.save("adjusted_samples.npy", X_adjusted)
print("Saved: adjusted_samples.npy")
