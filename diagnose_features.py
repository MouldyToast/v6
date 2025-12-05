"""
Diagnose feature-level differences between real and generated trajectories.
This helps identify which specific features are causing the high discriminative score.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from timegan_v6 import TimeGANV6
from data_loader_v6 import load_v6_data

# Configuration
CHECKPOINT_PATH = "./checkpoints/v6/run_20251205_100119/final.pt"
DATA_DIR = "processed_data_v4"
N_SAMPLES = 1000  # Number of samples to generate for comparison

FEATURE_NAMES = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

print("="*70)
print(" Feature-Level Diagnostic Analysis")
print("="*70)
print()

# Load model
print("Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
config = checkpoint['config']
model = TimeGANV6(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load real data
print("Loading real test data...")
data = load_v6_data(DATA_DIR)
X_test = data['X_test'][:N_SAMPLES]  # (N, T, 8)
C_test = data['C_test'][:N_SAMPLES]  # (N,)
L_test = data['L_test'][:N_SAMPLES]  # (N,)

# Generate fake data
print(f"Generating {N_SAMPLES} synthetic trajectories...")
with torch.no_grad():
    C_gen = torch.from_numpy(C_test).float().unsqueeze(1).to(device)
    max_len = int(L_test.max())
    X_gen = model.generate(N_SAMPLES, C_gen, seq_len=max_len)
    X_gen = X_gen.cpu().numpy()

print()
print("="*70)
print(" Feature Statistics Comparison")
print("="*70)
print()

# Compute statistics for each feature
print(f"{'Feature':<10} {'Real Mean':<12} {'Fake Mean':<12} {'Diff':<10} {'Real Std':<12} {'Fake Std':<12} {'Diff':<10}")
print("-"*90)

feature_stats = []
for i, feat_name in enumerate(FEATURE_NAMES):
    real_feat = X_test[:, :, i].flatten()
    fake_feat = X_gen[:, :, i].flatten()

    real_mean = real_feat.mean()
    fake_mean = fake_feat.mean()
    mean_diff = abs(real_mean - fake_mean)

    real_std = real_feat.std()
    fake_std = fake_feat.std()
    std_diff = abs(real_std - fake_std)

    print(f"{feat_name:<10} {real_mean:>11.4f} {fake_mean:>11.4f} {mean_diff:>9.4f} {real_std:>11.4f} {fake_std:>11.4f} {std_diff:>9.4f}")

    feature_stats.append({
        'name': feat_name,
        'real_mean': real_mean,
        'fake_mean': fake_mean,
        'mean_diff': mean_diff,
        'real_std': real_std,
        'fake_std': fake_std,
        'std_diff': std_diff,
        'real_min': real_feat.min(),
        'fake_min': fake_feat.min(),
        'real_max': real_feat.max(),
        'fake_max': fake_feat.max(),
    })

print()
print("="*70)
print(" Range Analysis (Should be within [-1, 1])")
print("="*70)
print()

print(f"{'Feature':<10} {'Real Min':<12} {'Fake Min':<12} {'Real Max':<12} {'Fake Max':<12}")
print("-"*70)
for stats in feature_stats:
    print(f"{stats['name']:<10} {stats['real_min']:>11.4f} {stats['fake_min']:>11.4f} {stats['real_max']:>11.4f} {stats['fake_max']:>11.4f}")

print()
print("="*70)
print(" Problematic Features (sorted by mean difference)")
print("="*70)
print()

sorted_by_mean = sorted(feature_stats, key=lambda x: x['mean_diff'], reverse=True)
for i, stats in enumerate(sorted_by_mean[:5], 1):
    print(f"{i}. {stats['name']}: mean diff = {stats['mean_diff']:.4f}")

print()
print("="*70)
print(" Problematic Features (sorted by std difference)")
print("="*70)
print()

sorted_by_std = sorted(feature_stats, key=lambda x: x['std_diff'], reverse=True)
for i, stats in enumerate(sorted_by_std[:5], 1):
    print(f"{i}. {stats['name']}: std diff = {stats['std_diff']:.4f}")

# Create visualization
print()
print("Creating feature distribution plots...")

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

for i, feat_name in enumerate(FEATURE_NAMES):
    ax = axes[i]
    real_feat = X_test[:, :, i].flatten()
    fake_feat = X_gen[:, :, i].flatten()

    # Create histograms
    bins = np.linspace(-1.5, 1.5, 50)
    ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', color='blue', density=True)
    ax.hist(fake_feat, bins=bins, alpha=0.5, label='Generated', color='red', density=True)

    ax.set_xlabel(feat_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{feat_name} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=-1, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: feature_distributions.png")

# Check for out-of-range values
print()
print("="*70)
print(" Out-of-Range Value Check")
print("="*70)
print()

for i, feat_name in enumerate(FEATURE_NAMES):
    fake_feat = X_gen[:, :, i].flatten()
    out_of_range = np.sum((fake_feat < -1.5) | (fake_feat > 1.5))
    pct = 100 * out_of_range / len(fake_feat)
    if out_of_range > 0:
        print(f"⚠️  {feat_name}: {out_of_range} values ({pct:.2f}%) outside [-1.5, 1.5]")
    else:
        print(f"✓  {feat_name}: All values within valid range")

print()
print("="*70)
print(" Diagnosis Complete!")
print("="*70)
print()
print("Next steps:")
print("1. Review feature_distributions.png to see which features look different")
print("2. Check if any features are outside valid range")
print("3. If means/stds are very different, consider continuing Stage 2 training")
print("4. If distributions overlap well but discriminative is still high,")
print("   there may be subtle temporal patterns the discriminator detects")
