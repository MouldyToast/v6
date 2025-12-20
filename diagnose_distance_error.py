"""
Diagnose Distance Prediction Error

Analyzes why generated trajectories don't reach correct endpoint distances.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load real data
data_dir = Path('processed_data_v6')
X_test = np.load(data_dir / 'X_test.npy')
C_test = np.load(data_dir / 'C_test.npy')
L_test = np.load(data_dir / 'L_test.npy')

# Load generated data
gen_dir = Path('results/diffusion_v7')
gen_trajectories = np.load(gen_dir / 'generated_trajectories.npy')
gen_conditions = np.load(gen_dir / 'generation_conditions.npy')
gen_lengths = np.load(gen_dir / 'generation_lengths.npy')

print("=" * 70)
print("DISTANCE PREDICTION DIAGNOSTIC")
print("=" * 70)

def compute_endpoint_distance(trajectories, lengths):
    """Compute actual endpoint distance for each trajectory."""
    distances = []
    for i in range(len(trajectories)):
        traj = trajectories[i]
        length = int(lengths[i])

        dx = traj[:length, 0]
        dy = traj[:length, 1]

        # Total displacement
        total_dx = np.sum(dx)
        total_dy = np.sum(dy)

        distance = np.sqrt(total_dx**2 + total_dy**2)
        distances.append(distance)

    return np.array(distances)

# Analyze real data
print("\n[REAL DATA ANALYSIS]")
real_distances = compute_endpoint_distance(X_test, L_test)
real_target_distances = C_test[:, 0]  # Normalized distance from condition

print(f"Real trajectory endpoint distances:")
print(f"  Mean: {real_distances.mean():.3f}")
print(f"  Median: {np.median(real_distances):.3f}")
print(f"  Std: {real_distances.std():.3f}")
print(f"  Min: {real_distances.min():.3f}")
print(f"  Max: {real_distances.max():.3f}")

print(f"\nReal target distances (normalized):")
print(f"  Mean: {real_target_distances.mean():.3f}")
print(f"  Median: {np.median(real_target_distances):.3f}")
print(f"  Std: {real_target_distances.std():.3f}")
print(f"  Min: {real_target_distances.min():.3f}")
print(f"  Max: {real_target_distances.max():.3f}")

# Compute correlation for real data
real_errors = np.abs(real_distances - real_target_distances)
print(f"\nReal data - target vs actual correlation: {np.corrcoef(real_target_distances, real_distances)[0,1]:.3f}")
print(f"Real data - mean absolute error: {real_errors.mean():.3f}")

# Analyze generated data
print("\n[GENERATED DATA ANALYSIS]")
gen_distances = compute_endpoint_distance(gen_trajectories, gen_lengths)
gen_target_distances = gen_conditions[:, 0]  # Normalized distance from condition

print(f"Generated trajectory endpoint distances:")
print(f"  Mean: {gen_distances.mean():.3f}")
print(f"  Median: {np.median(gen_distances):.3f}")
print(f"  Std: {gen_distances.std():.3f}")
print(f"  Min: {gen_distances.min():.3f}")
print(f"  Max: {gen_distances.max():.3f}")

print(f"\nGenerated target distances (normalized):")
print(f"  Mean: {gen_target_distances.mean():.3f}")
print(f"  Median: {np.median(gen_target_distances):.3f}")
print(f"  Std: {gen_target_distances.std():.3f}")
print(f"  Min: {gen_target_distances.min():.3f}")
print(f"  Max: {gen_target_distances.max():.3f}")

# Compute correlation for generated data
gen_errors = np.abs(gen_distances - gen_target_distances)
print(f"\nGenerated - target vs actual correlation: {np.corrcoef(gen_target_distances, gen_distances)[0,1]:.3f}")
print(f"Generated - mean absolute error: {gen_errors.mean():.3f}")

# Check if there's a scale mismatch
print("\n" + "=" * 70)
print("SCALE ANALYSIS")
print("=" * 70)

# The target distances are in [-1, 1] (normalized)
# But actual trajectory distances might be in pixel space
# Let's check if there's a consistent scale factor

# Find scale factor that minimizes error
from scipy.optimize import minimize_scalar

def error_with_scale(scale):
    scaled_targets = gen_target_distances * scale
    return np.mean(np.abs(gen_distances - scaled_targets))

result = minimize_scalar(error_with_scale, bounds=(0.1, 100), method='bounded')
best_scale = result.x
best_error = result.fun

print(f"\nBest scale factor for target distances: {best_scale:.3f}")
print(f"Error with this scale: {best_error:.3f}")
print(f"Current error (scale=1): {gen_errors.mean():.3f}")
print(f"Improvement: {gen_errors.mean() - best_error:.3f}")

if best_scale > 2.0 or best_scale < 0.5:
    print(f"\n⚠️  SCALE MISMATCH DETECTED!")
    print(f"   Target distances need to be scaled by {best_scale:.2f}x")
    print(f"   This suggests normalization parameters might be wrong.")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Real - Target vs Actual Distance
ax = axes[0, 0]
ax.scatter(real_target_distances, real_distances, alpha=0.5, s=20)
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Target Distance (normalized)')
ax.set_ylabel('Actual Distance')
ax.set_title(f'Real Trajectories\nCorrelation: {np.corrcoef(real_target_distances, real_distances)[0,1]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Generated - Target vs Actual Distance
ax = axes[0, 1]
ax.scatter(gen_target_distances, gen_distances, alpha=0.5, s=20, c='blue')
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Target Distance (normalized)')
ax.set_ylabel('Actual Distance')
ax.set_title(f'Generated Trajectories\nCorrelation: {np.corrcoef(gen_target_distances, gen_distances)[0,1]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax = axes[1, 0]
ax.hist(real_errors, bins=50, alpha=0.7, color='green', edgecolor='black', label='Real')
ax.hist(gen_errors, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Generated')
ax.set_xlabel('Absolute Distance Error')
ax.set_ylabel('Count')
ax.set_title('Distance Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: With best scale applied
ax = axes[1, 1]
scaled_targets = gen_target_distances * best_scale
ax.scatter(scaled_targets, gen_distances, alpha=0.5, s=20, c='purple')
min_val = min(scaled_targets.min(), gen_distances.min())
max_val = max(scaled_targets.max(), gen_distances.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel(f'Target Distance × {best_scale:.2f}')
ax.set_ylabel('Actual Distance')
ax.set_title(f'Generated with Optimal Scaling\nError: {best_error:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distance_diagnostic.png', dpi=150)
print("\n✓ Saved distance_diagnostic.png")

# Diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if np.corrcoef(gen_target_distances, gen_distances)[0,1] < 0.3:
    print("\n❌ CRITICAL: Model is NOT learning distance conditioning!")
    print("   Correlation < 0.3 means model ignores target distance.")
    print("\n   Possible causes:")
    print("   1. CFG dropout too high (model learned to ignore conditions)")
    print("   2. Distance feature is not properly normalized")
    print("   3. Model capacity too small")
    print("   4. Need to train much longer")
elif abs(best_scale - 1.0) > 0.5:
    print("\n⚠️  NORMALIZATION ISSUE!")
    print(f"   Target distances need {best_scale:.2f}x scaling.")
    print("   The normalization parameters in preprocessing might be wrong.")
    print("\n   Check: processed_data_v6/normalization_params.json")
elif gen_errors.mean() > 0.3:
    print("\n⚠️  MODEL NEEDS MORE TRAINING!")
    print(f"   Current error: {gen_errors.mean():.3f}")
    print(f"   Real data error: {real_errors.mean():.3f}")
    print("\n   Solutions:")
    print("   1. Train for more epochs")
    print("   2. Increase CFG scale during generation (try 5.0 or 10.0)")
    print("   3. Reduce CFG dropout during training")
else:
    print("\n✓ Distance prediction is reasonable!")
    print(f"   Error: {gen_errors.mean():.3f}")

print("\n" + "=" * 70)
