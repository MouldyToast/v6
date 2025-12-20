"""
Compare Real vs Generated Trajectory Straightness
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_dir = Path('processed_data_v6')
X_test = np.load(data_dir / 'X_test.npy')
L_test = np.load(data_dir / 'L_test.npy')

print("=" * 70)
print("REAL vs GENERATED STRAIGHTNESS COMPARISON")
print("=" * 70)

def compute_straightness(trajectories, lengths):
    """Compute straightness for trajectories."""
    straightness = []
    for i in range(len(trajectories)):
        traj = trajectories[i]
        length = int(lengths[i])

        dx = traj[:length, 0]
        dy = traj[:length, 1]

        # Direct distance
        direct_dist = np.sqrt(np.sum(dx)**2 + np.sum(dy)**2)

        # Path length
        path_length = np.sum(np.sqrt(dx**2 + dy**2))

        if path_length > 0:
            straightness.append(direct_dist / path_length)

    return np.array(straightness)

# Analyze real trajectories
print("\n[REAL TRAJECTORIES]")
real_straightness = compute_straightness(X_test, L_test)
print(f"  Mean: {real_straightness.mean():.3f}")
print(f"  Median: {np.median(real_straightness):.3f}")
print(f"  Std: {real_straightness.std():.3f}")
print(f"  Min: {real_straightness.min():.3f}")
print(f"  Max: {real_straightness.max():.3f}")

# Load generated if available
gen_path = Path('results/diffusion_v7/generated_trajectories.npy')
if gen_path.exists():
    print("\n[GENERATED TRAJECTORIES]")
    gen_trajectories = np.load('results/diffusion_v7/generated_trajectories.npy')
    gen_lengths = np.load('results/diffusion_v7/generation_lengths.npy')

    gen_straightness = compute_straightness(gen_trajectories, gen_lengths)
    print(f"  Mean: {gen_straightness.mean():.3f}")
    print(f"  Median: {np.median(gen_straightness):.3f}")
    print(f"  Std: {gen_straightness.std():.3f}")
    print(f"  Min: {gen_straightness.min():.3f}")
    print(f"  Max: {gen_straightness.max():.3f}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(real_straightness, bins=50, alpha=0.7, color='green', edgecolor='black', label='Real')
    ax1.hist(gen_straightness, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Generated')
    ax1.set_xlabel('Straightness (1.0 = perfectly straight)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Straightness Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(real_straightness.mean(), color='green', linestyle='--', linewidth=2, label=f'Real Mean: {real_straightness.mean():.3f}')
    ax1.axvline(gen_straightness.mean(), color='blue', linestyle='--', linewidth=2, label=f'Gen Mean: {gen_straightness.mean():.3f}')

    # Box plot
    ax2.boxplot([real_straightness, gen_straightness],
                labels=['Real', 'Generated'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Straightness', fontsize=12)
    ax2.set_title('Straightness Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('straightness_comparison.png', dpi=150)
    print("\n✓ Saved straightness_comparison.png")

    # Analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if real_straightness.mean() > 0.9:
        print("\n⚠️  PROBLEM: Real trajectories are also very straight!")
        print("   Issue is in the training data, not the model.")
        print("   Possible causes:")
        print("   - Data preprocessing removed curvature")
        print("   - Recording conditions favored straight movements")
        print("   - Need to collect more natural curved trajectories")
    elif gen_straightness.mean() > real_straightness.mean() + 0.1:
        print("\n⚠️  PROBLEM: Model generates straighter paths than real data!")
        print(f"   Real mean: {real_straightness.mean():.3f}")
        print(f"   Generated mean: {gen_straightness.mean():.3f}")
        print(f"   Difference: {gen_straightness.mean() - real_straightness.mean():.3f}")
        print("\n   Possible solutions:")
        print("   1. Add curvature regularization to training")
        print("   2. Increase model capacity (more layers)")
        print("   3. Train longer (model might need more epochs)")
        print("   4. Check if diffusion is too aggressive (beta schedule)")
    else:
        print("\n✓ Model straightness matches real data!")
        print("   The model is correctly learning from the training distribution.")
else:
    print("\n⚠️  No generated trajectories found at results/diffusion_v7/")
    print("   Run generation first.")

# Plot examples
print("\n" + "=" * 70)
print("VISUAL COMPARISON")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Real examples
for i in range(3):
    ax = axes[0, i]
    idx = np.random.randint(0, len(X_test))
    traj = X_test[idx]
    length = int(L_test[idx])

    dx = traj[:length, 0]
    dy = traj[:length, 1]
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])

    ax.plot(x, y, 'g-', linewidth=2, label='Real')
    ax.scatter([0], [0], color='green', s=100, marker='o', zorder=5)
    ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='X', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f'Real Example {i+1}\nStraightness: {real_straightness[idx]:.3f}')
    ax.grid(True, alpha=0.3)

# Generated examples
if gen_path.exists():
    for i in range(3):
        ax = axes[1, i]
        idx = np.random.randint(0, len(gen_trajectories))
        traj = gen_trajectories[idx]
        length = int(gen_lengths[idx])

        dx = traj[:length, 0]
        dy = traj[:length, 1]
        x = np.concatenate([[0], np.cumsum(dx)])
        y = np.concatenate([[0], np.cumsum(dy)])

        ax.plot(x, y, 'b-', linewidth=2, label='Generated')
        ax.scatter([0], [0], color='green', s=100, marker='o', zorder=5)
        ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='X', zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f'Generated Example {i+1}\nStraightness: {gen_straightness[idx]:.3f}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('real_vs_generated_examples.png', dpi=150)
print("✓ Saved real_vs_generated_examples.png")

print("\n" + "=" * 70)
