"""
Diagnostic Visualization for Generated Trajectories

Investigates why trajectories look like straight lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load generated data
results_dir = Path('results/diffusion_v7')
trajectories = np.load(results_dir / 'generated_trajectories.npy')
conditions = np.load(results_dir / 'generation_conditions.npy')
lengths = np.load(results_dir / 'generation_lengths.npy')

print("=" * 70)
print("TRAJECTORY DIAGNOSTICS")
print("=" * 70)
print(f"\nLoaded {len(trajectories)} trajectories")
print(f"Shape: {trajectories.shape}")

# Analyze conditions
distances = conditions[:, 0]
angles = np.arctan2(conditions[:, 2], conditions[:, 1])

print(f"\nCondition Statistics:")
print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
print(f"  Angle range: [{np.degrees(angles.min()):.1f}°, {np.degrees(angles.max()):.1f}°]")
print(f"  Unique angles: {len(np.unique(np.round(angles, 2)))}")

# Plot 1: Condition distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(distances, bins=30, edgecolor='black')
ax1.set_xlabel('Target Distance (normalized)')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Target Distances')
ax1.grid(True, alpha=0.3)

ax2.hist(np.degrees(angles), bins=30, edgecolor='black')
ax2.set_xlabel('Target Angle (degrees)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Target Angles')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('condition_distribution.png', dpi=150)
print("\n✓ Saved condition_distribution.png")
plt.close()

# Plot 2: Trajectories colored by target angle
fig, ax = plt.subplots(figsize=(12, 10))

# Select subset for visualization
num_plot = min(50, len(trajectories))
colors = plt.cm.hsv(np.linspace(0, 1, num_plot))

for i in range(num_plot):
    traj = trajectories[i]
    length = int(lengths[i])

    # Compute positions
    dx = traj[:length, 0]
    dy = traj[:length, 1]
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])

    # Get target angle for coloring
    target_angle = np.degrees(angles[i])

    ax.plot(x, y, alpha=0.6, linewidth=1.5, color=colors[i],
            label=f'{target_angle:.0f}°' if i < 10 else '')

ax.scatter([0], [0], color='green', s=200, marker='o',
           label='Start', zorder=5, edgecolors='black', linewidths=2)

ax.set_xlabel('X Position', fontsize=12)
ax.set_ylabel('Y Position', fontsize=12)
ax.set_title(f'Generated Trajectories (n={num_plot}) - Colored by Target Angle', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('trajectories_by_angle.png', dpi=150, bbox_inches='tight')
print("✓ Saved trajectories_by_angle.png")
plt.close()

# Plot 3: Individual examples with different target angles
angles_deg = np.degrees(angles)
# Find trajectories with different target angles
angle_bins = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
selected_indices = []

for i in range(len(angle_bins) - 1):
    low, high = angle_bins[i], angle_bins[i + 1]
    mask = (angles_deg >= low) & (angles_deg < high)
    if mask.any():
        idx = np.where(mask)[0][0]
        selected_indices.append(idx)

if len(selected_indices) >= 6:
    selected_indices = selected_indices[:6]
else:
    # Fill with random if not enough
    while len(selected_indices) < 6:
        selected_indices.append(np.random.randint(0, len(trajectories)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for plot_idx, traj_idx in enumerate(selected_indices):
    ax = axes[plot_idx]
    traj = trajectories[traj_idx]
    length = int(lengths[traj_idx])

    # Compute positions
    dx = traj[:length, 0]
    dy = traj[:length, 1]
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])

    # Plot
    ax.plot(x, y, 'b-', linewidth=2)
    ax.scatter([0], [0], color='green', s=100, marker='o', zorder=5)
    ax.scatter([x[-1]], [y[-1]], color='red', s=100, marker='X', zorder=5)

    # Draw target direction
    target_dist = distances[traj_idx]
    target_angle = angles[traj_idx]
    target_x = np.cos(target_angle) * 2  # Arbitrary scale for visualization
    target_y = np.sin(target_angle) * 2
    ax.arrow(0, 0, target_x, target_y, head_width=0.2, head_length=0.1,
             fc='orange', ec='orange', alpha=0.5, linewidth=2, linestyle='--',
             label='Target Direction')

    ax.set_aspect('equal')
    ax.set_title(f'Target: {np.degrees(angles[traj_idx]):.1f}° | Dist: {target_dist:.2f}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('individual_with_targets.png', dpi=150)
print("✓ Saved individual_with_targets.png")
plt.close()

# Plot 4: Endpoint scatter (achieved vs target)
fig, ax = plt.subplots(figsize=(10, 10))

for i in range(len(trajectories)):
    traj = trajectories[i]
    length = int(lengths[i])

    # Achieved endpoint
    dx = traj[:length, 0]
    dy = traj[:length, 1]
    final_x = np.sum(dx)
    final_y = np.sum(dy)

    # Target endpoint (approximate from normalized distance)
    target_angle = angles[i]
    target_dist = distances[i]  # This is normalized
    # For visualization, assume similar scale
    target_x = np.cos(target_angle) * abs(target_dist) * 2
    target_y = np.sin(target_angle) * abs(target_dist) * 2

    ax.scatter(final_x, final_y, c='blue', s=30, alpha=0.6, label='Achieved' if i == 0 else '')
    ax.scatter(target_x, target_y, c='red', s=30, alpha=0.3, marker='x', label='Target' if i == 0 else '')

ax.scatter([0], [0], color='green', s=200, marker='o', label='Start', zorder=5)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Endpoint Distribution: Achieved (blue) vs Target (red x)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('endpoint_scatter.png', dpi=150)
print("✓ Saved endpoint_scatter.png")
plt.close()

# Analyze trajectory shapes
print("\n" + "=" * 70)
print("TRAJECTORY SHAPE ANALYSIS")
print("=" * 70)

straightness_scores = []
for i in range(len(trajectories)):
    traj = trajectories[i]
    length = int(lengths[i])

    # Compute cumulative positions
    dx = traj[:length, 0]
    dy = traj[:length, 1]
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    # Measure straightness: ratio of direct distance to path length
    if length > 1:
        direct_dist = np.sqrt(x[-1]**2 + y[-1]**2)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        if path_length > 0:
            straightness = direct_dist / path_length
            straightness_scores.append(straightness)

straightness_scores = np.array(straightness_scores)
print(f"\nStraightness (1.0 = perfectly straight):")
print(f"  Mean: {straightness_scores.mean():.3f}")
print(f"  Median: {np.median(straightness_scores):.3f}")
print(f"  Min: {straightness_scores.min():.3f}")
print(f"  Max: {straightness_scores.max():.3f}")

if straightness_scores.mean() > 0.9:
    print("\n⚠️  WARNING: Trajectories are very straight!")
    print("   This suggests the model may not be generating realistic curved paths.")

print("\n" + "=" * 70)
print("Check the generated images:")
print("  - condition_distribution.png: Are target angles evenly distributed?")
print("  - trajectories_by_angle.png: Do trajectories follow their target angles?")
print("  - individual_with_targets.png: Do paths align with orange target arrows?")
print("  - endpoint_scatter.png: Do blue dots cluster around red x's?")
print("=" * 70)
