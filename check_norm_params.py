"""
Check what normalization parameters were saved during preprocessing
"""
import numpy as np
import json
from pathlib import Path

# Load normalization parameters
norm_path = Path('processed_data_v6/normalization_params.npy')
if norm_path.exists():
    params = np.load(norm_path, allow_pickle=True).item()

    print("=" * 70)
    print("NORMALIZATION PARAMETERS FROM PREPROCESSING")
    print("=" * 70)

    for key in sorted(params.keys()):
        value = params[key]
        print(f"\n{key}: {value}")

    # Check if goal distance parameters exist
    if 'goal_dist_min' in params and 'goal_dist_max' in params:
        print("\n" + "=" * 70)
        print("GOAL DISTANCE NORMALIZATION")
        print("=" * 70)
        print(f"Distance range in training data: {params['goal_dist_min']:.2f} to {params['goal_dist_max']:.2f} pixels")
        print(f"\nTo denormalize:")
        print(f"  pixel_distance = (norm_distance + 1) / 2 * ({params['goal_dist_max']:.2f} - {params['goal_dist_min']:.2f}) + {params['goal_dist_min']:.2f}")

        # Test denormalization
        test_norms = [-1.0, -0.5, 0.0, 0.5, 1.0]
        print(f"\nTest denormalization:")
        for norm_val in test_norms:
            pixel_val = (norm_val + 1) / 2 * (params['goal_dist_max'] - params['goal_dist_min']) + params['goal_dist_min']
            print(f"  Normalized {norm_val:5.1f} → {pixel_val:7.2f} pixels")

        # Save as JSON for easy access
        json_path = Path('processed_data_v6/normalization_params.json')
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\n✓ Saved to {json_path} for easier access")
    else:
        print("\n⚠️  WARNING: goal_dist_min and goal_dist_max not found!")
        print("   Preprocessing might not have run correctly.")
else:
    print(f"❌ File not found: {norm_path}")
