"""Check normalization parameters"""
import numpy as np

params = np.load('processed_data_v6/normalization_params.npy', allow_pickle=True).item()

print("=" * 70)
print("NORMALIZATION PARAMETERS")
print("=" * 70)

for key, value in params.items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {value}")
