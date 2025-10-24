#!/usr/bin/env python3
"""
Re-check brain masking with understanding that background = min(brain z-scores)
"""

import torch
import numpy as np
import glob
import os

# Load sample data files
data_dir = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img"
sample_files = glob.glob(f"{data_dir}/*/frame_*.pt")[:10]

print("="*80)
print("BRAIN MASKING RE-CHECK - Understanding background = min(brain)")
print("="*80)

all_background_ratios = []

for i, sample_file in enumerate(sample_files[:5]):
    print(f"\n[Sample {i+1}] {os.path.basename(os.path.dirname(sample_file))}/{os.path.basename(sample_file)}")

    # Load data
    data = torch.load(sample_file)

    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)

    # Flatten data
    data_flat = data_np.flatten()

    # Find the minimum value (this should be the background)
    min_val = data_flat.min()

    # Count how many voxels have exactly this minimum value
    background_voxels = np.sum(data_flat == min_val)
    total_voxels = len(data_flat)
    background_ratio = background_voxels / total_voxels * 100

    all_background_ratios.append(background_ratio)

    print(f"  Shape: {data.shape}")
    print(f"  Total voxels: {total_voxels:,}")
    print(f"  Min value (background): {min_val:.4f}")
    print(f"  Background voxels: {background_voxels:,} ({background_ratio:.2f}%)")
    print(f"  Brain voxels: {total_voxels - background_voxels:,} ({100-background_ratio:.2f}%)")
    print(f"  Value range: [{data_np.min():.4f}, {data_np.max():.4f}]")

    # Brain statistics (excluding background)
    brain_voxels = data_flat[data_flat != min_val]
    if len(brain_voxels) > 0:
        print(f"  Brain mean: {brain_voxels.mean():.4f}")
        print(f"  Brain std: {brain_voxels.std():.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

avg_background_ratio = np.mean(all_background_ratios)
print(f"\nAverage background ratio: {avg_background_ratio:.1f}%")

if 50 <= avg_background_ratio <= 75:
    print(f"✓ BRAIN MASKING CONFIRMED!")
    print(f"  - Background occupies {avg_background_ratio:.1f}% (typical: 50-70%)")
    print(f"  - Brain occupies {100-avg_background_ratio:.1f}%")
    print(f"  - Background filled with minimum z-score value")
elif avg_background_ratio < 50:
    print(f"⚠ UNEXPECTED: Background ratio too low ({avg_background_ratio:.1f}%)")
    print(f"  - Expected 50-70% for brain-masked data")
else:
    print(f"⚠ UNEXPECTED: Background ratio too high ({avg_background_ratio:.1f}%)")

print("\n" + "="*80)
print("CHECKING datasets.py BACKGROUND VALUE USAGE")
print("="*80)

# Check what datasets.py does
print("\nFrom datasets.py line 145:")
print("  background_value = y.flatten()[0]")
print(f"\nThis extracts: {data_flat[0]:.4f}")
print(f"Actual background value should be: {min_val:.4f}")

if abs(data_flat[0] - min_val) < 0.01:
    print("✓ First voxel IS the background value - padding will use background!")
else:
    print("⚠ First voxel is NOT the background value - padding might use wrong value!")
    print(f"  Difference: {abs(data_flat[0] - min_val):.4f}")

print("="*80)
