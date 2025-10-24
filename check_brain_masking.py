#!/usr/bin/env python3
"""
Check if brain masking is applied to input fMRI data
"""

import torch
import numpy as np
import glob
import os

# Load a sample data file
data_dir = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img"
sample_files = glob.glob(f"{data_dir}/*/frame_*.pt")[:5]

print("="*80)
print("BRAIN MASKING CHECK - INPUT DATA")
print("="*80)

for i, sample_file in enumerate(sample_files):
    print(f"\n[Sample {i+1}] {os.path.basename(os.path.dirname(sample_file))}/{os.path.basename(sample_file)}")

    # Load data
    data = torch.load(sample_file)

    # Check data shape and type
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")

    # Convert to numpy for analysis
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)

    # Flatten data
    data_flat = data_np.flatten()

    # Check statistics
    total_voxels = len(data_flat)
    zero_voxels = np.sum(data_flat == 0)
    non_zero_voxels = total_voxels - zero_voxels
    zero_ratio = zero_voxels / total_voxels * 100

    print(f"  Total voxels: {total_voxels:,}")
    print(f"  Zero voxels: {zero_voxels:,} ({zero_ratio:.2f}%)")
    print(f"  Non-zero voxels: {non_zero_voxels:,} ({100-zero_ratio:.2f}%)")
    print(f"  Value range: [{data_np.min():.4f}, {data_np.max():.4f}]")
    print(f"  Mean (non-zero): {data_flat[data_flat != 0].mean():.4f}")
    print(f"  Std (non-zero): {data_flat[data_flat != 0].std():.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

# Typical brain mask has ~60-70% zeros (outside brain)
typical_zero_ratio = zero_ratio
if typical_zero_ratio > 50:
    print(f"✓ Likely BRAIN MASKED: {typical_zero_ratio:.1f}% zeros suggests brain masking is applied")
    print("  (Typical brain occupies 30-50% of 96x96x96 volume)")
else:
    print(f"✗ Likely NOT MASKED: Only {typical_zero_ratio:.1f}% zeros suggests full volume")
    print("  (Would expect 50-70% zeros if brain masked)")

print("="*80)
