#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path

# Check one fMRI input file
data_path = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
subject = "sub-NDARMZ366UY8"

# Find subject files
import os
subject_files = []
for root, dirs, files in os.walk(data_path):
    for f in files:
        if subject in f and f.endswith('.nii.gz'):
            subject_files.append(os.path.join(root, f))
            if len(subject_files) >= 1:
                break
    if subject_files:
        break

if subject_files:
    print(f"Found: {subject_files[0]}")
    img = nib.load(subject_files[0])
    data = img.get_fdata()
    
    print(f"\nInput fMRI data shape: {data.shape}")
    print(f"First TR statistics:")
    first_tr = data[:, :, :, 0] if data.ndim == 4 else data
    print(f"  Min: {first_tr.min():.6f}")
    print(f"  Max: {first_tr.max():.6f}")
    print(f"  Mean: {first_tr.mean():.6f}")
    print(f"  Non-zero voxels: {np.count_nonzero(first_tr)}/{first_tr.size}")
    print(f"  Zero voxels: {np.sum(first_tr == 0)}")
    
    # Check if there's a clear brain mask (values near zero outside brain)
    threshold = 0.01
    brain_voxels = np.abs(first_tr) > threshold
    print(f"\n  Voxels > {threshold}: {np.sum(brain_voxels)}/{first_tr.size} ({100*np.sum(brain_voxels)/first_tr.size:.1f}%)")
else:
    print("No subject files found")

# Check IG map files
print("\n" + "="*70)
print("IG Map Analysis:")
print("="*70)

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# first_10sec baseline
pos_10sec = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments/sub-NDARMZ366UY8/target5_Positive/sub-NDARMZ366UY8_Positive_TR450_rank01_AVGpred_positive.nii.gz"
if pos_10sec.exists():
    img = nib.load(str(pos_10sec))
    data = img.get_fdata()
    print(f"\nfirst_10sec baseline (Positive):")
    print(f"  Shape: {data.shape}")
    print(f"  Non-zero voxels: {np.count_nonzero(data)}/{data.size} ({100*np.count_nonzero(data)/data.size:.1f}%)")
    print(f"  Min: {data.min():.8f}, Max: {data.max():.8f}")
    print(f"  Mean (all): {data.mean():.8f}")
    print(f"  Mean (non-zero): {data[data > 0].mean():.8f}")

# zeros baseline
pos_zeros = project_root / "analysis/4_IGmap/baseline_zeros_selective/opr6oq97/nii_segments/sub-NDARMZ366UY8/target5_Positive/sub-NDARMZ366UY8_Positive_TR450_rank01_AVGpred_positive.nii.gz"
if pos_zeros.exists():
    img = nib.load(str(pos_zeros))
    data = img.get_fdata()
    print(f"\nzeros baseline (Positive):")
    print(f"  Shape: {data.shape}")
    print(f"  Non-zero voxels: {np.count_nonzero(data)}/{data.size} ({100*np.count_nonzero(data)/data.size:.1f}%)")
    print(f"  Min: {data.min():.8f}, Max: {data.max():.8f}")
    print(f"  Mean (all): {data.mean():.8f}")
    print(f"  Mean (non-zero): {data[data > 0].mean():.8f}")
    
    # Check distribution in different regions
    # Assume center is brain, edges are outside brain
    center_slice = data[32:64, 32:64, 32:64]  # Center region
    edge_slice = data[0:16, 0:16, 0:16]  # Edge region (likely outside brain)
    
    print(f"\n  Center region (likely brain):")
    print(f"    Mean: {center_slice.mean():.8f}")
    print(f"    Non-zero: {np.count_nonzero(center_slice)}/{center_slice.size}")
    
    print(f"  Edge region (likely outside brain):")
    print(f"    Mean: {edge_slice.mean():.8f}")
    print(f"    Non-zero: {np.count_nonzero(edge_slice)}/{edge_slice.size}")

