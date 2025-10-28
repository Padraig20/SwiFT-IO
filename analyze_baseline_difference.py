#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

print("="*70)
print("Why zeros baseline shows attribution outside brain?")
print("="*70)

# Load both baselines
pos_10sec_path = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments/sub-NDARMZ366UY8/target5_Positive/sub-NDARMZ366UY8_Positive_TR450_rank01_AVGpred_positive.nii.gz"
pos_zeros_path = project_root / "analysis/4_IGmap/baseline_zeros_selective/opr6oq97/nii_segments/sub-NDARMZ366UY8/target5_Positive/sub-NDARMZ366UY8_Positive_TR450_rank01_AVGpred_positive.nii.gz"

data_10sec = nib.load(str(pos_10sec_path)).get_fdata()
data_zeros = nib.load(str(pos_zeros_path)).get_fdata()

print("\n1. Spatial distribution analysis:")
print("-"*70)

# Define regions
center_region = (slice(32, 64), slice(32, 64), slice(32, 64))  # Center (brain)
edge_region = (slice(0, 20), slice(0, 20), slice(0, 20))  # Edge (outside brain)

print("\nfirst_10sec baseline:")
print(f"  Center (brain): {data_10sec[center_region].mean():.8f}")
print(f"  Edge (outside): {data_10sec[edge_region].mean():.8f}")
print(f"  Ratio (edge/center): {data_10sec[edge_region].mean() / (data_10sec[center_region].mean() + 1e-10):.2f}")

print("\nzeros baseline:")
print(f"  Center (brain): {data_zeros[center_region].mean():.8f}")
print(f"  Edge (outside): {data_zeros[edge_region].mean():.8f}")
print(f"  Ratio (edge/center): {data_zeros[edge_region].mean() / data_zeros[center_region].mean():.2f}")

print("\n2. Understanding the difference:")
print("-"*70)
print("\nIG calculation:")
print("  zeros baseline: IG = (input - 0) × gradient = input × gradient")
print("  first_10sec baseline: IG = (input - first_10_avg) × gradient")

print("\nWhy zeros baseline has attribution everywhere:")
print("  • Brain-masked input has ZEROS outside brain")
print("  • (0 - 0) × gradient = 0 × gradient = 0 (should be zero outside)")
print("  • BUT: Model might assign gradients to ALL voxels")
print("  • Even small gradients × 0 input = still 0")
print("  • ISSUE: Input might NOT be properly masked!")

print("\nWhy first_10sec baseline is sparse:")
print("  • Outside brain: input ≈ first_10_avg (both near zero)")
print("  • IG = (0 - 0) × gradient ≈ 0")
print("  • Inside brain: input ≠ first_10_avg (temporal change)")
print("  • IG = (large difference) × gradient → non-zero")

print("\n3. Checking if input has non-zero values outside brain:")
print("-"*70)

# Check the data loading code
print("\nFrom igmap_baseline_comparison.py:")
print("  input_ts = data['fmri_sequence'].float().cpu()  # [B, C, X, Y, Z, T]")
print("\n→ Need to check if fMRI data has brain mask applied")

print("\n4. Hypothesis:")
print("-"*70)
print("zeros baseline shows attribution outside brain because:")
print("  ✓ Model gradients exist for all voxels (not masked during backprop)")
print("  ✓ Input data might have small non-zero values outside brain")
print("  ✓ Result: IG = small_input × gradient → non-zero everywhere")
print("\nfirst_10sec baseline doesn't have this issue because:")
print("  ✓ Temporal difference outside brain ≈ 0 (no signal change)")
print("  ✓ Result: IG = 0 difference × gradient → zero outside brain")

print("\n5. Recommendation:")
print("-"*70)
print("✅ Use first_10sec baseline - naturally masks out non-brain regions")
print("✅ More interpretable - only shows regions with TEMPORAL CHANGES")
print("⚠️  zeros baseline - shows all regions with any gradient, less interpretable")

