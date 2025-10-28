import nibabel as nib
import numpy as np
from pathlib import Path

# Load one example
project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
base_dir = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments"

# Anger, first subject, first sequence
pos_path = base_dir / "sub-NDARUC771VM5/target0_Anger/sub-NDARUC771VM5_Anger_TR060_rank05_AVGpred_positive.nii.gz"
neg_path = base_dir / "sub-NDARUC771VM5/target0_Anger/sub-NDARUC771VM5_Anger_TR060_rank05_AVGpred_negative.nii.gz"

pos_data = nib.load(str(pos_path)).get_fdata()
neg_data = nib.load(str(neg_path)).get_fdata()

print("Positive IG Map:")
print(f"  Shape: {pos_data.shape}")
print(f"  Min: {pos_data.min():.8f}, Max: {pos_data.max():.8f}")
print(f"  Mean: {pos_data.mean():.8f}")
print(f"  Non-zero voxels: {np.sum(pos_data != 0)}")
print(f"  Positive voxels: {np.sum(pos_data > 0)}")
print(f"  Negative voxels: {np.sum(pos_data < 0)}")

print("\nNegative IG Map:")
print(f"  Shape: {neg_data.shape}")
print(f"  Min: {neg_data.min():.8f}, Max: {neg_data.max():.8f}")
print(f"  Mean: {neg_data.mean():.8f}")
print(f"  Non-zero voxels: {np.sum(neg_data != 0)}")
print(f"  Positive voxels: {np.sum(neg_data > 0)}")
print(f"  Negative voxels: {np.sum(neg_data < 0)}")

# Check spatial overlap
pos_active = pos_data > 0
neg_active = neg_data < 0

overlap = np.sum(pos_active & neg_active)
only_pos = np.sum(pos_active & ~neg_active)
only_neg = np.sum(~pos_active & neg_active)

print("\nSpatial Pattern:")
print(f"  Voxels active in both: {overlap}")
print(f"  Only positive: {only_pos}")
print(f"  Only negative: {only_neg}")

# Check if they're just inverted versions
correlation = np.corrcoef(pos_data.flatten(), neg_data.flatten())[0, 1]
print(f"\nCorrelation between pos and neg: {correlation:.4f}")

# Sample some values
print("\nSample voxel values (first 10 non-zero):")
pos_nonzero_idx = np.where(pos_data != 0)
neg_nonzero_idx = np.where(neg_data != 0)

print("Positive map (first 10):")
for i in range(min(10, len(pos_nonzero_idx[0]))):
    idx = (pos_nonzero_idx[0][i], pos_nonzero_idx[1][i], pos_nonzero_idx[2][i])
    print(f"  {idx}: {pos_data[idx]:.8f}")

print("\nNegative map (first 10):")
for i in range(min(10, len(neg_nonzero_idx[0]))):
    idx = (neg_nonzero_idx[0][i], neg_nonzero_idx[1][i], neg_nonzero_idx[2][i])
    print(f"  {idx}: {neg_data[idx]:.8f}")

