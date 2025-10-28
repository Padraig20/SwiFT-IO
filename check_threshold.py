import nibabel as nib
import numpy as np
from pathlib import Path

# Load one example IG map
project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
ig_path = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments/sub-NDARUC771VM5/target0_Anger/sub-NDARUC771VM5_Anger_TR060_rank05_AVGpred_positive.nii.gz"

data = nib.load(str(ig_path)).get_fdata()
pos_values = data[data > 0]

print(f"Total positive voxels: {len(pos_values)}")
print(f"Min: {pos_values.min():.8f}")
print(f"Max: {pos_values.max():.8f}")
print(f"Mean: {pos_values.mean():.8f}")
print(f"Median: {np.median(pos_values):.8f}")
print()

# Test different percentiles
for p in [90, 95, 99, 99.5, 99.9]:
    threshold = np.percentile(pos_values, p)
    above_threshold = np.sum(pos_values >= threshold)
    percentage = (above_threshold / len(pos_values)) * 100
    print(f"Percentile {p:5.1f}: threshold={threshold:.8f}, above_threshold={above_threshold:6d} ({percentage:5.2f}%)")

