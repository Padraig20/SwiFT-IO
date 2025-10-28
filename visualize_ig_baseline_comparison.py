#!/usr/bin/env python3
"""
Visualize IG maps from baseline comparison
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn import plotting
import sys

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Results directory
baseline_type = "first_10sec"  # or "zeros"
subject = "sub-NDARMZ366UY8"
emotion = "Positive"
emotion_idx = 5

result_dir = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments/{subject}/target{emotion_idx}_{emotion}"

print(f"{'='*70}")
print(f"IG Map Visualization - Baseline: {baseline_type}")
print(f"Subject: {subject} (Positive top 1)")
print(f"{'='*70}")

# Get all result files
pos_files = sorted(result_dir.glob("*_positive.nii.gz"))
neg_files = sorted(result_dir.glob("*_negative.nii.gz"))

print(f"\nFound {len(pos_files)} positive and {len(neg_files)} negative attribution maps")

# Create output directory for visualizations
vis_dir = project_root / f"analysis/4_IGmap/visualizations/baseline_comparison_{baseline_type}"
vis_dir.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {vis_dir}")

# Process each rank
ranks_info = [
    ("TR450", "rank01", 4.018),
    ("TR300", "rank02", 2.827),
    ("TR420", "rank03", 2.730),
    ("TR480", "rank04", 1.817),
    ("TR090", "rank05", 1.580),
]

print(f"\n{'='*70}")
print("Statistics and Visualization")
print(f"{'='*70}")

for tr, rank, avg_score in ranks_info:
    # Find files
    pos_file = [f for f in pos_files if f"{tr}_{rank}" in f.name]
    neg_file = [f for f in neg_files if f"{tr}_{rank}" in f.name]

    if not pos_file or not neg_file:
        print(f"\n⚠️ Missing files for {rank} ({tr})")
        continue

    pos_file = pos_file[0]
    neg_file = neg_file[0]

    print(f"\n{'-'*70}")
    print(f"Rank {rank[-1]}: {tr} (Positive avg: {avg_score:.3f})")
    print(f"{'-'*70}")

    # Load data
    pos_img = nib.load(str(pos_file))
    neg_img = nib.load(str(neg_file))
    pos_data = pos_img.get_fdata()
    neg_data = neg_img.get_fdata()

    # Statistics
    print(f"\nPositive Attribution:")
    print(f"  Shape: {pos_data.shape}")
    print(f"  Min: {np.min(pos_data):.6f}")
    print(f"  Max: {np.max(pos_data):.6f}")
    print(f"  Mean: {np.mean(pos_data):.6f}")
    print(f"  Non-zero voxels: {np.count_nonzero(pos_data)}/{pos_data.size}")
    print(f"  95th percentile: {np.percentile(pos_data, 95):.6f}")
    print(f"  99th percentile: {np.percentile(pos_data, 99):.6f}")

    print(f"\nNegative Attribution:")
    print(f"  Shape: {neg_data.shape}")
    print(f"  Min: {np.min(neg_data):.6f}")
    print(f"  Max: {np.max(neg_data):.6f}")
    print(f"  Mean: {np.mean(neg_data):.6f}")
    print(f"  Non-zero voxels: {np.count_nonzero(neg_data)}/{neg_data.size}")
    print(f"  5th percentile: {np.percentile(neg_data, 5):.6f}")
    print(f"  1st percentile: {np.percentile(neg_data, 1):.6f}")

    # Create visualizations
    fig = plt.figure(figsize=(20, 10))

    # Positive attribution - glass brain
    ax1 = plt.subplot(2, 2, 1)
    try:
        # Use 95th percentile for threshold
        threshold = np.percentile(pos_data[pos_data > 0], 95) if np.any(pos_data > 0) else 0
        plotting.plot_glass_brain(
            pos_img,
            threshold=threshold,
            colorbar=True,
            cmap='hot',
            plot_abs=False,
            display_mode='ortho',
            title=f'Positive Attribution - {rank} ({tr})\nThreshold: 95th percentile',
            axes=ax1
        )
    except Exception as e:
        print(f"  ⚠️ Glass brain visualization failed: {e}")

    # Negative attribution - glass brain
    ax2 = plt.subplot(2, 2, 2)
    try:
        # Use 5th percentile for threshold (negative values)
        threshold = abs(np.percentile(neg_data[neg_data < 0], 5)) if np.any(neg_data < 0) else 0
        plotting.plot_glass_brain(
            neg_img,
            threshold=-threshold,
            colorbar=True,
            cmap='cold',
            plot_abs=False,
            display_mode='ortho',
            title=f'Negative Attribution - {rank} ({tr})\nThreshold: 5th percentile',
            axes=ax2
        )
    except Exception as e:
        print(f"  ⚠️ Glass brain visualization failed: {e}")

    # Positive - statistical map
    ax3 = plt.subplot(2, 2, 3)
    try:
        threshold = np.percentile(pos_data[pos_data > 0], 90) if np.any(pos_data > 0) else 0
        plotting.plot_stat_map(
            pos_img,
            threshold=threshold,
            colorbar=True,
            cmap='hot',
            cut_coords=5,
            display_mode='z',
            title=f'Positive - Axial slices (90th %ile)',
            axes=ax3
        )
    except Exception as e:
        print(f"  ⚠️ Stat map visualization failed: {e}")

    # Negative - statistical map
    ax4 = plt.subplot(2, 2, 4)
    try:
        threshold = abs(np.percentile(neg_data[neg_data < 0], 10)) if np.any(neg_data < 0) else 0
        plotting.plot_stat_map(
            neg_img,
            threshold=-threshold,
            colorbar=True,
            cmap='cold',
            cut_coords=5,
            display_mode='z',
            title=f'Negative - Axial slices (10th %ile)',
            axes=ax4
        )
    except Exception as e:
        print(f"  ⚠️ Stat map visualization failed: {e}")

    plt.tight_layout()

    # Save figure
    output_path = vis_dir / f"{subject}_{emotion}_{tr}_{rank}_baseline_{baseline_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved: {output_path.name}")
    plt.close()

print(f"\n{'='*70}")
print(f"✅ All visualizations complete!")
print(f"   Output: {vis_dir}")
print(f"{'='*70}")
