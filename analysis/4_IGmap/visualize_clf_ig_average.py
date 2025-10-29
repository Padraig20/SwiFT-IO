#!/usr/bin/env python3
"""
Visualize averaged IG maps for classification model
Creates brain maps similar to the regression visualization
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')

# Configuration
project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
ig_base = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/mc3r4vhf/nii_segments"
output_dir = project_root / "analysis/4_IGmap/clf_visualizations"
output_dir.mkdir(exist_ok=True, parents=True)

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

print("="*70)
print("Classification IG Map Averaging and Visualization")
print("="*70)

# Process each emotion
for emotion_idx, emotion in enumerate(emotion_labels):
    print(f"\n{emotion}:")

    # Collect all positive and negative IG maps for this emotion
    positive_maps = []
    negative_maps = []

    # Iterate through all subjects
    for subject_dir in ig_base.glob("sub-*"):
        emotion_dir = subject_dir / f"target{emotion_idx}_{emotion}"

        if not emotion_dir.exists():
            continue

        # Collect all positive and negative files
        pos_files = list(emotion_dir.glob("*_AVGpred_positive.nii.gz"))
        neg_files = list(emotion_dir.glob("*_AVGpred_negative.nii.gz"))

        for pos_file in pos_files:
            try:
                img = nib.load(str(pos_file))
                data = img.get_fdata()
                positive_maps.append(data)
            except Exception as e:
                print(f"  ⚠️  Error loading {pos_file.name}: {e}")

        for neg_file in neg_files:
            try:
                img = nib.load(str(neg_file))
                data = img.get_fdata()
                negative_maps.append(np.abs(data))  # Take absolute value for negative
            except Exception as e:
                print(f"  ⚠️  Error loading {neg_file.name}: {e}")

    if not positive_maps or not negative_maps:
        print(f"  ⚠️  No maps found for {emotion}")
        continue

    print(f"  Loaded {len(positive_maps)} positive maps, {len(negative_maps)} negative maps")

    # Calculate average
    avg_positive = np.mean(positive_maps, axis=0)
    avg_negative = np.mean(negative_maps, axis=0)

    # Get affine from any subject's first file
    first_file = None
    for subject_dir in ig_base.glob("sub-*"):
        emotion_dir_check = subject_dir / f"target{emotion_idx}_{emotion}"
        if emotion_dir_check.exists():
            files = list(emotion_dir_check.glob("*.nii.gz"))
            if files:
                first_file = files[0]
                break

    if first_file is None:
        print(f"  ⚠️  Could not find affine reference file")
        continue

    affine = nib.load(str(first_file)).affine

    # Save averaged maps
    avg_pos_path = output_dir / f"{emotion}_average_positive.nii.gz"
    avg_neg_path = output_dir / f"{emotion}_average_negative.nii.gz"

    nib.save(nib.Nifti1Image(avg_positive, affine), avg_pos_path)
    nib.save(nib.Nifti1Image(avg_negative, affine), avg_neg_path)

    print(f"  ✅ Saved: {avg_pos_path.name}, {avg_neg_path.name}")

    # Calculate statistics
    pos_nonzero = avg_positive[avg_positive > 0]
    neg_nonzero = avg_negative[avg_negative > 0]

    # Use 99th percentile as threshold
    if len(pos_nonzero) > 0:
        pos_threshold = np.percentile(pos_nonzero, 99)
    else:
        pos_threshold = 0

    if len(neg_nonzero) > 0:
        neg_threshold = np.percentile(neg_nonzero, 99)
    else:
        neg_threshold = 0

    print(f"  Positive - mean: {avg_positive.mean():.2e}, max: {avg_positive.max():.2e}, 99%: {pos_threshold:.2e}")
    print(f"  Negative - mean: {avg_negative.mean():.2e}, max: {avg_negative.max():.2e}, 99%: {neg_threshold:.2e}")

    # Create visualization
    fig = plt.figure(figsize=(14, 6))

    # Positive attribution
    ax1 = plt.subplot(1, 2, 1)
    display = plotting.plot_stat_map(
        avg_pos_path,
        threshold=pos_threshold,
        cmap='hot',
        colorbar=True,
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        annotate=True,
        black_bg=False,
        draw_cross=True,
        figure=fig,
        axes=ax1,
        title=f'{emotion} - Average Positive Attribution | first_10sec baseline\nN={len(positive_maps)} sequences, {len(set([p.parent.parent.name for p in emotion_dir.glob("*_positive.nii.gz")]))} subjects'
    )

    # Negative attribution (absolute values)
    ax2 = plt.subplot(1, 2, 2)
    display = plotting.plot_stat_map(
        avg_neg_path,
        threshold=neg_threshold,
        cmap='cool',
        colorbar=True,
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        annotate=True,
        black_bg=False,
        draw_cross=True,
        figure=fig,
        axes=ax2,
        title=f'{emotion} - Average Negative Attribution (abs) | first_10sec baseline\nN={len(negative_maps)} sequences, {len(set([p.parent.parent.name for p in emotion_dir.glob("*_negative.nii.gz")]))} subjects'
    )

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"{emotion}_clf_ig_average.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  📊 Saved visualization: {fig_path.name}")

print("\n" + "="*70)
print("✅ All visualizations completed!")
print(f"   Output directory: {output_dir}")
print("="*70)
