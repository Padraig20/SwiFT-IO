#!/usr/bin/env python3
"""
Visualize averaged IG maps with tri-color coding:
- Red: Positive dominant
- Blue: Negative dominant
- Purple: Mixed (both positive and negative)
"""

import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
from collections import defaultdict

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def load_emotion_ig_maps(emotion, baseline_type='first_10sec', project_root='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO'):
    """Load all IG maps for a specific emotion across all subjects"""
    project_root = Path(project_root)
    base_dir = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments"

    json_path = project_root / "analysis/4_IGmap/subject_performance/opr6oq97_emotion_specific_mse.json"
    with open(json_path) as f:
        data = json.load(f)

    emotion_idx = emotion_labels.index(emotion)
    subjects = data['top5_per_emotion'][emotion]

    print(f"\n{'='*70}")
    print(f"Loading IG maps for emotion: {emotion}")
    print(f"  Baseline: {baseline_type}")
    print(f"  Subjects: {subjects}")
    print(f"{'='*70}\n")

    pos_maps = []
    neg_maps = []

    for subject in subjects:
        subject_dir = base_dir / subject / f"target{emotion_idx}_{emotion}"

        if not subject_dir.exists():
            print(f"⚠️  Subject directory not found: {subject}")
            continue

        pos_files = sorted(subject_dir.glob(f"{subject}_{emotion}_TR*_rank*_AVGpred_positive.nii.gz"))

        for pos_path in pos_files:
            filename = pos_path.stem.replace('.nii', '')
            parts = filename.split('_')
            tr_str = [p for p in parts if p.startswith('TR')][0]
            rank_str = [p for p in parts if p.startswith('rank')][0]

            tr = int(tr_str.replace('TR', ''))
            rank = int(rank_str.replace('rank', ''))

            neg_path = pos_path.parent / pos_path.name.replace('positive', 'negative')

            if not neg_path.exists():
                print(f"⚠️  Negative file missing: {neg_path.name}")
                continue

            pos_data = nib.load(str(pos_path)).get_fdata()
            neg_data = nib.load(str(neg_path)).get_fdata()

            seq_info = {'subject': subject, 'tr': tr, 'rank': rank}

            pos_maps.append((subject, seq_info, pos_data))
            neg_maps.append((subject, seq_info, neg_data))

            print(f"  ✅ Loaded: {subject} - TR{tr:03d} Rank{rank}")

    print(f"\n📊 Total loaded: {len(pos_maps)} sequences from {len(subjects)} subjects\n")

    return pos_maps, neg_maps


def compute_net_and_category_maps(pos_maps, neg_maps, percentile_threshold=99):
    """
    Compute net attribution and categorize voxels

    Returns:
        category_map: 0=background, 1=positive_dominant, 2=negative_dominant, 3=mixed
        net_map: net attribution (positive + negative)
        pos_mean: average positive attribution
        neg_mean: average negative attribution
    """
    # Extract data arrays
    pos_arrays = [data for _, _, data in pos_maps]
    neg_arrays = [data for _, _, data in neg_maps]

    # Average
    pos_mean = np.mean(pos_arrays, axis=0)
    neg_mean = np.mean(neg_arrays, axis=0)

    # Net attribution
    net_map = pos_mean + neg_mean  # negative values are already negative

    # Get absolute values for thresholding
    pos_abs = np.abs(pos_mean)
    neg_abs = np.abs(neg_mean)

    # Compute threshold based on the stronger signal
    max_abs = np.maximum(pos_abs, neg_abs)
    active_voxels = max_abs[max_abs > 0]

    if len(active_voxels) > 0:
        threshold = np.percentile(active_voxels, percentile_threshold)
    else:
        threshold = 0

    print(f"Threshold (top {100-percentile_threshold}%): {threshold:.8f}")

    # Categorize voxels
    category_map = np.zeros_like(pos_mean, dtype=np.int8)

    # Above threshold mask
    above_threshold = max_abs >= threshold

    # Define dominance: ratio of positive vs total
    total_abs = pos_abs + neg_abs
    # Avoid division by zero
    ratio = np.divide(pos_abs, total_abs, out=np.zeros_like(pos_abs), where=total_abs > 0)

    # Categorize:
    # 1 = Positive dominant (ratio > 0.7)
    # 2 = Negative dominant (ratio < 0.3)
    # 3 = Mixed (0.3 <= ratio <= 0.7)

    positive_dominant = above_threshold & (ratio > 0.7)
    negative_dominant = above_threshold & (ratio < 0.3)
    mixed = above_threshold & (ratio >= 0.3) & (ratio <= 0.7)

    category_map[positive_dominant] = 1
    category_map[negative_dominant] = 2
    category_map[mixed] = 3

    # Statistics
    n_pos = np.sum(positive_dominant)
    n_neg = np.sum(negative_dominant)
    n_mixed = np.sum(mixed)
    total_active = n_pos + n_neg + n_mixed

    print(f"\nVoxel categories (above threshold):")
    print(f"  Positive dominant: {n_pos:6d} ({100*n_pos/total_active:.1f}%)")
    print(f"  Negative dominant: {n_neg:6d} ({100*n_neg/total_active:.1f}%)")
    print(f"  Mixed (both):      {n_mixed:6d} ({100*n_mixed/total_active:.1f}%)")
    print(f"  Total active:      {total_active:6d}")

    return category_map, net_map, pos_mean, neg_mean, threshold


def visualize_emotion_tricolor(emotion, baseline_type='first_10sec', percentile_threshold=99):
    """Visualize with tri-color coding"""

    # Load maps
    pos_maps, neg_maps = load_emotion_ig_maps(emotion, baseline_type)

    if len(pos_maps) == 0:
        print(f"❌ No IG maps found for {emotion}")
        return

    print("Computing net attribution and categories...")
    category_map, net_map, pos_mean, neg_mean, threshold = compute_net_and_category_maps(
        pos_maps, neg_maps, percentile_threshold
    )

    # Get affine
    project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
    emotion_idx = emotion_labels.index(emotion)
    first_subject = pos_maps[0][0]
    first_tr = pos_maps[0][1]['tr']
    first_rank = pos_maps[0][1]['rank']

    ref_path = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments" / \
               first_subject / f"target{emotion_idx}_{emotion}" / \
               f"{first_subject}_{emotion}_TR{first_tr:03d}_rank{first_rank:02d}_AVGpred_positive.nii.gz"
    ref_img = nib.load(str(ref_path))
    affine = ref_img.affine

    # Create tri-color map
    # We'll create a single map where:
    # - Positive dominant: positive values of net_map
    # - Negative dominant: negative values of net_map
    # - Mixed: we need a special approach

    # For visualization, let's create 3 separate maps
    pos_dominant_map = np.zeros_like(net_map)
    neg_dominant_map = np.zeros_like(net_map)
    mixed_map = np.zeros_like(net_map)

    pos_dominant_map[category_map == 1] = pos_mean[category_map == 1]
    neg_dominant_map[category_map == 2] = np.abs(neg_mean[category_map == 2])
    mixed_map[category_map == 3] = np.maximum(pos_mean[category_map == 3], np.abs(neg_mean[category_map == 3]))

    # Create NIfTI images
    pos_dominant_img = nib.Nifti1Image(pos_dominant_map, affine)
    neg_dominant_img = nib.Nifti1Image(neg_dominant_map, affine)
    mixed_img = nib.Nifti1Image(mixed_map, affine)
    net_img = nib.Nifti1Image(net_map, affine)

    # Create figure
    fig = plt.figure(figsize=(20, 15))

    # Row 1: Tri-color separated
    ax1 = plt.subplot(3, 3, 1)
    if pos_dominant_map.max() > 0:
        plotting.plot_stat_map(
            pos_dominant_img,
            title=f"{emotion} - Positive Dominant (Red)\n{np.sum(category_map==1)} voxels",
            display_mode='ortho',
            cmap='Reds',
            threshold=1e-10,
            cut_coords=(0, 0, 0),
            colorbar=True,
            axes=ax1,
            vmax=pos_dominant_map.max()
        )

    ax2 = plt.subplot(3, 3, 2)
    if neg_dominant_map.max() > 0:
        plotting.plot_stat_map(
            neg_dominant_img,
            title=f"{emotion} - Negative Dominant (Blue)\n{np.sum(category_map==2)} voxels",
            display_mode='ortho',
            cmap='Blues',
            threshold=1e-10,
            cut_coords=(0, 0, 0),
            colorbar=True,
            axes=ax2,
            vmax=neg_dominant_map.max()
        )

    ax3 = plt.subplot(3, 3, 3)
    if mixed_map.max() > 0:
        plotting.plot_stat_map(
            mixed_img,
            title=f"{emotion} - Mixed/Both (Purple)\n{np.sum(category_map==3)} voxels",
            display_mode='ortho',
            cmap='Purples',
            threshold=1e-10,
            cut_coords=(0, 0, 0),
            colorbar=True,
            axes=ax3,
            vmax=mixed_map.max()
        )

    # Row 2: Net attribution (diverging colormap)
    ax4 = plt.subplot(3, 3, 4)
    plotting.plot_stat_map(
        net_img,
        title=f"{emotion} - Net Attribution (Pos + Neg)\nRed=Positive, Blue=Negative",
        display_mode='ortho',
        cmap='RdBu_r',
        threshold=threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax4,
        vmax=max(abs(net_map.min()), abs(net_map.max()))
    )

    # Row 2: Category distribution pie chart
    ax5 = plt.subplot(3, 3, 5)
    counts = [np.sum(category_map == 1), np.sum(category_map == 2), np.sum(category_map == 3)]
    labels = ['Positive\nDominant', 'Negative\nDominant', 'Mixed\n(Both)']
    colors = ['red', 'blue', 'purple']
    ax5.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Voxel Category Distribution')

    # Row 2: Ratio histogram
    ax6 = plt.subplot(3, 3, 6)
    active_voxels = category_map > 0
    pos_abs = np.abs(pos_mean[active_voxels])
    neg_abs = np.abs(neg_mean[active_voxels])
    total_abs = pos_abs + neg_abs
    ratio_values = pos_abs / total_abs

    ax6.hist(ratio_values, bins=50, color='gray', alpha=0.7, edgecolor='black')
    ax6.axvline(0.3, color='blue', linestyle='--', linewidth=2, label='Negative threshold')
    ax6.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Positive threshold')
    ax6.set_xlabel('Positive ratio = |pos| / (|pos| + |neg|)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Positive/Negative Ratio\n(active voxels only)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Row 3: Original positive and negative averages
    ax7 = plt.subplot(3, 3, 7)
    pos_mean_img = nib.Nifti1Image(pos_mean, affine)
    plotting.plot_stat_map(
        pos_mean_img,
        title=f"{emotion} - Average Positive Attribution",
        display_mode='ortho',
        cmap='Reds',
        threshold=threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax7,
        vmax=pos_mean.max()
    )

    ax8 = plt.subplot(3, 3, 8)
    neg_abs_img = nib.Nifti1Image(np.abs(neg_mean), affine)
    plotting.plot_stat_map(
        neg_abs_img,
        title=f"{emotion} - Average Negative Attribution (abs)",
        display_mode='ortho',
        cmap='Blues',
        threshold=threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax8,
        vmax=np.abs(neg_mean).max()
    )

    # Row 3: Statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    stats_text = f"""
{emotion} Statistics
N = {len(pos_maps)} sequences, {len(set([s for s, _, _ in pos_maps]))} subjects
Baseline: {baseline_type}
Threshold: top {100-percentile_threshold}% = {threshold:.8f}

Net Attribution:
  Min: {net_map.min():.8f}
  Max: {net_map.max():.8f}
  Mean: {net_map.mean():.8f}

Positive Attribution:
  Max: {pos_mean.max():.8f}
  Mean (active): {pos_mean[pos_mean > 0].mean():.8f}

Negative Attribution:
  Min: {neg_mean.min():.8f}
  Mean (active): {neg_mean[neg_mean < 0].mean():.8f}

Categorization:
  Positive dominant: {np.sum(category_map == 1)} voxels
  Negative dominant: {np.sum(category_map == 2)} voxels
  Mixed (both): {np.sum(category_map == 3)} voxels
"""
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)

    plt.tight_layout()

    # Save
    out_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/analysis/4_IGmap/visualizations/emotion_tricolor_{baseline_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{emotion}_tricolor_N{len(pos_maps)}_baseline_{baseline_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {out_path}\n")

    plt.close()

    return category_map, net_map


def main():
    """Process all emotions"""

    baseline_type = 'first_10sec'
    percentile_threshold = 99  # top 1%

    print(f"\n{'='*80}")
    print(f"Visualizing Emotion IG Maps with Tri-Color Coding")
    print(f"  Baseline: {baseline_type}")
    print(f"  Threshold: top {100-percentile_threshold}%")
    print(f"  Emotions: {emotion_labels}")
    print(f"{'='*80}\n")

    for emotion in emotion_labels:
        try:
            visualize_emotion_tricolor(emotion, baseline_type=baseline_type,
                                      percentile_threshold=percentile_threshold)
        except Exception as e:
            print(f"❌ Error processing {emotion}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("✅ All emotions processed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
