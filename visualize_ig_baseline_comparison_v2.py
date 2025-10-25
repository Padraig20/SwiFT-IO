#!/usr/bin/env python3
"""
Visualize IG baseline comparison with same style as previous work
"""

import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from pathlib import Path

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def visualize_igmap_baseline(subject, emotion_idx, baseline_type, rank_info, base_dir, percentile_threshold=95):
    """Visualize IG map for baseline comparison"""

    emotion = emotion_labels[emotion_idx]
    tr, rank, avg_score = rank_info

    # Paths for positive and negative attribution
    pos_path = base_dir / subject / f"target{emotion_idx}_{emotion}" / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_AVGpred_positive.nii.gz"
    neg_path = base_dir / subject / f"target{emotion_idx}_{emotion}" / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_AVGpred_negative.nii.gz"

    if not pos_path.exists() or not neg_path.exists():
        print(f"❌ Files not found: {pos_path}")
        return None

    # Load IG maps
    print(f"📂 Loading: {pos_path.name}")
    pos_img = nib.load(str(pos_path))
    pos_data = pos_img.get_fdata()

    neg_img = nib.load(str(neg_path))
    neg_data = neg_img.get_fdata()

    # Basic statistics
    print(f"\n📊 Statistics for {subject} - {emotion} - TR{tr} Rank{rank} (baseline={baseline_type}):")
    print(f"   Positive - Min: {pos_data.min():.8f}, Max: {pos_data.max():.8f}, Mean: {pos_data.mean():.8f}")
    print(f"   Negative - Min: {neg_data.min():.8f}, Max: {neg_data.max():.8f}, Mean: {neg_data.mean():.8f}")

    # Get non-zero values
    pos_values = pos_data[pos_data > 0]
    neg_values = neg_data[neg_data < 0]

    print(f"   Positive voxels: {len(pos_values)} (mean: {pos_values.mean():.8f})")
    print(f"   Negative voxels: {len(neg_values)} (mean: {neg_values.mean():.8f})")

    # Calculate thresholds
    if len(pos_values) > 0:
        pos_threshold = np.percentile(pos_values, percentile_threshold)
    else:
        pos_threshold = 0

    if len(neg_values) > 0:
        neg_threshold = np.percentile(np.abs(neg_values), percentile_threshold)
    else:
        neg_threshold = 0

    print(f"   Positive threshold (top {100-percentile_threshold}%): {pos_threshold:.8f}")
    print(f"   Negative threshold (top {100-percentile_threshold}%): {neg_threshold:.8f}")

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Plot 1: Positive contributions
    ax1 = plt.subplot(2, 2, 1)
    plotting.plot_stat_map(
        pos_img,
        title=f"{emotion} - TR{tr} Rank{rank} (Positive) | {baseline_type} baseline\nAvg score: {avg_score:.3f}",
        display_mode='ortho',
        cmap='hot',
        threshold=pos_threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax1,
        vmax=pos_data.max() if pos_data.max() > 0 else 1e-8
    )

    # Plot 2: Negative contributions (use absolute values)
    ax2 = plt.subplot(2, 2, 2)
    neg_abs_data = np.abs(neg_data)
    neg_abs_img = nib.Nifti1Image(neg_abs_data, neg_img.affine)
    plotting.plot_stat_map(
        neg_abs_img,
        title=f"{emotion} - TR{tr} Rank{rank} (Negative - abs) | {baseline_type} baseline\nAvg score: {avg_score:.3f}",
        display_mode='ortho',
        cmap='hot',
        threshold=neg_threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax2,
        vmax=neg_abs_data.max() if neg_abs_data.max() > 0 else 1e-8
    )

    # Plot 3: Histogram of positive values
    ax3 = plt.subplot(2, 2, 3)
    if len(pos_values) > 0:
        ax3.hist(pos_values, bins=100, color='red', alpha=0.7)
        ax3.axvline(pos_threshold, color='black', linestyle='--', linewidth=2,
                   label=f'{100-percentile_threshold}% threshold')
        ax3.set_xlabel('IG values')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Positive IG values')
        ax3.legend()
        ax3.set_yscale('log')

    # Plot 4: Histogram of negative values
    ax4 = plt.subplot(2, 2, 4)
    if len(neg_values) > 0:
        ax4.hist(neg_values, bins=100, color='blue', alpha=0.7)
        ax4.axvline(-neg_threshold, color='black', linestyle='--', linewidth=2,
                   label=f'{100-percentile_threshold}% threshold')
        ax4.set_xlabel('IG values')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Negative IG values')
        ax4.legend()
        ax4.set_yscale('log')

    plt.tight_layout()

    # Save figure
    out_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/analysis/4_IGmap/visualizations/baseline_comparison_{baseline_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_baseline_{baseline_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {out_path.name}\n")

    plt.close()

    return pos_data, neg_data

def compare_baselines_for_rank(subject, emotion_idx, rank_info, percentile_threshold=95):
    """Compare two baselines (zeros vs first_10sec) for the same rank"""

    emotion = emotion_labels[emotion_idx]
    tr, rank, avg_score = rank_info

    project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    for row, baseline_type in enumerate(['first_10sec', 'zeros']):
        base_dir = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments"

        # Paths
        pos_path = base_dir / subject / f"target{emotion_idx}_{emotion}" / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_AVGpred_positive.nii.gz"
        neg_path = base_dir / subject / f"target{emotion_idx}_{emotion}" / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_AVGpred_negative.nii.gz"

        if not pos_path.exists():
            print(f"⚠️ Skipping {baseline_type} - file not found")
            continue

        # Load data
        pos_img = nib.load(str(pos_path))
        pos_data = pos_img.get_fdata()
        neg_img = nib.load(str(neg_path))
        neg_data = neg_img.get_fdata()

        # Calculate thresholds
        pos_values = pos_data[pos_data > 0]
        neg_values = neg_data[neg_data < 0]

        if len(pos_values) > 0:
            pos_threshold = np.percentile(pos_values, percentile_threshold)
        else:
            pos_threshold = 0

        if len(neg_values) > 0:
            neg_threshold = np.percentile(np.abs(neg_values), percentile_threshold)
        else:
            neg_threshold = 0

        # Plot positive
        plotting.plot_stat_map(
            pos_img,
            title=f"Baseline: {baseline_type} (Positive)",
            display_mode='z',
            cut_coords=5,
            cmap='hot',
            threshold=pos_threshold,
            colorbar=True,
            axes=axes[row, 0],
            vmax=pos_data.max() if pos_data.max() > 0 else 1e-8
        )

        # Plot negative
        neg_abs_data = np.abs(neg_data)
        neg_abs_img = nib.Nifti1Image(neg_abs_data, neg_img.affine)
        plotting.plot_stat_map(
            neg_abs_img,
            title=f"Baseline: {baseline_type} (Negative - abs)",
            display_mode='z',
            cut_coords=5,
            cmap='hot',
            threshold=neg_threshold,
            colorbar=True,
            axes=axes[row, 1],
            vmax=neg_abs_data.max() if neg_abs_data.max() > 0 else 1e-8
        )

        # Combined histogram
        ax_hist = axes[row, 2]
        if len(pos_values) > 0:
            ax_hist.hist(pos_values, bins=50, color='red', alpha=0.5, label='Positive')
        if len(neg_values) > 0:
            ax_hist.hist(neg_values, bins=50, color='blue', alpha=0.5, label='Negative')
        ax_hist.set_xlabel('IG values')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'Baseline: {baseline_type}')
        ax_hist.legend()
        ax_hist.set_yscale('log')

    plt.suptitle(f"{subject} - {emotion} - TR{tr} Rank{rank} (Avg: {avg_score:.3f})\nBaseline Comparison",
                fontsize=16, y=0.995)
    plt.tight_layout()

    # Save
    out_dir = project_root / "analysis/4_IGmap/visualizations/baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{emotion}_TR{tr:03d}_rank{rank:02d}_baseline_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved baseline comparison: {out_path.name}")

    plt.close()

if __name__ == "__main__":
    project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

    subject = "sub-NDARMZ366UY8"
    emotion_idx = 5  # Positive
    emotion = emotion_labels[emotion_idx]

    # Rank information: (TR, rank, avg_score)
    ranks_info = [
        (450, 1, 4.018),
        (300, 2, 2.827),
        (420, 3, 2.730),
        (480, 4, 1.817),
        (90, 5, 1.580),
    ]

    print("="*70)
    print(f"IG Map Baseline Comparison Visualization")
    print(f"Subject: {subject} (Positive top 1)")
    print(f"Emotion: {emotion}")
    print("="*70)

    # Visualize first_10sec baseline
    print("\n📊 Processing baseline: first_10sec")
    print("-"*70)
    base_dir_10sec = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments"
    for rank_info in ranks_info:
        visualize_igmap_baseline(subject, emotion_idx, 'first_10sec', rank_info, base_dir_10sec)

    # Visualize zeros baseline (if available)
    print("\n📊 Processing baseline: zeros")
    print("-"*70)
    base_dir_zeros = project_root / "analysis/4_IGmap/baseline_zeros_selective/opr6oq97/nii_segments"
    for rank_info in ranks_info:
        result = visualize_igmap_baseline(subject, emotion_idx, 'zeros', rank_info, base_dir_zeros)
        if result is None:
            print("⚠️ Zeros baseline not complete yet, skipping remaining ranks")
            break

    # Create baseline comparison for rank 1 (if both available)
    print("\n📊 Creating baseline comparison for Rank 1")
    print("-"*70)
    compare_baselines_for_rank(subject, emotion_idx, ranks_info[0])

    print("\n"+"="*70)
    print("✅ All visualizations complete!")
    print("="*70)
