#!/usr/bin/env python3
"""
Comprehensive visualization of classification IG maps
Including:
1. Average brain maps (positive/negative)
2. Distribution histograms
3. Subject-level variance
4. Sequence-level variance (5 different sequences per emotion)
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
print("Comprehensive Classification IG Map Analysis")
print("="*70)

# Process each emotion
for emotion_idx, emotion in enumerate(emotion_labels):
    print(f"\n{emotion}:")

    # Organize data by subject and sequence rank
    subject_data = {}  # {subject: {rank: {'positive': data, 'negative': data}}}

    # Iterate through all subjects
    for subject_dir in ig_base.glob("sub-*"):
        subject_name = subject_dir.name
        emotion_dir = subject_dir / f"target{emotion_idx}_{emotion}"

        if not emotion_dir.exists():
            continue

        subject_data[subject_name] = {}

        # Collect files by rank
        for rank in range(1, 6):  # rank 1-5
            pos_files = list(emotion_dir.glob(f"*_rank{rank:02d}_AVGpred_positive.nii.gz"))
            neg_files = list(emotion_dir.glob(f"*_rank{rank:02d}_AVGpred_negative.nii.gz"))

            if pos_files and neg_files:
                try:
                    pos_data = nib.load(str(pos_files[0])).get_fdata()
                    neg_data = nib.load(str(neg_files[0])).get_fdata()

                    subject_data[subject_name][rank] = {
                        'positive': pos_data,
                        'negative': np.abs(neg_data)
                    }
                except Exception as e:
                    print(f"  ⚠️  Error loading {subject_name} rank {rank}: {e}")

    if not subject_data:
        print(f"  ⚠️  No data found for {emotion}")
        continue

    # Count sequences
    n_subjects = len(subject_data)
    total_sequences = sum(len(ranks) for ranks in subject_data.values())
    print(f"  {n_subjects} subjects, {total_sequences} sequences total")

    # Collect all positive and negative maps for averaging
    all_positive = []
    all_negative = []

    # Collect by subject for subject-level variance
    subject_means_pos = []
    subject_means_neg = []

    # Collect by sequence rank for sequence-level variance
    rank_means_pos = {r: [] for r in range(1, 6)}
    rank_means_neg = {r: [] for r in range(1, 6)}

    for subject_name, ranks_data in subject_data.items():
        subj_pos_maps = []
        subj_neg_maps = []

        for rank, data in ranks_data.items():
            pos_map = data['positive']
            neg_map = data['negative']

            all_positive.append(pos_map)
            all_negative.append(neg_map)

            subj_pos_maps.append(pos_map)
            subj_neg_maps.append(neg_map)

            rank_means_pos[rank].append(pos_map.mean())
            rank_means_neg[rank].append(neg_map.mean())

        if subj_pos_maps:
            subject_means_pos.append(np.mean(subj_pos_maps, axis=0).mean())
            subject_means_neg.append(np.mean(subj_neg_maps, axis=0).mean())

    # Calculate overall average
    avg_positive = np.mean(all_positive, axis=0)
    avg_negative = np.mean(all_negative, axis=0)

    # Get affine
    first_file = None
    for subject_dir in ig_base.glob("sub-*"):
        emotion_dir_check = subject_dir / f"target{emotion_idx}_{emotion}"
        if emotion_dir_check.exists():
            files = list(emotion_dir_check.glob("*.nii.gz"))
            if files:
                first_file = files[0]
                break

    if first_file is None:
        print(f"  ⚠️  Could not find affine reference")
        continue

    affine = nib.load(str(first_file)).affine

    # Save averaged maps
    avg_pos_path = output_dir / f"{emotion}_average_positive.nii.gz"
    avg_neg_path = output_dir / f"{emotion}_average_negative.nii.gz"
    nib.save(nib.Nifti1Image(avg_positive, affine), avg_pos_path)
    nib.save(nib.Nifti1Image(avg_negative, affine), avg_neg_path)

    # Calculate statistics
    pos_nonzero = avg_positive[avg_positive > 0]
    neg_nonzero = avg_negative[avg_negative > 0]

    # Use 90% threshold for better visualization (show top 10% of voxels)
    if len(pos_nonzero) > 100:
        pos_threshold = np.percentile(pos_nonzero, 90)
    else:
        pos_threshold = 0

    if len(neg_nonzero) > 100:
        neg_threshold = np.percentile(neg_nonzero, 90)
    else:
        neg_threshold = 0

    print(f"  Threshold (90%ile) - Positive: {pos_threshold:.2e}, Negative: {neg_threshold:.2e}")
    print(f"  Max values - Positive: {avg_positive.max():.2e}, Negative: {avg_negative.max():.2e}")

    # Also calculate mean within brain mask (non-zero voxels)
    print(f"  Mean (non-zero) - Positive: {pos_nonzero.mean():.2e}, Negative: {neg_nonzero.mean():.2e}")

    # ========== CREATE COMPREHENSIVE VISUALIZATION ==========
    fig = plt.figure(figsize=(16, 12))

    # Top row: Brain maps
    ax1 = plt.subplot(3, 2, 1)

    # For better visualization, show actual range info
    pos_max = avg_positive.max()
    pos_95 = np.percentile(pos_nonzero, 95) if len(pos_nonzero) > 0 else 0

    display = plotting.plot_stat_map(
        avg_pos_path,
        threshold=pos_threshold,
        cmap='hot',
        colorbar=True,
        vmax=pos_max,
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        annotate=True,
        black_bg=False,
        draw_cross=True,
        figure=fig,
        axes=ax1,
        title=f'{emotion} - Average Positive Attribution | first_10sec baseline\nN={total_sequences} sequences, {n_subjects} subjects\nmax={pos_max:.2e}, 95%={pos_95:.2e}'
    )

    ax2 = plt.subplot(3, 2, 2)

    neg_max = avg_negative.max()
    neg_95 = np.percentile(neg_nonzero, 95) if len(neg_nonzero) > 0 else 0

    display = plotting.plot_stat_map(
        avg_neg_path,
        threshold=neg_threshold,
        cmap='cool',
        colorbar=True,
        vmax=neg_max,
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        annotate=True,
        black_bg=False,
        draw_cross=True,
        figure=fig,
        axes=ax2,
        title=f'{emotion} - Average Negative Attribution (abs) | first_10sec baseline\nN={total_sequences} sequences, {n_subjects} subjects\nmax={neg_max:.2e}, 95%={neg_95:.2e}'
    )

    # Middle row: Distribution histograms
    ax3 = plt.subplot(3, 2, 3)
    pos_values = avg_positive[avg_positive > 0].flatten()
    if len(pos_values) > 0:
        ax3.hist(pos_values, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax3.axvline(pos_threshold, color='black', linestyle='--', linewidth=2, label='90% threshold')
        ax3.set_xlabel('IG values')
        ax3.set_ylabel('Frequency')
        ax3.set_yscale('log')
        ax3.set_title('Distribution of Positive IG values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 2, 4)
    neg_values = avg_negative[avg_negative > 0].flatten()
    if len(neg_values) > 0:
        ax4.hist(neg_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.axvline(neg_threshold, color='black', linestyle='--', linewidth=2, label='90% threshold')
        ax4.set_xlabel('IG values')
        ax4.set_ylabel('Frequency')
        ax4.set_yscale('log')
        ax4.set_title('Distribution of Negative IG values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Bottom left: Subject-level variance
    ax5 = plt.subplot(3, 2, 5)
    if len(subject_means_pos) > 1:
        subjects_sorted = sorted(subject_data.keys())
        subject_pos_vals = []
        subject_neg_vals = []

        for subj in subjects_sorted:
            if subject_data[subj]:
                subj_maps_pos = [subject_data[subj][r]['positive'] for r in subject_data[subj].keys()]
                subj_maps_neg = [subject_data[subj][r]['negative'] for r in subject_data[subj].keys()]
                subject_pos_vals.append(np.mean(subj_maps_pos, axis=0).mean())
                subject_neg_vals.append(np.mean(subj_maps_neg, axis=0).mean())

        x = np.arange(len(subjects_sorted))
        width = 0.35
        ax5.bar(x - width/2, subject_pos_vals, width, label='Positive', color='red', alpha=0.7)
        ax5.bar(x + width/2, subject_neg_vals, width, label='Negative', color='blue', alpha=0.7)
        ax5.set_xlabel('Subject')
        ax5.set_ylabel('Variance from mean')
        ax5.set_title(f'Subject-level Variance\n(variance between subject mean and overall mean)')
        ax5.set_xticks(x)
        ax5.set_xticklabels([s.replace('sub-NDAR', '') for s in subjects_sorted], rotation=45, ha='right', fontsize=8)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_yscale('log')

    # Bottom right: Sequence-level variance
    ax6 = plt.subplot(3, 2, 6)
    rank_labels = ['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5']
    rank_pos_means = [np.mean(rank_means_pos[r]) if rank_means_pos[r] else 0 for r in range(1, 6)]
    rank_neg_means = [np.mean(rank_means_neg[r]) if rank_means_neg[r] else 0 for r in range(1, 6)]

    x = np.arange(5)
    width = 0.35
    ax6.bar(x - width/2, rank_pos_means, width, label='Positive', color='red', alpha=0.7)
    ax6.bar(x + width/2, rank_neg_means, width, label='Negative', color='blue', alpha=0.7)
    ax6.set_xlabel('Variance from mean')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Sequence-level Variance Distribution\n(N={total_sequences} sequences)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(rank_labels)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"{emotion}_comprehensive_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  📊 Saved: {fig_path.name}")

print("\n" + "="*70)
print("✅ All comprehensive analyses completed!")
print(f"   Output directory: {output_dir}")
print("="*70)
