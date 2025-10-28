#!/usr/bin/env python3
"""
Visualize averaged IG maps across multiple subjects for each emotion
with subject-level and sequence-level variance analysis
"""

import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def load_emotion_ig_maps(emotion, baseline_type='first_10sec', project_root='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO'):
    """
    Load all IG maps for a specific emotion across all subjects

    Returns:
        pos_maps: list of (subject, sequence_info, positive_data) tuples
        neg_maps: list of (subject, sequence_info, negative_data) tuples
    """
    project_root = Path(project_root)
    base_dir = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments"

    # Load emotion-specific top 5 subjects
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

        # Find all IG map files for this subject
        pos_files = sorted(subject_dir.glob(f"{subject}_{emotion}_TR*_rank*_AVGpred_positive.nii.gz"))

        for pos_path in pos_files:
            # Extract TR and rank from filename
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

            # Load data
            pos_data = nib.load(str(pos_path)).get_fdata()
            neg_data = nib.load(str(neg_path)).get_fdata()

            seq_info = {'subject': subject, 'tr': tr, 'rank': rank}

            pos_maps.append((subject, seq_info, pos_data))
            neg_maps.append((subject, seq_info, neg_data))

            print(f"  ✅ Loaded: {subject} - TR{tr:03d} Rank{rank}")

    print(f"\n📊 Total loaded: {len(pos_maps)} sequences from {len(subjects)} subjects\n")

    return pos_maps, neg_maps


def compute_variance_stats(maps_list):
    """
    Compute subject-level and sequence-level variance

    Args:
        maps_list: list of (subject, seq_info, data) tuples

    Returns:
        subject_variances: dict {subject: variance_value}
        sequence_variances: list of variance values per sequence
        mean_map: averaged map across all
    """
    # Extract data arrays
    data_arrays = [data for _, _, data in maps_list]
    subjects = [subj for subj, _, _ in maps_list]

    # Overall mean across all subjects and sequences
    mean_map = np.mean(data_arrays, axis=0)

    # Subject-level variance: compute each subject's mean, then variance from overall mean
    subject_means = {}
    for subj, seq_info, data in maps_list:
        if subj not in subject_means:
            subject_means[subj] = []
        subject_means[subj].append(data)

    subject_variances = {}
    for subj, data_list in subject_means.items():
        subj_mean = np.mean(data_list, axis=0)
        # Compute variance between subject mean and overall mean
        variance = np.var(subj_mean - mean_map)
        subject_variances[subj] = variance

    # Sequence-level variance: variance of each sequence from overall mean
    sequence_variances = []
    for _, seq_info, data in maps_list:
        variance = np.var(data - mean_map)
        sequence_variances.append(variance)

    return subject_variances, sequence_variances, mean_map


def visualize_emotion_average(emotion, baseline_type='first_10sec', percentile_threshold=99):
    """
    Visualize averaged IG map for an emotion across all subjects
    with variance analysis
    """

    # Load all IG maps for this emotion
    pos_maps, neg_maps = load_emotion_ig_maps(emotion, baseline_type)

    if len(pos_maps) == 0:
        print(f"❌ No IG maps found for {emotion}")
        return

    # Compute variance statistics
    print("Computing variance statistics...")
    pos_subj_var, pos_seq_var, pos_mean = compute_variance_stats(pos_maps)
    neg_subj_var, neg_seq_var, neg_mean = compute_variance_stats(neg_maps)

    # Get reference affine from first map
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

    # Create NIfTI images
    pos_mean_img = nib.Nifti1Image(pos_mean, affine)
    neg_mean_img = nib.Nifti1Image(neg_mean, affine)

    # Statistics
    print(f"\n📊 Average IG Map Statistics for {emotion}:")
    print(f"   Positive - Min: {pos_mean.min():.8f}, Max: {pos_mean.max():.8f}, Mean: {pos_mean.mean():.8f}")
    print(f"   Negative - Min: {neg_mean.min():.8f}, Max: {neg_mean.max():.8f}, Mean: {neg_mean.mean():.8f}")

    # Get non-zero values
    pos_values = pos_mean[pos_mean > 0]
    neg_values = neg_mean[neg_mean < 0]

    print(f"   Positive voxels: {len(pos_values)} (mean: {pos_values.mean():.8f})")
    print(f"   Negative voxels: {len(neg_values)} (mean: {neg_values.mean():.8f})")

    # Calculate thresholds
    pos_threshold = np.percentile(pos_values, percentile_threshold) if len(pos_values) > 0 else 0
    neg_threshold = np.percentile(np.abs(neg_values), percentile_threshold) if len(neg_values) > 0 else 0

    print(f"   Positive threshold (top {100-percentile_threshold}%): {pos_threshold:.8f}")
    print(f"   Negative threshold (top {100-percentile_threshold}%): {neg_threshold:.8f}")

    # Variance stats
    print(f"\n📈 Variance Statistics:")
    print(f"   Subject-level variance (Positive): {list(pos_subj_var.values())}")
    print(f"   Subject-level variance (Negative): {list(neg_subj_var.values())}")
    print(f"   Mean sequence variance (Positive): {np.mean(pos_seq_var):.8e}")
    print(f"   Mean sequence variance (Negative): {np.mean(neg_seq_var):.8e}")

    # Create figure with 3 rows
    fig = plt.figure(figsize=(20, 18))

    # Row 1: Positive/Negative ortho views
    ax1 = plt.subplot(3, 2, 1)
    plotting.plot_stat_map(
        pos_mean_img,
        title=f"{emotion} - Average Positive Attribution | {baseline_type} baseline\nN={len(pos_maps)} sequences, {len(pos_subj_var)} subjects",
        display_mode='ortho',
        cmap='hot',
        threshold=pos_threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax1,
        vmax=pos_mean.max() if pos_mean.max() > 0 else 1e-8
    )

    ax2 = plt.subplot(3, 2, 2)
    neg_abs_mean = np.abs(neg_mean)
    neg_abs_img = nib.Nifti1Image(neg_abs_mean, affine)
    plotting.plot_stat_map(
        neg_abs_img,
        title=f"{emotion} - Average Negative Attribution (abs) | {baseline_type} baseline\nN={len(neg_maps)} sequences, {len(neg_subj_var)} subjects",
        display_mode='ortho',
        cmap='hot',
        threshold=neg_threshold,
        cut_coords=(0, 0, 0),
        colorbar=True,
        axes=ax2,
        vmax=neg_abs_mean.max() if neg_abs_mean.max() > 0 else 1e-8
    )

    # Row 2: Histograms
    ax3 = plt.subplot(3, 2, 3)
    if len(pos_values) > 0:
        ax3.hist(pos_values, bins=100, color='red', alpha=0.7)
        ax3.axvline(pos_threshold, color='black', linestyle='--', linewidth=2,
                   label=f'{100-percentile_threshold}% threshold')
        ax3.set_xlabel('IG values')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Positive IG values')
        ax3.legend()
        ax3.set_yscale('log')

    ax4 = plt.subplot(3, 2, 4)
    if len(neg_values) > 0:
        ax4.hist(neg_values, bins=100, color='blue', alpha=0.7)
        ax4.axvline(-neg_threshold, color='black', linestyle='--', linewidth=2,
                   label=f'{100-percentile_threshold}% threshold')
        ax4.set_xlabel('IG values')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Negative IG values')
        ax4.legend()
        ax4.set_yscale('log')

    # Row 3: Variance plots
    ax5 = plt.subplot(3, 2, 5)
    subjects_list = list(pos_subj_var.keys())
    pos_var_vals = [pos_subj_var[s] for s in subjects_list]
    neg_var_vals = [neg_subj_var[s] for s in subjects_list]

    x = np.arange(len(subjects_list))
    width = 0.35
    ax5.bar(x - width/2, pos_var_vals, width, label='Positive', color='red', alpha=0.7)
    ax5.bar(x + width/2, neg_var_vals, width, label='Negative', color='blue', alpha=0.7)
    ax5.set_xlabel('Subject')
    ax5.set_ylabel('Variance from mean')
    ax5.set_title('Subject-level Variance\n(variance between subject mean and overall mean)')
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.split('-')[1][:8] for s in subjects_list], rotation=45, ha='right')
    ax5.legend()
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 2, 6)
    ax6.hist([pos_seq_var, neg_seq_var], bins=20, label=['Positive', 'Negative'],
             color=['red', 'blue'], alpha=0.7)
    ax6.set_xlabel('Variance from mean')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Sequence-level Variance Distribution\n(N={len(pos_seq_var)} sequences)')
    ax6.legend()
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    out_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/analysis/4_IGmap/visualizations/emotion_average_{baseline_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{emotion}_average_N{len(pos_maps)}_baseline_{baseline_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {out_path}\n")

    plt.close()

    return pos_mean, neg_mean, pos_subj_var, neg_subj_var


def main():
    """Process all emotions"""

    baseline_type = 'first_10sec'  # or 'zeros'

    print(f"\n{'='*80}")
    print(f"Visualizing Emotion-averaged IG Maps")
    print(f"  Baseline: {baseline_type}")
    print(f"  Emotions: {emotion_labels}")
    print(f"{'='*80}\n")

    for emotion in emotion_labels:
        try:
            visualize_emotion_average(emotion, baseline_type=baseline_type)
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
