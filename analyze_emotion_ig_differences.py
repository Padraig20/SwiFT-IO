#!/usr/bin/env python3
"""
Analyze differences between emotions' IG maps to verify
if the model learned distinct representations for each emotion
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def load_emotion_average_ig(emotion, baseline_type='first_10sec', project_root='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO'):
    """Load and average all IG maps for one emotion"""
    project_root = Path(project_root)
    base_dir = project_root / f"analysis/4_IGmap/baseline_{baseline_type}_selective/opr6oq97/nii_segments"

    json_path = project_root / "analysis/4_IGmap/subject_performance/opr6oq97_emotion_specific_mse.json"
    with open(json_path) as f:
        data = json.load(f)

    emotion_idx = emotion_labels.index(emotion)
    subjects = data['top5_per_emotion'][emotion]

    pos_arrays = []
    neg_arrays = []

    for subject in subjects:
        subject_dir = base_dir / subject / f"target{emotion_idx}_{emotion}"
        if not subject_dir.exists():
            continue

        pos_files = sorted(subject_dir.glob(f"{subject}_{emotion}_TR*_rank*_AVGpred_positive.nii.gz"))

        for pos_path in pos_files:
            neg_path = pos_path.parent / pos_path.name.replace('positive', 'negative')
            if not neg_path.exists():
                continue

            pos_data = nib.load(str(pos_path)).get_fdata()
            neg_data = nib.load(str(neg_path)).get_fdata()

            pos_arrays.append(pos_data)
            neg_arrays.append(neg_data)

    if len(pos_arrays) == 0:
        return None, None

    pos_mean = np.mean(pos_arrays, axis=0)
    neg_mean = np.mean(neg_arrays, axis=0)

    return pos_mean, neg_mean


def compute_similarity_matrix(emotion_maps, map_type='positive'):
    """
    Compute pairwise similarity between emotions

    Returns:
        correlation_matrix: Pearson correlation
        cosine_similarity_matrix: Cosine similarity
    """
    n_emotions = len(emotion_maps)
    correlation_matrix = np.zeros((n_emotions, n_emotions))
    cosine_sim_matrix = np.zeros((n_emotions, n_emotions))

    for i in range(n_emotions):
        for j in range(n_emotions):
            map_i = emotion_maps[i].flatten()
            map_j = emotion_maps[j].flatten()

            # Pearson correlation
            corr, _ = pearsonr(map_i, map_j)
            correlation_matrix[i, j] = corr

            # Cosine similarity
            cosine_sim = 1 - cosine(map_i, map_j)
            cosine_sim_matrix[i, j] = cosine_sim

    return correlation_matrix, cosine_sim_matrix


def find_emotion_specific_voxels(emotion_maps, percentile=95):
    """
    Find voxels that are specifically active for each emotion

    Returns:
        specific_voxels: dict mapping emotion_idx to voxel coordinates
        specificity_scores: how unique each voxel is to that emotion
    """
    n_emotions = len(emotion_maps)
    shape = emotion_maps[0].shape

    specific_voxels = {}
    specificity_maps = []

    for i, emotion in enumerate(emotion_labels):
        this_map = emotion_maps[i]
        other_maps = [emotion_maps[j] for j in range(n_emotions) if j != i]
        other_mean = np.mean(other_maps, axis=0)

        # Specificity: this emotion's activation - average of other emotions
        specificity = this_map - other_mean
        specificity_maps.append(specificity)

        # Find voxels with high specificity
        threshold = np.percentile(specificity[specificity > 0], percentile) if np.any(specificity > 0) else 0
        specific_coords = np.where(specificity >= threshold)

        specific_voxels[emotion] = {
            'n_voxels': len(specific_coords[0]),
            'threshold': threshold,
            'max_specificity': specificity.max(),
            'mean_specificity': specificity[specificity > 0].mean() if np.any(specificity > 0) else 0
        }

        print(f"{emotion:10s}: {len(specific_coords[0]):5d} specific voxels (threshold={threshold:.8f})")

    return specific_voxels, specificity_maps


def visualize_emotion_differences(emotion_pos_maps, emotion_neg_maps, baseline_type='first_10sec'):
    """Create comprehensive visualization of emotion differences"""

    # Compute similarity matrices
    print("\nComputing similarity matrices...")
    pos_corr, pos_cosine = compute_similarity_matrix(emotion_pos_maps, 'positive')
    neg_corr, neg_cosine = compute_similarity_matrix(emotion_neg_maps, 'negative')

    # Net maps (positive + negative)
    net_maps = [pos + neg for pos, neg in zip(emotion_pos_maps, emotion_neg_maps)]
    net_corr, net_cosine = compute_similarity_matrix(net_maps, 'net')

    # Find emotion-specific voxels
    print("\nFinding emotion-specific voxels (positive attribution)...")
    specific_voxels_pos, specificity_maps_pos = find_emotion_specific_voxels(emotion_pos_maps)

    print("\nFinding emotion-specific voxels (negative attribution)...")
    specific_voxels_neg, specificity_maps_neg = find_emotion_specific_voxels([np.abs(m) for m in emotion_neg_maps])

    # Create visualization
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Correlation matrices
    ax1 = plt.subplot(2, 4, 1)
    sns.heatmap(pos_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Positive Attribution\nPearson Correlation')

    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(neg_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Negative Attribution\nPearson Correlation')

    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(net_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=-1, vmax=1, ax=ax3, cbar_kws={'label': 'Correlation'})
    ax3.set_title('Net Attribution (Pos + Neg)\nPearson Correlation')

    # Compute average correlation (excluding diagonal)
    mask = ~np.eye(len(emotion_labels), dtype=bool)
    avg_pos_corr = pos_corr[mask].mean()
    avg_neg_corr = neg_corr[mask].mean()
    avg_net_corr = net_corr[mask].mean()

    ax4 = plt.subplot(2, 4, 4)
    ax4.bar(['Positive', 'Negative', 'Net'],
            [avg_pos_corr, avg_neg_corr, avg_net_corr],
            color=['red', 'blue', 'purple'], alpha=0.7)
    ax4.set_ylabel('Average Correlation')
    ax4.set_title('Average Inter-Emotion Correlation\n(excluding diagonal)')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_ylim([-1, 1])
    ax4.grid(True, alpha=0.3)

    # Add text with values
    for i, (val, label) in enumerate(zip([avg_pos_corr, avg_neg_corr, avg_net_corr],
                                          ['Positive', 'Negative', 'Net'])):
        ax4.text(i, val + 0.05 if val > 0 else val - 0.05, f'{val:.3f}',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    # Row 2: Cosine similarity matrices
    ax5 = plt.subplot(2, 4, 5)
    sns.heatmap(pos_cosine, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=0, vmax=1, ax=ax5, cbar_kws={'label': 'Cosine Similarity'})
    ax5.set_title('Positive Attribution\nCosine Similarity')

    ax6 = plt.subplot(2, 4, 6)
    sns.heatmap(neg_cosine, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=0, vmax=1, ax=ax6, cbar_kws={'label': 'Cosine Similarity'})
    ax6.set_title('Negative Attribution\nCosine Similarity')

    ax7 = plt.subplot(2, 4, 7)
    sns.heatmap(net_cosine, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                vmin=0, vmax=1, ax=ax7, cbar_kws={'label': 'Cosine Similarity'})
    ax7.set_title('Net Attribution\nCosine Similarity')

    # Average cosine similarity
    avg_pos_cosine = pos_cosine[mask].mean()
    avg_neg_cosine = neg_cosine[mask].mean()
    avg_net_cosine = net_cosine[mask].mean()

    ax8 = plt.subplot(2, 4, 8)
    ax8.bar(['Positive', 'Negative', 'Net'],
            [avg_pos_cosine, avg_neg_cosine, avg_net_cosine],
            color=['red', 'blue', 'purple'], alpha=0.7)
    ax8.set_ylabel('Average Cosine Similarity')
    ax8.set_title('Average Inter-Emotion Cosine Similarity\n(excluding diagonal)')
    ax8.set_ylim([0, 1])
    ax8.grid(True, alpha=0.3)

    for i, (val, label) in enumerate(zip([avg_pos_cosine, avg_neg_cosine, avg_net_cosine],
                                          ['Positive', 'Negative', 'Net'])):
        ax8.text(i, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    out_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/analysis/4_IGmap/visualizations/emotion_differences_{baseline_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"emotion_similarity_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {out_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("EMOTION SIMILARITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nAverage inter-emotion correlation:")
    print(f"  Positive attribution: {avg_pos_corr:.3f}")
    print(f"  Negative attribution: {avg_neg_corr:.3f}")
    print(f"  Net attribution:      {avg_net_corr:.3f}")
    print(f"\nAverage inter-emotion cosine similarity:")
    print(f"  Positive attribution: {avg_pos_cosine:.3f}")
    print(f"  Negative attribution: {avg_neg_cosine:.3f}")
    print(f"  Net attribution:      {avg_net_cosine:.3f}")

    print(f"\nInterpretation:")
    if avg_net_corr > 0.8:
        print("  ⚠️  HIGH SIMILARITY (>0.8): Emotions show very similar IG patterns")
        print("     → Model may not be learning distinct emotion representations")
    elif avg_net_corr > 0.5:
        print("  ⚡ MODERATE SIMILARITY (0.5-0.8): Some shared patterns, some differences")
        print("     → Model has partial emotion-specific representations")
    else:
        print("  ✅ LOW SIMILARITY (<0.5): Emotions show distinct IG patterns")
        print("     → Model learned distinct emotion representations")

    print("\nEmotion-specific voxels (positive attribution, top 5%):")
    for emotion in emotion_labels:
        n_voxels = specific_voxels_pos[emotion]['n_voxels']
        max_spec = specific_voxels_pos[emotion]['max_specificity']
        print(f"  {emotion:10s}: {n_voxels:5d} voxels (max specificity: {max_spec:.8f})")

    print("="*70)

    return {
        'pos_corr': pos_corr,
        'neg_corr': neg_corr,
        'net_corr': net_corr,
        'pos_cosine': pos_cosine,
        'neg_cosine': neg_cosine,
        'net_cosine': net_cosine,
        'specific_voxels_pos': specific_voxels_pos,
        'specific_voxels_neg': specific_voxels_neg,
        'specificity_maps_pos': specificity_maps_pos,
        'specificity_maps_neg': specificity_maps_neg
    }


def main():
    baseline_type = 'first_10sec'

    print(f"\n{'='*70}")
    print(f"Analyzing Emotion IG Map Differences")
    print(f"  Baseline: {baseline_type}")
    print(f"{'='*70}\n")

    # Load all emotion maps
    emotion_pos_maps = []
    emotion_neg_maps = []

    for emotion in emotion_labels:
        print(f"Loading {emotion}...")
        pos_mean, neg_mean = load_emotion_average_ig(emotion, baseline_type)

        if pos_mean is None:
            print(f"  ❌ Failed to load {emotion}")
            continue

        emotion_pos_maps.append(pos_mean)
        emotion_neg_maps.append(neg_mean)
        print(f"  ✅ Loaded {emotion}")

    if len(emotion_pos_maps) != len(emotion_labels):
        print(f"❌ Could not load all emotions")
        return

    # Analyze differences
    results = visualize_emotion_differences(emotion_pos_maps, emotion_neg_maps, baseline_type)

    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
