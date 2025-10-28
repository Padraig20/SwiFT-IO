#!/usr/bin/env python3
"""
Compare classification predictions: Seq 20 vs Seq 30
Visualize predictions vs ground truth over 750 frames
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

def load_and_predict(run_id, seq_length, device):
    """Load model and generate predictions for test set"""
    print(f"\n{'='*80}")
    print(f"Processing {run_id} (seq {seq_length})")
    print(f"{'='*80}\n")

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{run_id}/last.ckpt"
    print(f"Loading: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt['hyper_parameters']

    # Update paths
    args["num_workers"] = 4
    args['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args['default_root_dir'] = str(project_root / "output/moviefmri")
    args['eval_batch_size'] = 4

    # Add missing params
    if 'stratified_params' not in args:
        args['stratified_params'] = None
    if 'dataset_split_seed' not in args:
        args['dataset_split_seed'] = args.get('seed', 777)
    if 'train_split' not in args:
        args['train_split'] = 0.7
    if 'val_split' not in args:
        args['val_split'] = 0.15
    if 'bad_subj_path' not in args:
        args['bad_subj_path'] = None
    if 'shuffle_time_sequence' not in args:
        args['shuffle_time_sequence'] = False
    if 'time_as_channel' not in args:
        args['time_as_channel'] = False
    if 'limit_training_samples' not in args:
        args['limit_training_samples'] = 0

    # Initialize data module
    data_module = fMRIDataModule(**args)
    data_module.prepare_data()
    data_module.setup(stage='test')

    # Load model
    model = LitClassifier(data_module=data_module, **args)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)

    print("✅ Model & data loaded")

    # Get test dataset
    testset = data_module.test_dataset
    test_subjects = sorted(list(set([str(x[1]) for x in testset.data])))
    print(f"Found {len(test_subjects)} test subjects")

    # Collect predictions by subject
    subject_predictions = {}

    for subj_idx, subject in enumerate(test_subjects[:10], 1):  # First 10 subjects for testing
        print(f"[{subj_idx}/10] {subject}...", flush=True, end="")

        # Get subject data with start_frames
        subj_data = [
            (idx, data_tuple) for idx, data_tuple in enumerate(testset.data)
            if str(data_tuple[1]) == subject
        ]

        # Extract start_frames from data tuples
        # data_tuple: (idx, subject_name, subject_path, start_frame, duration, num_frames, target, sex)
        subj_indices = [item[0] for item in subj_data]
        actual_start_frames = [item[1][3] for item in subj_data]  # start_frame is at index 3

        test_loader = DataLoader(Subset(testset, subj_indices),
                                batch_size=4, shuffle=False, num_workers=4)

        # Collect sequences
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri_sequence'].to(device).float()  # Ensure float32
                targets = batch['target']  # (B, T, num_targets)

                # Get logits
                logits = model(fmri)  # (B, T, num_targets, num_classes)

                # Softmax to probabilities
                probs = torch.softmax(logits, dim=-1)  # (B, T, num_targets, 2)
                probs_pos = probs[..., 1].cpu().numpy()  # (B, T, num_targets) - prob of class 1

                all_probs.append(probs_pos)
                all_targets.append(targets.cpu().numpy())

        # Concatenate
        all_probs = np.concatenate(all_probs, axis=0)  # (N, T, 7)
        all_targets = np.concatenate(all_targets, axis=0)  # (N, T, 7)

        # Reconstruct to 750 frames using actual start_frames
        reconstructed_preds = np.full((750, 7), np.nan)
        reconstructed_targets = np.full((750, 7), np.nan)
        counts = np.zeros((750, 7))

        for seq_idx, start_frame in enumerate(actual_start_frames):
            for t in range(seq_length):
                frame_idx = int(start_frame) + t
                if frame_idx < 750:
                    if np.isnan(reconstructed_preds[frame_idx, 0]):
                        reconstructed_preds[frame_idx] = all_probs[seq_idx, t]
                        reconstructed_targets[frame_idx] = all_targets[seq_idx, t]
                        counts[frame_idx] = 1
                    else:
                        reconstructed_preds[frame_idx] += all_probs[seq_idx, t]
                        counts[frame_idx] += 1

        # Average overlapping predictions
        for i in range(750):
            if counts[i, 0] > 1:
                reconstructed_preds[i] /= counts[i, 0]

        subject_predictions[subject] = {
            'predictions': reconstructed_preds,
            'targets': reconstructed_targets,
            'coverage': np.sum(~np.isnan(reconstructed_preds[:, 0])) / 750
        }

        print(f" coverage={subject_predictions[subject]['coverage']:.1%}")

    print(f"\n✅ Collected predictions for {len(subject_predictions)} subjects\n")
    return subject_predictions


def plot_comparison(pred_seq20, pred_seq30, output_dir, run_id_seq20, run_id_seq30):
    """Generate comparison plots"""
    print("Generating plots...\n")

    # Save to analysis/plots folder
    output_dir = project_root / "analysis" / "plots" / "classification_seq20_vs_seq30"
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    common_subjects = list(set(pred_seq20.keys()) & set(pred_seq30.keys()))

    print(f"Common subjects: {len(common_subjects)}")

    # Try to load optimal thresholds
    optimal_thresholds_path = project_root / "analysis" / "optimal_thresholds.json"
    optimal_thresholds = None
    if optimal_thresholds_path.exists():
        import json
        with open(optimal_thresholds_path, 'r') as f:
            optimal_thresholds = json.load(f)
        print(f"✅ Loaded optimal thresholds from {optimal_thresholds_path}")
    else:
        print(f"⚠️  No optimal thresholds found at {optimal_thresholds_path}")

    # Calculate AUROC for each subject/emotion
    auroc_seq20 = {emo: [] for emo in emotion_names}
    auroc_seq30 = {emo: [] for emo in emotion_names}

    for subject in common_subjects:
        for emo_idx, emo_name in enumerate(emotion_names):
            # Seq 20
            preds20 = pred_seq20[subject]['predictions'][:, emo_idx]
            targets20 = pred_seq20[subject]['targets'][:, emo_idx]
            mask20 = ~np.isnan(preds20) & ~np.isnan(targets20)

            if mask20.sum() > 10 and len(np.unique(targets20[mask20])) == 2:
                try:
                    auroc = roc_auc_score(targets20[mask20], preds20[mask20])
                    auroc_seq20[emo_name].append(auroc)
                except:
                    auroc_seq20[emo_name].append(np.nan)
            else:
                auroc_seq20[emo_name].append(np.nan)

            # Seq 30
            preds30 = pred_seq30[subject]['predictions'][:, emo_idx]
            targets30 = pred_seq30[subject]['targets'][:, emo_idx]
            mask30 = ~np.isnan(preds30) & ~np.isnan(targets30)

            if mask30.sum() > 10 and len(np.unique(targets30[mask30])) == 2:
                try:
                    auroc = roc_auc_score(targets30[mask30], preds30[mask30])
                    auroc_seq30[emo_name].append(auroc)
                except:
                    auroc_seq30[emo_name].append(np.nan)
            else:
                auroc_seq30[emo_name].append(np.nan)

    # Plot for emotions with sufficient data
    for emo_idx, emo_name in enumerate(emotion_names):
        valid_aurocs20 = [a for a in auroc_seq20[emo_name] if not np.isnan(a)]
        valid_aurocs30 = [a for a in auroc_seq30[emo_name] if not np.isnan(a)]

        if len(valid_aurocs20) < 3:
            print(f"  Skipping {emo_name} (insufficient data)")
            continue

        print(f"  Plotting {emo_name}...")

        # Find best subject (highest seq20 AUROC)
        valid_indices = [i for i, a in enumerate(auroc_seq20[emo_name]) if not np.isnan(a)]
        best_idx = valid_indices[np.argmax([auroc_seq20[emo_name][i] for i in valid_indices])]
        best_subject = common_subjects[best_idx]

        # Create plot with 750 frames (5 subplots: GT, Seq20, Seq30, Overlay, Heatmap)
        fig, axes = plt.subplots(5, 1, figsize=(24, 15), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.4]})
        fig.suptitle(f'{emo_name}: Seq 20 ({run_id_seq20}) vs Seq 30 ({run_id_seq30}) | Best: {best_subject}',
                     fontsize=16, fontweight='bold')

        preds20 = pred_seq20[best_subject]['predictions'][:, emo_idx]
        preds30 = pred_seq30[best_subject]['predictions'][:, emo_idx]
        targets = pred_seq20[best_subject]['targets'][:, emo_idx]

        frames = np.arange(750)

        # Ground truth
        # Use step plot for binary data
        valid_mask = ~np.isnan(targets)
        axes[0].fill_between(frames[valid_mask], targets[valid_mask], alpha=0.6, color='black', step='mid')
        axes[0].set_ylabel('Binary Label', fontsize=12)
        axes[0].set_title(f'Ground Truth (n={int(np.nansum(targets))} positive frames)', fontsize=12)
        axes[0].set_ylim([-0.1, 1.1])
        axes[0].set_xlim([0, 750])
        axes[0].grid(True, alpha=0.3)

        # Seq 20
        valid20 = ~np.isnan(preds20)
        axes[1].plot(frames[valid20], preds20[valid20], color='#2ca02c', linewidth=1.5, alpha=0.8)
        axes[1].fill_between(frames[valid20], preds20[valid20], alpha=0.3, color='#2ca02c')
        axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold=0.5')

        # Add optimal threshold line if available
        if optimal_thresholds and run_id_seq20 in optimal_thresholds:
            opt_thr_data = optimal_thresholds[run_id_seq20].get('optimal_thresholds_from_validation', {})
            if emo_name in opt_thr_data:
                opt_thr = opt_thr_data[emo_name]['threshold']
                axes[1].axhline(y=opt_thr, color='orange', linestyle='-.', linewidth=2, alpha=0.8,
                              label=f'Optimal={opt_thr:.3f}')

        axes[1].set_ylabel('P(class=1)', fontsize=12)
        axes[1].set_title(f'Seq 20 Predictions (AUROC={auroc_seq20[emo_name][best_idx]:.4f}, coverage={valid20.sum()/750:.1%})', fontsize=12)
        axes[1].set_ylim([-0.05, 1.05])
        axes[1].set_xlim([0, 750])
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Seq 30
        valid30 = ~np.isnan(preds30)
        axes[2].plot(frames[valid30], preds30[valid30], color='#d62728', linewidth=1.5, alpha=0.8)
        axes[2].fill_between(frames[valid30], preds30[valid30], alpha=0.3, color='#d62728')
        axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold=0.5')

        # Add optimal threshold line if available
        if optimal_thresholds and run_id_seq30 in optimal_thresholds:
            opt_thr_data = optimal_thresholds[run_id_seq30].get('optimal_thresholds_from_validation', {})
            if emo_name in opt_thr_data:
                opt_thr = opt_thr_data[emo_name]['threshold']
                axes[2].axhline(y=opt_thr, color='orange', linestyle='-.', linewidth=2, alpha=0.8,
                              label=f'Optimal={opt_thr:.3f}')

        axes[2].set_ylabel('P(class=1)', fontsize=12)
        axes[2].set_title(f'Seq 30 Predictions (AUROC={auroc_seq30[emo_name][best_idx]:.4f}, coverage={valid30.sum()/750:.1%})', fontsize=12)
        axes[2].set_ylim([-0.05, 1.05])
        axes[2].set_xlim([0, 750])
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # Overlay
        if valid20.any():
            axes[3].plot(frames[valid20], preds20[valid20], color='#2ca02c', linewidth=1.5, alpha=0.7,
                        label=f'Seq 20 (AUROC={auroc_seq20[emo_name][best_idx]:.3f})')
        if valid30.any():
            axes[3].plot(frames[valid30], preds30[valid30], color='#d62728', linewidth=1.5, alpha=0.7,
                        label=f'Seq 30 (AUROC={auroc_seq30[emo_name][best_idx]:.3f})')
        if valid_mask.any():
            axes[3].fill_between(frames[valid_mask], targets[valid_mask] * 1.1, alpha=0.2, color='black',
                               label='Ground Truth (scaled)', step='mid')
        axes[3].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold=0.5')

        # Add optimal threshold lines if available
        if optimal_thresholds:
            if run_id_seq20 in optimal_thresholds:
                opt_thr_data = optimal_thresholds[run_id_seq20].get('optimal_thresholds_from_validation', {})
                if emo_name in opt_thr_data:
                    opt_thr = opt_thr_data[emo_name]['threshold']
                    axes[3].axhline(y=opt_thr, color='#2ca02c', linestyle='-.', linewidth=2, alpha=0.7,
                                  label=f'Seq20 Opt={opt_thr:.3f}')

            if run_id_seq30 in optimal_thresholds:
                opt_thr_data = optimal_thresholds[run_id_seq30].get('optimal_thresholds_from_validation', {})
                if emo_name in opt_thr_data:
                    opt_thr = opt_thr_data[emo_name]['threshold']
                    axes[3].axhline(y=opt_thr, color='#d62728', linestyle='-.', linewidth=2, alpha=0.7,
                                  label=f'Seq30 Opt={opt_thr:.3f}')

        axes[3].set_ylabel('P(class=1)', fontsize=12)
        axes[3].set_title('Seq 20 vs Seq 30 Overlay', fontsize=12)
        axes[3].set_ylim([-0.05, 1.15])
        axes[3].set_xlim([0, 750])
        axes[3].legend(loc='upper right', fontsize=9)
        axes[3].grid(True, alpha=0.3)

        # ========================================
        # Heatmap: Binary Correctness (Correct=1, Incorrect=0)
        # ========================================

        # Get optimal thresholds
        opt_thr_20 = 0.5  # Default
        opt_thr_30 = 0.5  # Default
        if optimal_thresholds:
            if run_id_seq20 in optimal_thresholds:
                opt_thr_data = optimal_thresholds[run_id_seq20].get('optimal_thresholds_from_validation', {})
                if emo_name in opt_thr_data:
                    opt_thr_20 = opt_thr_data[emo_name]['threshold']
            if run_id_seq30 in optimal_thresholds:
                opt_thr_data = optimal_thresholds[run_id_seq30].get('optimal_thresholds_from_validation', {})
                if emo_name in opt_thr_data:
                    opt_thr_30 = opt_thr_data[emo_name]['threshold']

        # Create binary predictions using optimal thresholds
        binary_preds20 = np.full(750, np.nan)
        binary_preds30 = np.full(750, np.nan)

        valid20_mask = ~np.isnan(preds20)
        valid30_mask = ~np.isnan(preds30)

        binary_preds20[valid20_mask] = (preds20[valid20_mask] >= opt_thr_20).astype(float)
        binary_preds30[valid30_mask] = (preds30[valid30_mask] >= opt_thr_30).astype(float)

        # Calculate correctness (1=correct, 0=incorrect, NaN=no data)
        correctness_seq20 = np.full(750, np.nan)
        correctness_seq30 = np.full(750, np.nan)

        # Only calculate where both prediction and target are valid
        valid_both_20 = valid20_mask & valid_mask
        valid_both_30 = valid30_mask & valid_mask

        correctness_seq20[valid_both_20] = (binary_preds20[valid_both_20] == targets[valid_both_20]).astype(float)
        correctness_seq30[valid_both_30] = (binary_preds30[valid_both_30] == targets[valid_both_30]).astype(float)

        # Create heatmap data: 2 rows (seq20, seq30) x 750 columns (frames)
        heatmap_data = np.array([correctness_seq20, correctness_seq30])

        # Plot heatmap
        import matplotlib.colors as mcolors

        # Custom colormap: incorrect=red, correct=green, no_data=white
        colors = ['#d62728', '#2ca02c', 'white']  # red, green, white
        n_bins = 3
        cmap = mcolors.ListedColormap(colors[:2])  # Only red and green for 0 and 1
        cmap.set_bad(color='white')  # NaN values will be white

        im = axes[4].imshow(heatmap_data, cmap=cmap, aspect='auto',
                           interpolation='nearest', vmin=0, vmax=1)

        axes[4].set_yticks([0, 1])
        axes[4].set_yticklabels([f'Seq 20\n(thr={opt_thr_20:.3f})',
                                f'Seq 30\n(thr={opt_thr_30:.3f})'], fontsize=11)
        axes[4].set_xlabel('Frame (1.3s TR)', fontsize=12)
        axes[4].set_xlim([0, 750])

        # Calculate accuracy for each model
        acc_seq20 = np.nanmean(correctness_seq20) * 100
        acc_seq30 = np.nanmean(correctness_seq30) * 100

        axes[4].set_title(f'Binary Correctness (Green=Correct, Red=Incorrect, White=No Data) | '
                         f'Seq20 Acc={acc_seq20:.1f}%, Seq30 Acc={acc_seq30:.1f}%',
                         fontsize=12, pad=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[4], orientation='vertical', pad=0.01, aspect=10)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Incorrect', 'Correct'])

        plt.tight_layout()
        plot_path = output_dir / f"clf_{run_id_seq20}_vs_{run_id_seq30}_{emo_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Saved plots to {output_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Model IDs
    run_id_seq20 = "mc3r4vhf"
    run_id_seq30 = "gajr5p1p"

    output_dir = project_root / "output/classification_comparison"

    # Run inference
    print("=" * 80)
    print(f"STEP 1: Seq 20 ({run_id_seq20})")
    print("=" * 80)
    pred_seq20 = load_and_predict(run_id_seq20, 20, device)

    print("\n" + "=" * 80)
    print(f"STEP 2: Seq 30 ({run_id_seq30})")
    print("=" * 80)
    pred_seq30 = load_and_predict(run_id_seq30, 30, device)

    print("\n" + "=" * 80)
    print("STEP 3: Generate Plots")
    print("=" * 80)
    plot_comparison(pred_seq20, pred_seq30, output_dir, run_id_seq20, run_id_seq30)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
