#!/usr/bin/env python3
"""
Visualize classification predictions (seq 20 vs 30) vs ground truth
Reconstruct to 750 frames like regression visualization
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, '/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/src')

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule
import pytorch_lightning as pl

def load_model_and_predict(ckpt_path, sequence_length, output_dir):
    """Load checkpoint and run inference on test set"""

    print(f"\n{'='*80}")
    print(f"Loading model: {ckpt_path}")
    print(f"Sequence length: {sequence_length}")
    print(f"{'='*80}\n")

    # Setup data module with correct sequence length FIRST
    # Use same config as training
    from argparse import Namespace
    hparams = Namespace(
        image_path="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
        dataset_name="HBN",
        input_type="movieDM",
        downstream_task="emotions",
        downstream_task_type="classification",
        sequence_length=sequence_length,
        img_size=[96, 96, 96, sequence_length],
        stride_between_seq=1,
        stride_within_seq=1,
        batch_size=2,
        eval_batch_size=2,
        num_workers=4,
        train_split=0.7,
        val_split=0.15,
        dataset_split_num="",
        with_voxel_norm=False,
        adjust_hrf=True,
        input_offset=0,
        augment_during_training=False,
        num_targets=7,
        decoder='series_decoder'
    )

    data_module = fMRIDataModule(**vars(hparams))
    data_module.prepare_data()
    data_module.setup('test')

    # Load model with data_module
    model = LitClassifier.load_from_checkpoint(ckpt_path, data_module=data_module, strict=False)
    model.eval()
    model = model.cuda()

    test_loader = data_module.test_dataloader()

    # Collect predictions by subject
    predictions_by_subject = defaultdict(lambda: {'preds': [], 'targets': [], 'frames': []})

    print(f"Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}")

            fmri = batch['fmri_sequence'].cuda()
            targets = batch['target']  # (B, T, num_targets)
            subject_names = batch['subject_name']
            start_frames = batch.get('start_frame', [0] * len(subject_names))

            # Forward pass
            logits = model(fmri)  # (B, T, num_targets, num_classes)

            # Get probabilities (for class 1)
            probs = torch.softmax(logits, dim=-1)  # (B, T, num_targets, 2)
            probs_class1 = probs[..., 1].cpu().numpy()  # (B, T, num_targets)

            targets_np = targets.cpu().numpy()  # (B, T, num_targets)

            # Store by subject
            for i, (subj, start_frame) in enumerate(zip(subject_names, start_frames)):
                pred_seq = probs_class1[i]  # (T, num_targets)
                target_seq = targets_np[i]  # (T, num_targets)

                # Frame indices for this sequence
                frames = np.arange(start_frame, start_frame + sequence_length)

                predictions_by_subject[subj]['preds'].append(pred_seq)
                predictions_by_subject[subj]['targets'].append(target_seq)
                predictions_by_subject[subj]['frames'].append(frames)

    print(f"\nCollected predictions for {len(predictions_by_subject)} subjects")

    # Reconstruct to 750 frames for each subject
    reconstructed_data = {}

    for subj, data in predictions_by_subject.items():
        # Initialize arrays
        all_preds = np.full((750, 7), np.nan)
        all_targets = np.full((750, 7), np.nan)
        counts = np.zeros((750, 7))

        # Aggregate overlapping predictions (average)
        for pred_seq, target_seq, frame_indices in zip(data['preds'], data['targets'], data['frames']):
            for t, frame_idx in enumerate(frame_indices):
                if frame_idx < 750:
                    if np.isnan(all_preds[frame_idx, 0]):
                        all_preds[frame_idx] = pred_seq[t]
                        all_targets[frame_idx] = target_seq[t]
                        counts[frame_idx] = 1
                    else:
                        all_preds[frame_idx] += pred_seq[t]
                        counts[frame_idx] += 1

        # Average overlapping predictions
        for i in range(750):
            if counts[i, 0] > 1:
                all_preds[i] /= counts[i, 0]

        reconstructed_data[subj] = {
            'predictions': all_preds,
            'targets': all_targets,
            'coverage': np.sum(~np.isnan(all_preds[:, 0])) / 750
        }

    print(f"Reconstructed {len(reconstructed_data)} subjects")
    print(f"Average coverage: {np.mean([d['coverage'] for d in reconstructed_data.values()]):.1%}")

    # Save predictions
    output_path = Path(output_dir) / f"test_predictions_seq{sequence_length}.npz"
    np.savez(output_path, **{subj: data for subj, data in reconstructed_data.items()})
    print(f"\nSaved predictions to: {output_path}")

    return reconstructed_data


def compare_seq20_vs_seq30(pred_seq20, pred_seq30, output_dir):
    """Compare seq 20 vs seq 30 predictions visually"""

    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}\n")

    # Get common subjects
    common_subjects = set(pred_seq20.keys()) & set(pred_seq30.keys())
    print(f"Common subjects: {len(common_subjects)}")

    if len(common_subjects) == 0:
        print("ERROR: No common subjects found!")
        return

    # Calculate per-subject AUROC for each emotion
    from sklearn.metrics import roc_auc_score

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    auroc_seq20 = {emo: [] for emo in emotion_names}
    auroc_seq30 = {emo: [] for emo in emotion_names}
    subjects_list = []

    for subj in common_subjects:
        subjects_list.append(subj)

        for emo_idx, emo_name in enumerate(emotion_names):
            # Seq 20
            preds20 = pred_seq20[subj]['predictions'][:, emo_idx]
            targets20 = pred_seq20[subj]['targets'][:, emo_idx]
            mask20 = ~np.isnan(preds20) & ~np.isnan(targets20)

            if mask20.sum() > 0 and len(np.unique(targets20[mask20])) == 2:
                try:
                    auroc = roc_auc_score(targets20[mask20], preds20[mask20])
                    auroc_seq20[emo_name].append(auroc)
                except:
                    auroc_seq20[emo_name].append(np.nan)
            else:
                auroc_seq20[emo_name].append(np.nan)

            # Seq 30
            preds30 = pred_seq30[subj]['predictions'][:, emo_idx]
            targets30 = pred_seq30[subj]['targets'][:, emo_idx]
            mask30 = ~np.isnan(preds30) & ~np.isnan(targets30)

            if mask30.sum() > 0 and len(np.unique(targets30[mask30])) == 2:
                try:
                    auroc = roc_auc_score(targets30[mask30], preds30[mask30])
                    auroc_seq30[emo_name].append(auroc)
                except:
                    auroc_seq30[emo_name].append(np.nan)
            else:
                auroc_seq30[emo_name].append(np.nan)

    # Plot best and worst subjects for each emotion
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for emo_idx, emo_name in enumerate(emotion_names):
        print(f"\nProcessing {emo_name}...")

        # Skip if too few valid subjects
        valid_aurocs = [a for a in auroc_seq20[emo_name] if not np.isnan(a)]
        if len(valid_aurocs) < 3:
            print(f"  Skipping {emo_name} (too few valid subjects)")
            continue

        # Find best subject (highest seq20 AUROC)
        valid_indices = [i for i, a in enumerate(auroc_seq20[emo_name]) if not np.isnan(a)]
        best_idx = valid_indices[np.argmax([auroc_seq20[emo_name][i] for i in valid_indices])]
        best_subj = subjects_list[best_idx]

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        fig.suptitle(f'{emo_name}: Seq 20 vs Seq 30 Predictions (Best Subject: {best_subj})',
                     fontsize=16, fontweight='bold')

        # Get data for best subject
        preds20 = pred_seq20[best_subj]['predictions'][:, emo_idx]
        preds30 = pred_seq30[best_subj]['predictions'][:, emo_idx]
        targets = pred_seq20[best_subj]['targets'][:, emo_idx]

        # Plot 1: Ground truth
        axes[0].fill_between(range(750), targets, alpha=0.6, color='black', label='Ground Truth')
        axes[0].set_ylabel('Binary Label', fontsize=12)
        axes[0].set_title(f'Ground Truth (Sum={np.nansum(targets):.0f} positive frames)', fontsize=12)
        axes[0].set_ylim([-0.1, 1.1])
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Seq 20 predictions
        axes[1].plot(range(750), preds20, color='#2ca02c', linewidth=2, alpha=0.8, label='Seq 20')
        axes[1].fill_between(range(750), preds20, alpha=0.3, color='#2ca02c')
        axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold=0.5')
        axes[1].set_ylabel('P(class=1)', fontsize=12)
        axes[1].set_title(f'Seq 20 Predictions (AUROC={auroc_seq20[emo_name][best_idx]:.4f})', fontsize=12)
        axes[1].set_ylim([-0.05, 1.05])
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Seq 30 predictions
        axes[2].plot(range(750), preds30, color='#d62728', linewidth=2, alpha=0.8, label='Seq 30')
        axes[2].fill_between(range(750), preds30, alpha=0.3, color='#d62728')
        axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold=0.5')
        axes[2].set_ylabel('P(class=1)', fontsize=12)
        axes[2].set_title(f'Seq 30 Predictions (AUROC={auroc_seq30[emo_name][best_idx]:.4f})', fontsize=12)
        axes[2].set_ylim([-0.05, 1.05])
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Direct comparison
        axes[3].plot(range(750), preds20, color='#2ca02c', linewidth=2, alpha=0.7, label=f'Seq 20 (AUROC={auroc_seq20[emo_name][best_idx]:.3f})')
        axes[3].plot(range(750), preds30, color='#d62728', linewidth=2, alpha=0.7, label=f'Seq 30 (AUROC={auroc_seq30[emo_name][best_idx]:.3f})')
        axes[3].fill_between(range(750), targets * 1.1, alpha=0.2, color='black', label='Ground Truth (scaled)')
        axes[3].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Threshold')
        axes[3].set_xlabel('Frame', fontsize=12)
        axes[3].set_ylabel('P(class=1)', fontsize=12)
        axes[3].set_title('Seq 20 vs Seq 30 Overlay', fontsize=12)
        axes[3].set_ylim([-0.05, 1.15])
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        plot_path = output_path / f"classification_comparison_{emo_name}_best.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")

    # Summary statistics plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for emo_idx, emo_name in enumerate(emotion_names):
        ax = axes[emo_idx]

        valid_seq20 = [a for a in auroc_seq20[emo_name] if not np.isnan(a)]
        valid_seq30 = [a for a in auroc_seq30[emo_name] if not np.isnan(a)]

        if len(valid_seq20) > 0 and len(valid_seq30) > 0:
            x = np.arange(min(len(valid_seq20), len(valid_seq30)))
            ax.scatter(x, sorted(valid_seq20)[:len(x)], color='#2ca02c', alpha=0.6, s=50, label='Seq 20')
            ax.scatter(x, sorted(valid_seq30)[:len(x)], color='#d62728', alpha=0.6, s=50, label='Seq 30')

            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
            ax.set_xlabel('Subject (sorted by AUROC)', fontsize=10)
            ax.set_ylabel('AUROC', fontsize=10)
            ax.set_title(f'{emo_name}\nSeq20: {np.mean(valid_seq20):.3f} | Seq30: {np.mean(valid_seq30):.3f}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        else:
            ax.text(0.5, 0.5, f'{emo_name}\nInsufficient data',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

    # Remove extra subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    summary_path = output_path / "classification_auroc_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    output_dir = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/classification_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seq 20
    ckpt_seq20 = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/mc3r4vhf/last.ckpt"
    pred_seq20_path = output_dir / "test_predictions_seq20.npz"

    if not pred_seq20_path.exists():
        print("Running inference for seq 20...")
        pred_seq20 = load_model_and_predict(ckpt_seq20, 20, output_dir)
    else:
        print(f"Loading cached seq 20 predictions from {pred_seq20_path}")
        data = np.load(pred_seq20_path, allow_pickle=True)
        pred_seq20 = {key: data[key].item() for key in data.files}

    # Seq 30
    ckpt_seq30 = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/gajr5p1p/last.ckpt"
    pred_seq30_path = output_dir / "test_predictions_seq30.npz"

    if not pred_seq30_path.exists():
        print("\nRunning inference for seq 30...")
        pred_seq30 = load_model_and_predict(ckpt_seq30, 30, output_dir)
    else:
        print(f"Loading cached seq 30 predictions from {pred_seq30_path}")
        data = np.load(pred_seq30_path, allow_pickle=True)
        pred_seq30 = {key: data[key].item() for key in data.files}

    # Compare
    compare_seq20_vs_seq30(pred_seq20, pred_seq30, output_dir)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
