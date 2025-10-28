#!/usr/bin/env python3
"""
Evaluate best checkpoint vs last checkpoint
Seq 30 best checkpoint was at epoch 3, but we evaluated epoch 24 (last)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule


def evaluate_checkpoint(ckpt_path, device):
    """Evaluate a specific checkpoint"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {ckpt_path}")
    print(f"{'='*80}\n")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt['hyper_parameters']

    # Update paths
    args["num_workers"] = 4
    args['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args['default_root_dir'] = str(project_root / "output/moviefmri")
    args['eval_batch_size'] = 4

    # Add missing params
    for param, default in [
        ('stratified_params', None),
        ('dataset_split_seed', 777),
        ('train_split', 0.7),
        ('val_split', 0.15),
        ('bad_subj_path', None),
        ('shuffle_time_sequence', False),
        ('time_as_channel', False),
        ('limit_training_samples', 0),
    ]:
        if param not in args:
            args[param] = default

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

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    # Evaluate on test set
    test_loader = DataLoader(data_module.test_dataset, batch_size=4, shuffle=False, num_workers=4)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(test_loader)}", flush=True)

            fmri = batch['fmri_sequence'].to(device).float()
            targets = batch['target']

            # Get logits
            logits = model(fmri)

            # Softmax to probabilities
            probs = torch.softmax(logits, dim=-1)
            probs_pos = probs[..., 1].cpu().numpy()

            all_probs.append(probs_pos)
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Flatten
    all_probs_flat = all_probs.reshape(-1, 7)
    all_targets_flat = all_targets.reshape(-1, 7)

    print(f"\n✅ Collected {all_probs_flat.shape[0]} samples")

    # Calculate metrics per emotion
    results = {}
    for emo_idx, emo_name in enumerate(emotion_names):
        probs = all_probs_flat[:, emo_idx]
        targets = all_targets_flat[:, emo_idx]

        # Remove NaN
        valid_mask = ~np.isnan(targets) & ~np.isnan(probs)
        probs = probs[valid_mask]
        targets = targets[valid_mask]

        if len(targets) == 0 or len(np.unique(targets)) < 2:
            results[emo_name] = {'auroc': np.nan, 'acc_0.5': np.nan}
            continue

        # AUROC
        auroc = roc_auc_score(targets, probs)

        # Accuracy with threshold=0.5
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(targets, preds)

        results[emo_name] = {'auroc': auroc, 'acc_0.5': acc}

        print(f"{emo_name:<10}: AUROC={auroc:.4f}, Acc(thr=0.5)={acc:.4f}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Evaluate both checkpoints for seq 30
    run_id_seq30 = "gajr5p1p"

    best_ckpt = project_root / f"output/moviefmri/{run_id_seq30}/checkpt-epoch=03-valid_acc=0.72.ckpt"
    last_ckpt = project_root / f"output/moviefmri/{run_id_seq30}/last.ckpt"

    print("="*80)
    print("COMPARING BEST vs LAST CHECKPOINT (Seq 30)")
    print("="*80)

    results_best = evaluate_checkpoint(best_ckpt, device)
    results_last = evaluate_checkpoint(last_ckpt, device)

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Emotion':<10} {'Best (epoch 3)':<20} {'Last (epoch 24)':<20} {'Difference':<15}")
    print("-"*80)

    for emo_name in results_best.keys():
        auroc_best = results_best[emo_name]['auroc']
        auroc_last = results_last[emo_name]['auroc']

        if not np.isnan(auroc_best) and not np.isnan(auroc_last):
            diff = auroc_best - auroc_last
            status = "✅ Better" if diff > 0.01 else "⚠️ Worse" if diff < -0.01 else "≈ Same"
            print(f"{emo_name:<10} {auroc_best:.4f}              {auroc_last:.4f}              {diff:+.4f} {status}")
        else:
            print(f"{emo_name:<10} {auroc_best:.4f}              {auroc_last:.4f}              N/A")

    print("\n" + "="*80)
