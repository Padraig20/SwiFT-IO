#!/usr/bin/env python3
"""
Find optimal classification thresholds using Youden Index
No retraining needed - just inference on existing checkpoints
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import json

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule


def find_optimal_threshold_youden(targets, probabilities):
    """
    Find optimal threshold using Youden Index
    J = Sensitivity + Specificity - 1 = TPR - FPR

    Args:
        targets: Binary labels (0 or 1)
        probabilities: Predicted probabilities for class 1

    Returns:
        optimal_threshold, youden_index, metrics_at_optimal
    """
    # Check if we have both classes
    unique_classes = np.unique(targets)
    if len(unique_classes) < 2:
        print(f"  Warning: Only one class present ({unique_classes})")
        return 0.5, 0.0, {}

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, probabilities)

    # Youden index = TPR - FPR
    j_scores = tpr - fpr

    # Find optimal threshold
    optimal_idx = j_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]
    youden_index = j_scores[optimal_idx]

    # Calculate metrics at optimal threshold
    predictions = (probabilities >= optimal_threshold).astype(int)

    metrics = {
        'threshold': float(optimal_threshold),
        'youden_index': float(youden_index),
        'tpr': float(tpr[optimal_idx]),
        'fpr': float(fpr[optimal_idx]),
        'specificity': float(1 - fpr[optimal_idx]),
        'sensitivity': float(tpr[optimal_idx]),
        'auroc': float(auc(fpr, tpr)),
        'accuracy': float(accuracy_score(targets, predictions)),
        'f1_score': float(f1_score(targets, predictions, zero_division=0)),
        'precision': float(precision_score(targets, predictions, zero_division=0)),
        'recall': float(recall_score(targets, predictions, zero_division=0))
    }

    return optimal_threshold, youden_index, metrics


def evaluate_model(run_id, seq_length, device):
    """Load model and find optimal thresholds on validation set"""
    print(f"\n{'='*80}")
    print(f"Evaluating {run_id} (seq {seq_length})")
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

    # Setup for validation and test
    data_module.setup(stage='fit')  # This creates val_dataset
    data_module.setup(stage='test')  # This creates test_dataset

    # Load model
    model = LitClassifier(data_module=data_module, **args)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)

    print("✅ Model & data loaded")

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    # ========================================
    # Step 1: Find optimal thresholds on VALIDATION set
    # ========================================
    print(f"\n{'='*80}")
    print("STEP 1: Finding optimal thresholds on VALIDATION set")
    print(f"{'='*80}\n")

    val_loader = DataLoader(data_module.val_dataset, batch_size=4, shuffle=False, num_workers=4)

    val_probs_all = []
    val_targets_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(val_loader)}", flush=True)

            fmri = batch['fmri_sequence'].to(device).float()
            targets = batch['target']  # (B, T, 7)

            # Get logits
            logits = model(fmri)  # (B, T, 7, 2)

            # Softmax to probabilities
            probs = torch.softmax(logits, dim=-1)  # (B, T, 7, 2)
            probs_pos = probs[..., 1].cpu().numpy()  # (B, T, 7) - prob of class 1

            val_probs_all.append(probs_pos)
            val_targets_all.append(targets.cpu().numpy())

    # Concatenate all batches
    val_probs_all = np.concatenate(val_probs_all, axis=0)  # (N, T, 7)
    val_targets_all = np.concatenate(val_targets_all, axis=0)  # (N, T, 7)

    # Flatten time dimension
    val_probs_flat = val_probs_all.reshape(-1, 7)  # (N*T, 7)
    val_targets_flat = val_targets_all.reshape(-1, 7)  # (N*T, 7)

    print(f"\n✅ Validation set collected: {val_probs_flat.shape[0]} samples")

    # Find optimal threshold for each emotion
    optimal_thresholds = {}

    for emo_idx, emo_name in enumerate(emotion_names):
        print(f"\n{emo_name}:")

        targets = val_targets_flat[:, emo_idx]
        probs = val_probs_flat[:, emo_idx]

        # Remove NaN
        valid_mask = ~np.isnan(targets) & ~np.isnan(probs)
        targets = targets[valid_mask]
        probs = probs[valid_mask]

        if len(targets) == 0:
            print(f"  No valid samples!")
            continue

        # Class distribution
        n_positive = (targets == 1).sum()
        n_negative = (targets == 0).sum()
        print(f"  Samples: {len(targets)} (pos={n_positive}, neg={n_negative}, ratio={n_positive/len(targets):.2%})")

        # Find optimal threshold
        opt_thr, youden, metrics = find_optimal_threshold_youden(targets, probs)

        optimal_thresholds[emo_name] = metrics

        print(f"  Optimal threshold: {opt_thr:.4f}")
        print(f"  Youden index: {youden:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} | Specificity: {metrics['specificity']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

    # ========================================
    # Step 2: Evaluate on TEST set with optimal thresholds
    # ========================================
    print(f"\n{'='*80}")
    print("STEP 2: Evaluating on TEST set with optimal thresholds")
    print(f"{'='*80}\n")

    test_loader = DataLoader(data_module.test_dataset, batch_size=4, shuffle=False, num_workers=4)

    test_probs_all = []
    test_targets_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(test_loader)}", flush=True)

            fmri = batch['fmri_sequence'].to(device).float()
            targets = batch['target']

            logits = model(fmri)
            probs = torch.softmax(logits, dim=-1)
            probs_pos = probs[..., 1].cpu().numpy()

            test_probs_all.append(probs_pos)
            test_targets_all.append(targets.cpu().numpy())

    test_probs_all = np.concatenate(test_probs_all, axis=0)
    test_targets_all = np.concatenate(test_targets_all, axis=0)

    test_probs_flat = test_probs_all.reshape(-1, 7)
    test_targets_flat = test_targets_all.reshape(-1, 7)

    print(f"\n✅ Test set collected: {test_probs_flat.shape[0]} samples")

    # Evaluate with optimal thresholds
    test_results = {}

    for emo_idx, emo_name in enumerate(emotion_names):
        print(f"\n{emo_name}:")

        if emo_name not in optimal_thresholds:
            print(f"  No optimal threshold found (skipped)")
            continue

        opt_thr = optimal_thresholds[emo_name]['threshold']

        targets = test_targets_flat[:, emo_idx]
        probs = test_probs_flat[:, emo_idx]

        # Remove NaN
        valid_mask = ~np.isnan(targets) & ~np.isnan(probs)
        targets = targets[valid_mask]
        probs = probs[valid_mask]

        if len(targets) == 0:
            print(f"  No valid samples!")
            continue

        # Predictions with optimal threshold
        predictions_opt = (probs >= opt_thr).astype(int)
        predictions_050 = (probs >= 0.5).astype(int)

        # Calculate metrics
        fpr, tpr, _ = roc_curve(targets, probs)
        auroc = auc(fpr, tpr)

        test_results[emo_name] = {
            'optimal_threshold': float(opt_thr),
            'auroc': float(auroc),
            'with_optimal_threshold': {
                'accuracy': float(accuracy_score(targets, predictions_opt)),
                'f1_score': float(f1_score(targets, predictions_opt, zero_division=0)),
                'precision': float(precision_score(targets, predictions_opt, zero_division=0)),
                'recall': float(recall_score(targets, predictions_opt, zero_division=0))
            },
            'with_threshold_0.5': {
                'accuracy': float(accuracy_score(targets, predictions_050)),
                'f1_score': float(f1_score(targets, predictions_050, zero_division=0)),
                'precision': float(precision_score(targets, predictions_050, zero_division=0)),
                'recall': float(recall_score(targets, predictions_050, zero_division=0))
            }
        }

        print(f"  Optimal threshold: {opt_thr:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  With optimal threshold - Acc: {test_results[emo_name]['with_optimal_threshold']['accuracy']:.4f}, "
              f"F1: {test_results[emo_name]['with_optimal_threshold']['f1_score']:.4f}")
        print(f"  With threshold 0.5 - Acc: {test_results[emo_name]['with_threshold_0.5']['accuracy']:.4f}, "
              f"F1: {test_results[emo_name]['with_threshold_0.5']['f1_score']:.4f}")

    return optimal_thresholds, test_results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Model IDs
    experiments = [
        ("mc3r4vhf", 20, "Seq 20"),
        ("gajr5p1p", 30, "Seq 30")
    ]

    all_results = {}

    for run_id, seq_length, name in experiments:
        optimal_thresholds, test_results = evaluate_model(run_id, seq_length, device)

        all_results[run_id] = {
            'name': name,
            'seq_length': seq_length,
            'optimal_thresholds_from_validation': optimal_thresholds,
            'test_set_results': test_results
        }

    # Save results
    output_path = project_root / "analysis" / "optimal_thresholds.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*80}")
