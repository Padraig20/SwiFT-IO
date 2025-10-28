#!/usr/bin/env python3
"""
Evaluate LSTM Classification Baseline from checkpoint

This script evaluates a trained LSTM classification model on the test set.

Usage:
    python evaluate_lstm_clf_baseline.py \
        --run_id <wandb_run_id> \
        --checkpoint_name checkpt-epoch=XX-valid_acc=X.XX.ckpt
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule


def evaluate_lstm_classification(run_id, checkpoint_name=None):
    """
    Evaluate LSTM classification baseline

    Args:
        run_id: WandB run ID
        checkpoint_name: Checkpoint filename (e.g., 'checkpt-epoch=XX-valid_acc=X.XX.ckpt')
    """
    print("="*70)
    print(f"LSTM Classification Baseline Evaluation")
    print(f"Run ID: {run_id}")
    print("="*70)

    # Find checkpoint
    run_dir = project_root / f"output/moviefmri/{run_id}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint_name is None:
        # Find best checkpoint
        ckpt_files = list(run_dir.glob("checkpt*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        # Try to find the one with highest accuracy
        ckpt_path = sorted(ckpt_files, key=lambda x: x.name, reverse=True)[0]
    else:
        ckpt_path = run_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path.name}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    # Setup data module
    args_model_dict["num_workers"] = 0
    args_model_dict["eval_num_workers"] = 0
    args_model_dict['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args_model_dict['default_root_dir'] = str(project_root / "output/moviefmri")
    args_model_dict['shuffle_time_sequence'] = False
    args_model_dict['time_as_channel'] = False
    args_model_dict['eval_batch_size'] = 16
    args_model_dict['bad_subj_path'] = None
    args_model_dict['limit_training_samples'] = 0
    args_model_dict['downstream_task'] = 'emotions'
    args_model_dict['downstream_task_type'] = 'classification'

    # Set defaults for missing keys
    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 2)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = ['Age', 'Sex']
    if 'train_split' not in args_model_dict:
        args_model_dict['train_split'] = 0.7
    if 'val_split' not in args_model_dict:
        args_model_dict['val_split'] = 0.15

    print("\nInitializing model & data...")
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

    model = LitClassifier(data_module=data_module, **args_model_dict)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cpu()

    print("Model & Data initialized")
    print(f"Test set size: {len(data_module.test_dataset)} samples")

    # Evaluate on test set
    print("\nEvaluating on test set...")

    test_loader = data_module.test_dataloader()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx}/{len(test_loader)}", flush=True)

            fmri_data = batch['fmri_sequence']
            targets = batch['target']  # (batch, seq_len, num_emotions)

            # Forward pass
            outputs = model(fmri_data)  # (batch, seq_len, num_emotions)

            # Collect predictions and targets
            preds = outputs.cpu().numpy()  # (batch, seq_len, num_emotions)
            targets = targets.cpu().numpy()  # (batch, seq_len, num_emotions)

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)  # (num_samples, seq_len, num_emotions)
    all_targets = np.concatenate(all_targets, axis=0)  # (num_samples, seq_len, num_emotions)

    print(f"Predictions shape: {all_preds.shape}")
    print(f"Targets shape: {all_targets.shape}")

    # Flatten for overall metrics
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()

    # Calculate metrics
    print("\nCalculating metrics...")

    overall_acc = accuracy_score(targets_flat, preds_flat)
    overall_f1 = f1_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    overall_precision = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    overall_recall = recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)

    metrics = {
        'test_acc': float(overall_acc),
        'test_f1': float(overall_f1),
        'test_precision': float(overall_precision),
        'test_recall': float(overall_recall),
    }

    # Per-emotion metrics
    emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    for e in range(7):
        preds_e = all_preds[:, :, e].flatten()
        targets_e = all_targets[:, :, e].flatten()

        acc_e = accuracy_score(targets_e, preds_e)
        f1_e = f1_score(targets_e, preds_e, average='weighted', zero_division=0)
        precision_e = precision_score(targets_e, preds_e, average='weighted', zero_division=0)
        recall_e = recall_score(targets_e, preds_e, average='weighted', zero_division=0)

        metrics[f'test_acc_{e}'] = float(acc_e)
        metrics[f'test_f1_{e}'] = float(f1_e)
        metrics[f'test_precision_{e}'] = float(precision_e)
        metrics[f'test_recall_{e}'] = float(recall_e)

    # Save results
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "lstm_clf_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("LSTM Classification Baseline - Test Results")
    print("="*70)

    print(f"\nOverall Test Performance:")
    print(f"  Accuracy:  {overall_acc:.4f}")
    print(f"  F1 Score:  {overall_f1:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")

    print(f"\nPer-Emotion Performance:")
    for e, name in enumerate(emotion_labels):
        print(f"  {name:8s}: Acc={metrics[f'test_acc_{e}']:.4f}, "
              f"F1={metrics[f'test_f1_{e}']:.4f}, "
              f"Precision={metrics[f'test_precision_{e}']:.4f}, "
              f"Recall={metrics[f'test_recall_{e}']:.4f}")

    print("="*70)

    # Save summary
    summary_path = output_dir / "lstm_clf_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LSTM Classification Baseline Evaluation Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Checkpoint: {ckpt_path.name}\n")
        f.write(f"Test samples: {len(data_module.test_dataset)}\n\n")

        f.write(f"Overall Test Performance:\n")
        f.write(f"  Accuracy:  {overall_acc:.4f}\n")
        f.write(f"  F1 Score:  {overall_f1:.4f}\n")
        f.write(f"  Precision: {overall_precision:.4f}\n")
        f.write(f"  Recall:    {overall_recall:.4f}\n\n")

        f.write(f"Per-Emotion Performance:\n")
        for e, name in enumerate(emotion_labels):
            f.write(f"  {name}: Acc={metrics[f'test_acc_{e}']:.4f}, "
                   f"F1={metrics[f'test_f1_{e}']:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                       help="WandB run ID")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                       help="Checkpoint filename (optional)")

    args = parser.parse_args()

    evaluate_lstm_classification(args.run_id, args.checkpoint_name)
