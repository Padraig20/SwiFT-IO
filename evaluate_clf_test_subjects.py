#!/usr/bin/env python3
"""
Evaluate all test set subjects for CLASSIFICATION task
Find top performers by per-emotion accuracy/F1 score
Run ID: mc3r4vhf (classification model)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

def evaluate_subjects(run_id='mc3r4vhf'):
    print("="*70)
    print(f"Evaluating Test Set Subjects - CLASSIFICATION - Run ID: {run_id}")
    print("="*70)

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{run_id}/checkpt-epoch=08-valid_acc=1.00.ckpt"
    if not ckpt_path.exists():
        # Try to find any checkpoint
        ckpt_files = list((project_root / f"output/moviefmri/{run_id}").glob("checkpt*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in output/moviefmri/{run_id}")
        ckpt_path = ckpt_files[0]

    print(f"\n✅ Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    # Setup data module
    args_model_dict["num_workers"] = 0
    args_model_dict["eval_num_workers"] = 0
    args_model_dict['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args_model_dict['default_root_dir'] = str(project_root / "output/moviefmri")
    args_model_dict['shuffle_time_sequence'] = False
    args_model_dict['time_as_channel'] = False
    args_model_dict['eval_batch_size'] = 1
    args_model_dict['bad_subj_path'] = None
    args_model_dict['limit_training_samples'] = 0
    args_model_dict['downstream_task'] = 'emotions'
    args_model_dict['downstream_task_type'] = 'classification'
    args_model_dict['decoder'] = 'series_decoder'

    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 2)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = ['Age', 'Sex']
    if 'train_split' not in args_model_dict:
        args_model_dict['train_split'] = 0.7
    if 'val_split' not in args_model_dict:
        args_model_dict['val_split'] = 0.15

    print("\n🚀 Initializing model & data...")
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

    model = LitClassifier(data_module=data_module, **args_model_dict)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cpu()
    print("✅ Model & Data initialized")

    # Get test dataset
    testset = data_module.test_dataset
    test_subjects = sorted(list(set([str(x[1]) for x in testset.data])))
    print(f"\n📊 Found {len(test_subjects)} unique subjects in test set")

    # Evaluate each subject
    subject_results = {}
    emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    for subj_idx, subject in enumerate(test_subjects, 1):
        print(f"\n[{subj_idx}/{len(test_subjects)}] Evaluating {subject}...", flush=True)

        # Get subject indices
        subj_indices = [
            idx for idx, data_tuple in enumerate(testset.data)
            if str(data_tuple[1]) == subject
        ]

        if not subj_indices:
            print(f"  ⚠️ No data found")
            continue

        print(f"  Found {len(subj_indices)} sequences")

        # Create dataloader
        test_loader = DataLoader(Subset(testset, subj_indices),
                                batch_size=1, shuffle=False, num_workers=0)

        # Collect predictions and targets
        all_preds = []  # Will store probabilities [num_sequences, time, 7, 2]
        all_pred_labels = []  # Will store predicted labels [num_sequences, time, 7]
        all_targets = []  # Will store target labels [num_sequences, time, 7]

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                fmri = data['fmri_sequence'].float().cpu()
                target = data['target'].float().cpu()
                subj_name = data['subject_name']

                # Forward pass
                output = model(fmri)  # [1, time, 7, 2] - logits for binary classification per emotion

                # Get probabilities and predictions
                probs = torch.softmax(output, dim=-1)  # [1, time, 7, 2]
                pred_labels = torch.argmax(probs, dim=-1)  # [1, time, 7] - predicted class (0 or 1)

                all_preds.append(probs.squeeze(0).cpu().numpy())  # [time, 7, 2]
                all_pred_labels.append(pred_labels.squeeze(0).cpu().numpy())  # [time, 7]
                all_targets.append(target.squeeze(0).cpu().numpy())  # [time, 7]

        # Stack results
        all_preds = np.array(all_preds)  # [num_sequences, time, 7, 2]
        all_pred_labels = np.array(all_pred_labels)  # [num_sequences, time, 7]
        all_targets = np.array(all_targets)  # [num_sequences, time, 7]

        # Calculate metrics per emotion
        emotion_metrics = {}
        for emo_idx, emotion_name in enumerate(emotion_labels):
            # Flatten across sequences and time for this emotion
            y_true = all_targets[:, :, emo_idx].flatten()
            y_pred = all_pred_labels[:, :, emo_idx].flatten()
            y_prob = all_preds[:, :, emo_idx, 1].flatten()  # Probability of class 1

            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            # Calculate AUROC and AUPRC (need probabilities)
            # Handle edge case where all samples are same class
            try:
                auroc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auroc = 0.0  # All samples same class

            try:
                auprc = average_precision_score(y_true, y_prob)
            except ValueError:
                auprc = 0.0  # All samples same class

            emotion_metrics[emotion_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'auroc': float(auroc),
                'auprc': float(auprc),
                'num_samples': len(y_true),
                'num_positive': int(np.sum(y_true)),
                'num_negative': int(len(y_true) - np.sum(y_true))
            }

        # Calculate overall metrics (average across emotions)
        avg_acc = np.mean([m['accuracy'] for m in emotion_metrics.values()])
        avg_f1 = np.mean([m['f1_score'] for m in emotion_metrics.values()])
        avg_auroc = np.mean([m['auroc'] for m in emotion_metrics.values()])
        avg_auprc = np.mean([m['auprc'] for m in emotion_metrics.values()])

        print(f"  Average Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, AUROC: {avg_auroc:.4f}, AUPRC: {avg_auprc:.4f}")

        subject_results[subject] = {
            'num_sequences': len(subj_indices),
            'avg_accuracy': float(avg_acc),
            'avg_f1_score': float(avg_f1),
            'avg_auroc': float(avg_auroc),
            'avg_auprc': float(avg_auprc),
            'emotion_metrics': emotion_metrics,
            'predictions': all_preds.tolist(),  # For detailed analysis later
            'targets': all_targets.tolist()
        }

    # Rank subjects by average performance (using AUPRC - more robust for imbalanced data)
    ranked_subjects = sorted(
        subject_results.items(),
        key=lambda x: x[1]['avg_auprc'],
        reverse=True  # Higher is better for classification
    )

    # Display overall results
    print("\n" + "="*80)
    print("OVERALL RESULTS - Ranked by Average AUPRC")
    print("="*80)
    print(f"{'Rank':<6} {'Subject':<20} {'AUROC':<10} {'AUPRC':<10} {'F1':<10} {'Acc':<10}")
    print("-"*80)
    for rank, (subject, result) in enumerate(ranked_subjects[:10], 1):
        print(f"{rank:<6} {subject:<20} {result['avg_auroc']:<10.4f} "
              f"{result['avg_auprc']:<10.4f} {result['avg_f1_score']:<10.4f} "
              f"{result['avg_accuracy']:<10.4f}")

    # Save results
    output_dir = project_root / "analysis/4_IGmap/subject_performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{run_id}_clf_test_subject_performance.json"
    with open(output_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'task_type': 'classification',
            'checkpoint': ckpt_path.name,
            'num_subjects': len(subject_results),
            'emotion_labels': emotion_labels,
            'top5_subjects': [s[0] for s in ranked_subjects[:5]],
            'top10_subjects': [s[0] for s in ranked_subjects[:10]],
            'all_subjects': subject_results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("="*70)

    return subject_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='mc3r4vhf',
                       help='Model run ID (classification)')
    args = parser.parse_args()

    results = evaluate_subjects(run_id=args.run_id)

    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
