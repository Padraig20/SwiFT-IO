#!/usr/bin/env python3
"""
Evaluate all test set subjects and find top performers by MSE
Run ID: opr6oq97
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import json

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

def evaluate_subjects(run_id='opr6oq97'):
    print("="*70)
    print(f"Evaluating Test Set Subjects - Run ID: {run_id}")
    print("="*70)

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{run_id}").glob("checkpt*"))[0]

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
    args_model_dict['decoder'] = 'series_decoder'

    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 777)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = None
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

        # Calculate MSE for this subject
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                if data is None:
                    continue

                fmri_seq = data['fmri_sequence'].float().cpu()
                target = data['target'].float().cpu()  # Shape: [batch, time, emotions]

                # Forward pass
                pred = model(fmri_seq)  # Shape: [time, emotions] or [batch, time, emotions]

                # Ensure batch dimension exists
                if pred.dim() == 2:
                    # [time, emotions] -> add batch dim
                    pred = pred.unsqueeze(0)  # [1, time, emotions]

                # Now pred should be [batch, time, emotions]
                # Average over time dimension
                pred_mean = pred.mean(dim=1)  # [batch, emotions]
                target_mean = target.mean(dim=1)  # [batch, emotions]

                # Remove batch dimension for single batch
                pred_mean = pred_mean.squeeze(0)  # [emotions]
                target_mean = target_mean.squeeze(0)  # [emotions]

                all_preds.append(pred_mean.numpy())
                all_targets.append(target_mean.numpy())

        if not all_preds:
            print(f"  ⚠️ No valid predictions")
            continue

        # Calculate MSE
        preds = np.vstack(all_preds)  # [num_sequences, emotions]
        targets = np.vstack(all_targets)  # [num_sequences, emotions]
        mse = np.mean((preds - targets) ** 2)

        subject_results[subject] = {
            'mse': float(mse),
            'num_sequences': len(subj_indices),
            'predictions': preds.tolist(),
            'targets': targets.tolist()
        }

        print(f"  ✅ MSE: {mse:.6f}")

    # Sort by MSE and get top 5
    sorted_subjects = sorted(subject_results.items(), key=lambda x: x[1]['mse'])

    print("\n" + "="*70)
    print("TOP 5 SUBJECTS (Lowest MSE):")
    print("="*70)
    print(f"{'Rank':<6} {'Subject':<20} {'MSE':<12} {'# Sequences':<12}")
    print("-"*70)

    top5_subjects = []
    for rank, (subject, result) in enumerate(sorted_subjects[:5], 1):
        print(f"{rank:<6} {subject:<20} {result['mse']:<12.6f} {result['num_sequences']:<12}")
        top5_subjects.append(subject)

    print("\n" + "="*70)
    print("BOTTOM 5 SUBJECTS (Highest MSE):")
    print("="*70)
    print(f"{'Rank':<6} {'Subject':<20} {'MSE':<12} {'# Sequences':<12}")
    print("-"*70)

    for rank, (subject, result) in enumerate(sorted_subjects[-5:], len(sorted_subjects)-4):
        print(f"{rank:<6} {subject:<20} {result['mse']:<12.6f} {result['num_sequences']:<12}")

    # Save results
    output_dir = project_root / f"analysis/4_IGmap/subject_performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    output_path = output_dir / f"{run_id}_test_subject_mse.json"
    with open(output_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'all_subjects': subject_results,
            'sorted_subjects': [(s, r['mse'], r['num_sequences']) for s, r in sorted_subjects],
            'top5_subjects': top5_subjects
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Save top 5 subject list
    top5_path = output_dir / f"{run_id}_top5_subjects.txt"
    with open(top5_path, 'w') as f:
        for subject in top5_subjects:
            f.write(f"{subject}\n")

    print(f"✅ Top 5 subjects saved to: {top5_path}")

    return top5_subjects, subject_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='opr6oq97')
    args = parser.parse_args()

    top5_subjects, results = evaluate_subjects(args.run_id)

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Total subjects evaluated: {len(results)}")
    print(f"Top 5 subjects: {', '.join(top5_subjects)}")
    print("="*70)
