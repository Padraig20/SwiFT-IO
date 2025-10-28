#!/usr/bin/env python3
"""
Select sequences with high prediction confidence for each emotion (CLASSIFICATION task).
These sequences are where the model predicts positive class (1) with high confidence.

Run ID: mc3r4vhf
Sequence length: 20 TRs
Output: 5-10 sequences per emotion per subject
"""

import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import json

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule
from torch.utils.data import DataLoader, Subset

# 7 emotions
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def select_peak_sequences(run_id='mc3r4vhf', top_k=10, subject_list=None):
    """
    Select sequences with high positive class prediction confidence for each emotion.

    Args:
        run_id: Model run ID (classification)
        top_k: Number of top sequences to select per emotion
        subject_list: Optional list of subjects to process. If None, process all test subjects.
    """
    print("="*70)
    print(f"Selecting High-Confidence Sequences - CLASSIFICATION - Run ID: {run_id}")
    print(f"Top K sequences per emotion: {top_k}")
    print("="*70)

    # Load checkpoint to get configuration
    ckpt_path = project_root / f"output/moviefmri/{run_id}/checkpt-epoch=08-valid_acc=1.00.ckpt"
    if not ckpt_path.exists():
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

    # Get test dataset
    testset = data_module.test_dataset
    all_subjects = sorted(list(set([str(x[1]) for x in testset.data])))

    # Filter by subject_list if provided
    if subject_list:
        subjects_to_process = [s for s in all_subjects if s in subject_list]
        print(f"\n📊 Processing {len(subjects_to_process)} specified subjects")
    else:
        subjects_to_process = all_subjects
        print(f"\n📊 Processing all {len(subjects_to_process)} test subjects")

    # Process each subject
    all_subject_results = {}

    for subj_idx, subject in enumerate(subjects_to_process, 1):
        print(f"\n{'='*70}")
        print(f"[{subj_idx}/{len(subjects_to_process)}] Processing {subject}")
        print(f"{'='*70}")

        # Get sequences for this subject
        subject_indices = [
            idx for idx, data_tuple in enumerate(testset.data)
            if str(data_tuple[1]) == subject
        ]

        if not subject_indices:
            print(f"  ⚠️ No sequences found")
            continue

        print(f"  Found {len(subject_indices)} sequences")

        # Create dataloader
        test_loader = DataLoader(Subset(testset, subject_indices),
                                batch_size=1, shuffle=False, num_workers=0)

        # Collect predictions for each sequence
        sequence_predictions = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                fmri, target, subj_name = data[0], data[1], data[2]

                # Get data tuple info
                data_tuple = testset.data[subject_indices[batch_idx]]
                start_frame = data_tuple[3]
                seq_length = data_tuple[4]
                end_frame = start_frame + seq_length - 1

                # Forward pass
                output = model(fmri)  # [1, time, 7, 2]
                probs = torch.softmax(output, dim=-1)  # [1, time, 7, 2]

                # Get probability of positive class (class 1) for each emotion
                pos_probs = probs[0, :, :, 1].cpu().numpy()  # [time, 7]
                targets = target[0].cpu().numpy()  # [time, 7]

                sequence_predictions.append({
                    'sequence_idx': batch_idx,
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'pos_probs': pos_probs,  # [time, 7]
                    'targets': targets  # [time, 7]
                })

        # For each emotion, select top-k sequences based on:
        # 1. Average positive prediction probability
        # 2. Correctness (for true positive cases)
        subject_top_sequences = {}

        for emotion_idx, emotion_name in enumerate(emotion_labels):
            # Score each sequence for this emotion
            scored_sequences = []

            for seq_info in sequence_predictions:
                pos_prob = seq_info['pos_probs'][:, emotion_idx]  # [time]
                target = seq_info['targets'][:, emotion_idx]  # [time]

                # Calculate metrics
                avg_prob = np.mean(pos_prob)
                max_prob = np.max(pos_prob)
                std_prob = np.std(pos_prob)

                # Check correctness: for positive predictions, check if target is also positive
                predictions = (pos_prob > 0.5).astype(int)
                accuracy = np.mean(predictions == target)

                # Combined score: favor high probability + high accuracy
                # We want sequences where model is confident AND correct
                score = avg_prob * accuracy

                scored_sequences.append({
                    'sequence_idx': seq_info['sequence_idx'],
                    'start_frame': seq_info['start_frame'],
                    'end_frame': seq_info['end_frame'],
                    'avg_prob': float(avg_prob),
                    'max_prob': float(max_prob),
                    'std_prob': float(std_prob),
                    'accuracy': float(accuracy),
                    'score': float(score),
                    'num_positive_targets': int(np.sum(target)),
                    'num_positive_predictions': int(np.sum(predictions))
                })

            # Sort by combined score (descending)
            sorted_sequences = sorted(scored_sequences, key=lambda x: x['score'], reverse=True)

            # Select top-k
            top_sequences = sorted_sequences[:top_k]

            subject_top_sequences[emotion_name] = top_sequences

            print(f"\n  {emotion_name}:")
            print(f"  {'Rank':<6} {'Seq':<5} {'TR Range':<15} {'Avg Prob':<10} {'Accuracy':<10} {'Score':<10}")
            print(f"  {'-'*66}")
            for rank, seq_info in enumerate(top_sequences, 1):
                tr_range = f"TR{seq_info['start_frame']:03d}-{seq_info['end_frame']:03d}"
                print(f"  {rank:<6} {seq_info['sequence_idx']:<5} {tr_range:<15} "
                      f"{seq_info['avg_prob']:<10.4f} {seq_info['accuracy']:<10.4f} "
                      f"{seq_info['score']:<10.4f}")

        all_subject_results[subject] = subject_top_sequences

    # Save results
    output_dir = project_root / f"analysis/4_IGmap/clf_peak_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sequence length from args
    seq_length = args_model_dict.get('sequence_length', 20)

    # Save full results
    output_path = output_dir / f"{run_id}_clf_peak_sequences_top{top_k}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'task_type': 'classification',
            'top_k': top_k,
            'sequence_length': seq_length,
            'emotion_labels': emotion_labels,
            'subjects': all_subject_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*70}")

    # Create summary
    print(f"\nSUMMARY:")
    print(f"  Total subjects processed: {len(all_subject_results)}")
    print(f"  Sequences per emotion per subject: {top_k}")
    print(f"  Total sequences selected: {len(all_subject_results) * len(emotion_labels) * top_k}")

    return all_subject_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='mc3r4vhf')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top sequences to select per emotion (default: 10)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='Specific subjects to process (default: all test subjects)')
    args = parser.parse_args()

    results = select_peak_sequences(
        run_id=args.run_id,
        top_k=args.top_k,
        subject_list=args.subjects
    )

    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
