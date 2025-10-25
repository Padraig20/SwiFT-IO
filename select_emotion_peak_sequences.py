#!/usr/bin/env python3
"""
Select sequences with peak emotion values for each of 7 emotions.
These sequences are model-agnostic (based on ground truth targets only).

Run ID: opr6oq97
Sequence length: 30 TRs
Output: 5-10 sequences per emotion
"""

import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import json

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.utils.data_module import fMRIDataModule

# 7 emotions
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def select_peak_sequences(run_id='opr6oq97', top_k=10, subject_list=None):
    """
    Select sequences with peak values for each emotion.

    Args:
        run_id: Model run ID (used to load checkpoint config)
        top_k: Number of top sequences to select per emotion
        subject_list: Optional list of subjects to process. If None, process all test subjects.
    """
    print("="*70)
    print(f"Selecting Emotion Peak Sequences - Run ID: {run_id}")
    print(f"Top K sequences per emotion: {top_k}")
    print("="*70)

    # Load checkpoint to get configuration
    ckpt_path = project_root / f"output/moviefmri/{run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{run_id}").glob("checkpt*"))[0]

    print(f"\n✅ Loading checkpoint config: {ckpt_path.name}")
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

    print("\n🚀 Initializing data module...")
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

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
        # data format: (i, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex)
        subject_sequences = [
            (idx, d) for idx, d in enumerate(testset.data)
            if str(d[1]) == subject
        ]

        if not subject_sequences:
            print(f"  ⚠️ No sequences found")
            continue

        print(f"  Found {len(subject_sequences)} sequences")

        # Collect emotion scores for each sequence
        sequence_emotion_scores = defaultdict(list)  # emotion_idx -> [(seq_idx, avg_score, tr_range, data)]

        for seq_idx, (data_idx, data_tuple) in enumerate(subject_sequences):
            # data_tuple: (i, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex)
            start_frame = data_tuple[3]
            seq_length = data_tuple[4]
            target = data_tuple[6]  # Shape: [time, emotions]

            end_frame = start_frame + seq_length - 1

            # Convert target to numpy if it's a tensor
            if torch.is_tensor(target):
                target = target.numpy()

            # Calculate average score for each emotion over the 30 TRs
            for emotion_idx, emotion_name in enumerate(emotion_labels):
                avg_score = np.mean(target[:, emotion_idx])

                sequence_emotion_scores[emotion_idx].append({
                    'sequence_idx': seq_idx,
                    'data_idx': data_idx,
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'avg_score': float(avg_score),
                    'max_score': float(np.max(target[:, emotion_idx])),
                    'min_score': float(np.min(target[:, emotion_idx])),
                    'std_score': float(np.std(target[:, emotion_idx])),
                })

        # Select top-k sequences for each emotion
        subject_top_sequences = {}

        for emotion_idx, emotion_name in enumerate(emotion_labels):
            # Sort by average score (descending)
            sorted_sequences = sorted(
                sequence_emotion_scores[emotion_idx],
                key=lambda x: x['avg_score'],
                reverse=True
            )

            # Select top-k
            top_sequences = sorted_sequences[:top_k]

            subject_top_sequences[emotion_name] = top_sequences

            print(f"\n  {emotion_name}:")
            print(f"  {'Rank':<6} {'Seq':<5} {'TR Range':<15} {'Avg':<8} {'Max':<8} {'Std':<8}")
            print(f"  {'-'*60}")
            for rank, seq_info in enumerate(top_sequences, 1):
                tr_range = f"TR{seq_info['start_frame']:03d}-{seq_info['end_frame']:03d}"
                print(f"  {rank:<6} {seq_info['sequence_idx']:<5} {tr_range:<15} "
                      f"{seq_info['avg_score']:<8.4f} {seq_info['max_score']:<8.4f} "
                      f"{seq_info['std_score']:<8.4f}")

        all_subject_results[subject] = subject_top_sequences

    # Save results
    output_dir = project_root / f"analysis/4_IGmap/emotion_peak_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    output_path = output_dir / f"{run_id}_emotion_peak_sequences_top{top_k}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'top_k': top_k,
            'sequence_length': 30,
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
    parser.add_argument('--run_id', type=str, default='opr6oq97')
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
