#!/usr/bin/env python3
"""
각 emotion별로 intensity가 높은 sequences 선택

이 스크립트는:
1. Subject의 모든 sequences를 로드
2. 각 emotion별로 intensity (평균값) 계산
3. 각 emotion별로 top N sequences 선택
4. 선택된 sequences의 인덱스 저장
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.append('/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/src')
from module.utils.data_module import fMRIDataModule

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def select_high_intensity_sequences(subject_name, ckpt_path, top_n=15):
    """
    각 emotion별로 intensity가 높은 top N sequences 선택

    Args:
        subject_name: Subject ID (e.g., 'sub-NDARRL379BET')
        ckpt_path: Model checkpoint path
        top_n: 각 emotion별로 선택할 sequences 수

    Returns:
        dict: {emotion_idx: [sequence_indices]}
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args_dict = ckpt['hyper_parameters']

    # Setup data module
    args_dict['image_path'] = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
    args_dict['num_workers'] = 0
    args_dict['eval_num_workers'] = 0
    args_dict['dataset_split_seed'] = args_dict.get('dataset_split_seed', 777)
    args_dict['stratified_params'] = None
    args_dict['train_split'] = 0.7
    args_dict['val_split'] = 0.15
    args_dict['bad_subj_path'] = None
    args_dict['limit_training_samples'] = 0
    args_dict['shuffle_time_sequence'] = False
    args_dict['eval_batch_size'] = 1

    data_module = fMRIDataModule(**args_dict)
    data_module.setup(stage='test')

    testset = data_module.test_dataset

    # Find all sequences for this subject
    subj_indices = [
        idx for idx, s in enumerate(testset)
        if (s["subject_name"] if isinstance(s["subject_name"], str) else s["subject_name"][0]) == subject_name
    ]

    if not subj_indices:
        raise ValueError(f"No sequences found for {subject_name}")

    print(f"Found {len(subj_indices)} sequences for {subject_name}")

    # Load data and compute emotion intensities
    test_loader = DataLoader(Subset(testset, subj_indices), batch_size=1, shuffle=False, num_workers=0)

    # Store intensities for each emotion
    emotion_intensities = {i: [] for i in range(7)}  # 7 emotions
    sequence_info = []  # Store (global_idx, TR_index, data)

    print("\nComputing emotion intensities for all sequences...")
    for local_idx, data in enumerate(test_loader):
        global_idx = subj_indices[local_idx]
        TR_index = int(data['TR'])
        target = data['target']  # Shape: [batch=1, time, num_emotions]

        # Compute mean intensity for each emotion across time
        target_np = target[0].cpu().numpy()  # [time, num_emotions]

        for emotion_idx in range(7):
            emotion_intensity = target_np[:, emotion_idx].mean()
            emotion_intensities[emotion_idx].append({
                'global_idx': global_idx,
                'local_idx': local_idx,
                'TR_index': TR_index,
                'intensity': emotion_intensity
            })

        if (local_idx + 1) % 100 == 0:
            print(f"  Processed {local_idx + 1}/{len(subj_indices)} sequences")

    # Select top N sequences for each emotion
    selected_sequences = {}

    print(f"\n{'='*80}")
    print(f"Selecting top {top_n} sequences for each emotion")
    print(f"{'='*80}")

    for emotion_idx, emotion_name in enumerate(emotion_labels):
        # Sort by intensity (descending)
        sorted_seqs = sorted(emotion_intensities[emotion_idx],
                           key=lambda x: x['intensity'],
                           reverse=True)

        # Select top N
        top_seqs = sorted_seqs[:top_n]
        selected_sequences[emotion_idx] = top_seqs

        print(f"\n{emotion_name} (emotion {emotion_idx}):")
        print(f"  Intensity range: [{sorted_seqs[-1]['intensity']:.4f}, {sorted_seqs[0]['intensity']:.4f}]")
        print(f"  Selected top {len(top_seqs)} sequences:")
        for i, seq in enumerate(top_seqs[:5]):  # Show first 5
            print(f"    {i+1}. TR {seq['TR_index']:03d}, intensity={seq['intensity']:.4f}")
        if len(top_seqs) > 5:
            print(f"    ... and {len(top_seqs)-5} more")

    return selected_sequences

def save_selected_sequences(selected_sequences, output_path):
    """Save selected sequences to file"""
    import json

    # Convert to serializable format
    output = {}
    for emotion_idx, seqs in selected_sequences.items():
        output[emotion_labels[emotion_idx]] = [
            {
                'global_idx': int(seq['global_idx']),
                'local_idx': int(seq['local_idx']),
                'TR_index': int(seq['TR_index']),
                'intensity': float(seq['intensity'])
            }
            for seq in seqs
        ]

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Selected sequences saved to: {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--run_id', type=str, default='opr6oq97')
    parser.add_argument('--top_n', type=int, default=15,
                       help='Number of top intensity sequences per emotion')
    parser.add_argument('--project_root', type=str,
                       default='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO')
    args = parser.parse_args()

    project_root = Path(args.project_root)
    ckpt_path = project_root / f"output/moviefmri/{args.run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"

    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*"))[0]

    print(f"{'='*80}")
    print(f"Selecting high-intensity sequences for {args.subject}")
    print(f"Model: {args.run_id}")
    print(f"Top N per emotion: {args.top_n}")
    print(f"{'='*80}")

    # Select sequences
    selected = select_high_intensity_sequences(args.subject, ckpt_path, top_n=args.top_n)

    # Save results
    output_dir = project_root / f"igmap/selected_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.subject}_top{args.top_n}.json"

    save_selected_sequences(selected, output_path)

    # Print summary
    total_sequences = sum(len(seqs) for seqs in selected.values())
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total selected sequences: {total_sequences}")
    print(f"Per emotion: {args.top_n}")
    print(f"\nEstimated IG computation time:")
    print(f"  {total_sequences} sequences × 20 steps × 3 sec = {total_sequences * 20 * 3 / 3600:.1f} hours")
    print(f"{'='*80}")
