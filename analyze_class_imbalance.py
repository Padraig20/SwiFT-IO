#!/usr/bin/env python3
"""
Analyze class imbalance for seq 20 vs seq 30
Check if longer sequences amplify class imbalance
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.utils.data_module import fMRIDataModule


def analyze_class_distribution(seq_length, split='train'):
    """Analyze class distribution for a given sequence length and split"""

    print(f"\n{'='*80}")
    print(f"Analyzing Seq {seq_length} - {split.upper()} set")
    print(f"{'='*80}\n")

    # Load checkpoint from trained model to get proper hyperparameters
    # Use seq20 run as template (gajr5p1p)
    ckpt_path = Path(f"/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/gajr5p1p")
    ckpt_file = list(ckpt_path.glob("checkpt*"))[0]

    print(f"📦 Loading checkpoint: {ckpt_file.name}")
    ckpt = torch.load(ckpt_file, map_location='cpu')
    args = ckpt['hyper_parameters']

    # Override sequence length
    args['sequence_length'] = seq_length
    args['img_size'] = [96, 96, 96, seq_length]
    args['num_workers'] = 4
    args['eval_batch_size'] = 4
    args['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"

    # Ensure required parameters exist
    if 'dataset_split_seed' not in args:
        args['dataset_split_seed'] = args.get('seed', 777)
    if 'stratified_params' not in args:
        args['stratified_params'] = None
    if 'train_split' not in args:
        args['train_split'] = 0.7
    if 'val_split' not in args:
        args['val_split'] = 0.15
    if 'bad_subj_path' not in args:
        args['bad_subj_path'] = None
    if 'limit_training_samples' not in args:
        args['limit_training_samples'] = 0

    # Initialize data module
    data_module = fMRIDataModule(**args)
    data_module.prepare_data()

    if split == 'train':
        data_module.setup(stage='fit')
        dataset = data_module.train_dataset
    elif split == 'val':
        data_module.setup(stage='fit')
        dataset = data_module.val_dataset
    else:  # test
        data_module.setup(stage='test')
        dataset = data_module.test_dataset

    print(f"Dataset size: {len(dataset)} sequences")

    # Collect all targets
    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    all_targets = []

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 100 == 0:
            print(f"  Processing batch {batch_idx}/{len(loader)}", flush=True)

        targets = batch['target']  # (B, T, 7)
        all_targets.append(targets.cpu().numpy())

    # Concatenate
    all_targets = np.concatenate(all_targets, axis=0)  # (N, T, 7)

    # Flatten
    all_targets_flat = all_targets.reshape(-1, 7)  # (N*T, 7)

    print(f"\n✅ Collected {all_targets_flat.shape[0]} total samples (sequences * time)")
    print(f"   → {len(dataset)} sequences × {seq_length} timesteps = {len(dataset) * seq_length} samples\n")

    # Analyze per emotion
    print(f"{'Emotion':<12} {'Total':<10} {'Positive':<10} {'Negative':<10} {'NaN':<10} {'Pos %':<10} {'Imbalance':<10}")
    print("-"*90)

    results = {}

    for emo_idx, emo_name in enumerate(emotion_names):
        targets = all_targets_flat[:, emo_idx]

        # Count
        total = len(targets)
        n_nan = np.isnan(targets).sum()
        valid_targets = targets[~np.isnan(targets)]
        n_positive = (valid_targets == 1).sum()
        n_negative = (valid_targets == 0).sum()

        pos_ratio = n_positive / len(valid_targets) if len(valid_targets) > 0 else 0
        imbalance_ratio = n_negative / n_positive if n_positive > 0 else float('inf')

        print(f"{emo_name:<12} {total:<10} {n_positive:<10} {n_negative:<10} {n_nan:<10} "
              f"{pos_ratio*100:<9.2f}% {imbalance_ratio:<9.1f}:1")

        results[emo_name] = {
            'total': total,
            'positive': int(n_positive),
            'negative': int(n_negative),
            'nan': int(n_nan),
            'pos_ratio': float(pos_ratio),
            'imbalance_ratio': float(imbalance_ratio),
        }

    return results


def compare_imbalance(seq_lengths=[20, 30], split='train'):
    """Compare class imbalance across different sequence lengths"""

    print("\n" + "="*80)
    print(f"CLASS IMBALANCE COMPARISON - {split.upper()} SET")
    print("="*80)

    results_all = {}

    for seq_len in seq_lengths:
        results_all[seq_len] = analyze_class_distribution(seq_len, split)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Seq 20 vs Seq 30")
    print("="*80)

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    print(f"\n{'Emotion':<12} {'Seq20 Pos%':<12} {'Seq30 Pos%':<12} {'Δ Pos%':<12} "
          f"{'Seq20 Imb':<12} {'Seq30 Imb':<12} {'Δ Imb':<12}")
    print("-"*90)

    for emo_name in emotion_names:
        pos20 = results_all[20][emo_name]['pos_ratio'] * 100
        pos30 = results_all[30][emo_name]['pos_ratio'] * 100
        delta_pos = pos30 - pos20

        imb20 = results_all[20][emo_name]['imbalance_ratio']
        imb30 = results_all[30][emo_name]['imbalance_ratio']
        delta_imb = imb30 - imb20

        worse = "⚠️ WORSE" if delta_imb > 1 else "✓ OK"

        print(f"{emo_name:<12} {pos20:<11.2f}% {pos30:<11.2f}% {delta_pos:+11.2f}% "
              f"{imb20:<11.1f}:1 {imb30:<11.1f}:1 {delta_imb:+11.1f} {worse}")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)

    print("\n1. Imbalance Amplification:")
    print("   If Δ Imb > 1, longer sequences worsen class imbalance")

    print("\n2. Sample Distribution:")
    print("   Seq 20: More sequences, shorter duration")
    print("   Seq 30: Fewer sequences, longer duration")
    print("   Total samples (N*T) should be similar, but distribution changes")

    print("\n3. Expected Pattern:")
    print("   If seq 30 has WORSE imbalance → explains collapse")
    print("   If seq 30 has SIMILAR imbalance → collapse due to other factors")

    print("\n" + "="*80)

    return results_all


if __name__ == "__main__":
    # Compare train set imbalance
    results = compare_imbalance(seq_lengths=[20, 30], split='train')

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
