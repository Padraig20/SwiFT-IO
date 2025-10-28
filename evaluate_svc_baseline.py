#!/usr/bin/env python3
"""
Evaluate SVC Baseline from saved checkpoint

This script loads a trained SVC baseline model and evaluates it on test set.

Usage:
    python evaluate_svc_baseline.py \
        --model_path output/svc_baseline_seq20/svr_model.pkl \
        --output_dir output/svc_baseline_seq20/evaluation
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from module.utils.data_module import fMRIDataModule
from baselines.svr_baseline import SVRBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved SVC model (.pkl)")
    parser.add_argument("--image_path", type=str,
                       default="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
                       help="Path to fMRI data")
    parser.add_argument("--dataset_split_seed", type=int, default=2,
                       help="Dataset split seed")
    parser.add_argument("--sequence_length", type=int, default=20,
                       help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of workers")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for evaluation results")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SVC Baseline Evaluation")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Load model
    print("\n[Step 1] Loading SVC model...")
    svc = SVRBaseline(num_emotions=7)  # Initialize with default params
    svc.load(args.model_path)

    print(f"Model loaded successfully!")
    print(f"  Task type: {svc.task_type}")
    print(f"  Num emotions: {svc.num_emotions}")
    print(f"  Sequence length: {svc.sequence_length}")
    print(f"  Kernel: {svc.kernel}")
    print(f"  Feature dim: {svc.feature_dim:,}")

    # Setup data module
    print("\n[Step 2] Setting up data module...")

    data_args = {
        'image_path': args.image_path,
        'dataset_name': 'HBN',
        'downstream_task': 'emotions',
        'downstream_task_type': 'classification',
        'input_type': 'movieDM',
        'dataset_split_seed': args.dataset_split_seed,
        'sequence_length': args.sequence_length,
        'batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'eval_num_workers': args.num_workers,
        'stride_between_seq': 1,
        'stride_within_seq': 1,
        'decoder': 'series_decoder',
        'num_targets': 7,
        'with_voxel_norm': False,
        'shuffle_time_sequence': False,
        'label_scaling_method': 'standardization',
        'use_contrastive': False,
        'contrastive_type': 0,
        'limit_training_samples': None,
        'input_offset': 0,
        'img_size': [96, 96, 96, args.sequence_length],
        'train_split': 0.7,
        'val_split': 0.15,
        'bad_subj_path': None,
    }

    data_module = fMRIDataModule(**data_args)
    data_module.setup(stage='test')

    print(f"Test set size: {len(data_module.test_dataset)} samples")

    # Evaluate on test set
    print("\n[Step 3] Evaluating on test set...")
    test_metrics = svc.evaluate(data_module.test_loader, mode='test')

    # Save results
    print("\n[Step 4] Saving results...")

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Results saved to: {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    if svc.task_type == 'classification':
        print(f"\nOverall Test Performance:")
        print(f"  Accuracy:  {test_metrics['test_acc']:.4f}")
        print(f"  F1 Score:  {test_metrics['test_f1']:.4f}")
        print(f"  Precision: {test_metrics['test_precision']:.4f}")
        print(f"  Recall:    {test_metrics['test_recall']:.4f}")

        print(f"\nPer-Emotion Performance:")
        for e, name in enumerate(emotion_names):
            print(f"  {name:8s}: Acc={test_metrics[f'test_acc_{e}']:.4f}, "
                  f"F1={test_metrics[f'test_f1_{e}']:.4f}")
    else:  # regression
        print(f"\nOverall Test Performance:")
        print(f"  MSE: {test_metrics['test_mse']:.4f}")
        print(f"  MAE: {test_metrics['test_mae']:.4f}")
        print(f"  R²:  {test_metrics['test_r2']:.4f}")

        print(f"\nPer-Emotion Performance:")
        for e, name in enumerate(emotion_names):
            print(f"  {name:8s}: MSE={test_metrics[f'test_mse_{e}']:.4f}, "
                  f"R²={test_metrics[f'test_r2_{e}']:.4f}")

    print("="*80)

    # Save summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SVC Baseline Evaluation Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Task type: {svc.task_type}\n")
        f.write(f"Test samples: {len(data_module.test_dataset)}\n\n")

        if svc.task_type == 'classification':
            f.write(f"Overall Test Performance:\n")
            f.write(f"  Accuracy:  {test_metrics['test_acc']:.4f}\n")
            f.write(f"  F1 Score:  {test_metrics['test_f1']:.4f}\n\n")
            f.write(f"Per-Emotion Performance:\n")
            for e, name in enumerate(emotion_names):
                f.write(f"  {name}: Acc={test_metrics[f'test_acc_{e}']:.4f}, "
                       f"F1={test_metrics[f'test_f1_{e}']:.4f}\n")
        else:
            f.write(f"Overall Test Performance:\n")
            f.write(f"  MSE: {test_metrics['test_mse']:.4f}\n")
            f.write(f"  R²:  {test_metrics['test_r2']:.4f}\n\n")
            f.write(f"Per-Emotion Performance:\n")
            for e, name in enumerate(emotion_names):
                f.write(f"  {name}: MSE={test_metrics[f'test_mse_{e}']:.4f}, "
                       f"R²={test_metrics[f'test_r2_{e}']:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
