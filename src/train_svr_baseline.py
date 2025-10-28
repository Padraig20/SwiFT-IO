"""
Train SVR baseline for emotion prediction

This script:
1. Reuses train/val/test splits from SwiFT-IO data_module
2. Loads 4D fMRI sequences (same as SwiFT-IO)
3. Trains SVR baseline
4. Evaluates on val and test sets
5. Saves results for comparison with SwiFT-IO

Usage:
    python train_svr_baseline.py \
        --image_path /scratch/HBN/9.2.movieDM_SwiFT \
        --downstream_task emotions \
        --sequence_length 30 \
        --dataset_split_seed 777 \
        --output_dir output/svr_baseline
"""

import os
import sys
import argparse
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.data_module import fMRIDataModule
from baselines.svr_baseline import SVRBaseline


def main():
    # Parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Data arguments (same as SwiFT-IO)
    parser.add_argument("--image_path", type=str, required=True, help="Path to fMRI image data")
    parser.add_argument("--dataset_name", type=str, default="HBN", help="Dataset name")
    parser.add_argument("--downstream_task", type=str, default="emotions",
                       help="Task: emotions, contents, features")
    parser.add_argument("--downstream_task_type", type=str, default="regression",
                       help="Task type")
    parser.add_argument("--input_type", type=str, default="movieDM",
                       choices=['movieDM', 'movieTP'], help="Movie type")
    parser.add_argument("--dataset_split_seed", type=int, default=777,
                       help="Random seed for data split (same as SwiFT-IO)")
    parser.add_argument("--stratified_params", nargs="+", default=None, type=str,
                       help="Stratified split parameters")
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Training set proportion")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Validation set proportion")
    parser.add_argument("--bad_subj_path", type=str, default=None,
                       help="Path to bad subjects file")
    parser.add_argument("--adjust_hrf", action='store_true',
                       help="Use HRF-adjusted emotion labels")

    # SVR-specific arguments
    parser.add_argument("--sequence_length", type=int, default=30,
                       help="Length of fMRI sequence (same as SwiFT-IO)")
    parser.add_argument("--kernel", type=str, default="rbf",
                       choices=['linear', 'rbf', 'poly'], help="SVR kernel type")
    parser.add_argument("--C", type=float, default=1.0,
                       help="SVR regularization parameter")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Epsilon in epsilon-SVR (only for regression)")
    parser.add_argument("--standardize", action='store_true', default=True,
                       help="Standardize features")
    parser.add_argument("--task_type", type=str, default="regression",
                       choices=['regression', 'classification'],
                       help="Task type: regression or classification")

    # Data loading arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output/svr_baseline",
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name")

    # Required dummy arguments for data_module compatibility
    parser.add_argument("--decoder", type=str, default="series_decoder")
    parser.add_argument("--num_targets", type=int, default=7)
    parser.add_argument("--with_voxel_norm", action='store_true', default=False)
    parser.add_argument("--shuffle_time_sequence", action='store_true', default=False)
    parser.add_argument("--label_scaling_method", type=str, default="standardization")
    parser.add_argument("--use_contrastive", action='store_true', default=False)
    parser.add_argument("--contrastive_type", type=int, default=0)
    parser.add_argument("--limit_training_samples", type=int, default=None)
    parser.add_argument("--input_offset", type=int, default=0)
    parser.add_argument("--img_size", nargs="+", default=[96, 96, 96, 30], type=int)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SVR Baseline Training")
    print("="*80)
    print(f"Task: {args.downstream_task}")
    print(f"Task type: {args.task_type}")
    print(f"Input type: {args.input_type}")
    print(f"Split seed: {args.dataset_split_seed}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"SVR kernel: {args.kernel}")
    if args.task_type == 'regression':
        print(f"C: {args.C}, epsilon: {args.epsilon}")
    else:
        print(f"C: {args.C}")
    print("="*80)

    # ===== 1. Setup data module (same as SwiFT-IO) =====
    print("\n[Step 1] Setting up data module...")

    # Determine number of emotions/targets
    if args.downstream_task == 'emotions':
        num_emotions = 7
        emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    elif args.downstream_task == 'contents':
        num_emotions = 6
        emotion_names = ['Closeup', 'Body', 'Face', 'NumberCharacters', 'SpokenWords', 'WrittenWords']
    elif args.downstream_task == 'features':
        num_emotions = 8
        emotion_names = ['Brightness', 'SaliencyFraction', 'Sharpness', 'Vibrance',
                        'Loudness', 'Motion', 'Tempo', 'LowLevelChange']
    else:
        raise ValueError(f"Unknown downstream_task: {args.downstream_task}")

    # Create data module
    data_module = fMRIDataModule(**vars(args))
    data_module.setup(stage='fit')

    print(f"Train subjects: {len(data_module.train_dataset.data)}")
    print(f"Val subjects: {len(data_module.val_dataset.data)}")
    print(f"Test subjects: {len(data_module.test_dataset.data)}")

    # ===== 2. Initialize SVR baseline =====
    print("\n[Step 2] Initializing SVR baseline...")

    svr = SVRBaseline(
        num_emotions=num_emotions,
        sequence_length=args.sequence_length,
        kernel=args.kernel,
        C=args.C,
        epsilon=args.epsilon,
        standardize=args.standardize,
        task_type=args.task_type
    )

    # ===== 3. Train SVR =====
    print("\n[Step 3] Training SVR baseline...")

    train_metrics = svr.fit(data_module.train_loader)

    # ===== 4. Evaluate on validation set =====
    print("\n[Step 4] Evaluating on validation set...")

    val_metrics = svr.evaluate(data_module.val_loader, mode='valid')

    # ===== 5. Evaluate on test set =====
    print("\n[Step 5] Evaluating on test set...")

    test_metrics = svr.evaluate(data_module.test_loader, mode='test')

    # ===== 6. Save results =====
    print("\n[Step 6] Saving results...")

    # Combine all metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "svr_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Save model
    model_path = os.path.join(args.output_dir, "svr_model.pkl")
    svr.save(model_path)

    # Save configuration
    config_path = os.path.join(args.output_dir, "svr_config.json")
    config = {
        'task': args.downstream_task,
        'input_type': args.input_type,
        'split_seed': args.dataset_split_seed,
        'sequence_length': args.sequence_length,
        'kernel': args.kernel,
        'C': args.C,
        'epsilon': args.epsilon,
        'standardize': args.standardize,
        'num_emotions': num_emotions,
        'emotion_names': emotion_names,
        'feature_dim': svr.feature_dim,
        'num_train_samples': len(data_module.train_dataset),
        'num_val_samples': len(data_module.val_dataset),
        'num_test_samples': len(data_module.test_dataset)
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # ===== 7. Print summary =====
    print("\n" + "="*80)
    print("SVR Baseline Training Complete!")
    print("="*80)
    print(f"\nFinal Results:")
    if args.task_type == 'regression':
        print(f"  Train MSE: {train_metrics['train_mse']:.4f}")
        print(f"  Train MAE: {train_metrics['train_mae']:.4f}")
        print(f"  Train R2:  {train_metrics['train_r2']:.4f}")
        print(f"\n  Valid MSE: {val_metrics['valid_mse']:.4f}")
        print(f"  Valid MAE: {val_metrics['valid_mae']:.4f}")
        print(f"  Valid R2:  {val_metrics['valid_r2']:.4f}")
        print(f"\n  Test MSE:  {test_metrics['test_mse']:.4f}")
        print(f"  Test MAE:  {test_metrics['test_mae']:.4f}")
        print(f"  Test R2:   {test_metrics['test_r2']:.4f}")
    else:  # classification
        print(f"  Train Acc: {train_metrics['train_acc']:.4f}")
        print(f"  Train F1:  {train_metrics['train_f1']:.4f}")
        print(f"\n  Valid Acc: {val_metrics['valid_acc']:.4f}")
        print(f"  Valid F1:  {val_metrics['valid_f1']:.4f}")
        print(f"\n  Test Acc:  {test_metrics['test_acc']:.4f}")
        print(f"  Test F1:   {test_metrics['test_f1']:.4f}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)

    # Save summary
    summary_path = os.path.join(args.output_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SVR Baseline Training Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Task: {args.downstream_task}\n")
        f.write(f"Task type: {args.task_type}\n")
        f.write(f"Input type: {args.input_type}\n")
        f.write(f"Sequence length: {args.sequence_length}\n")
        f.write(f"Split seed: {args.dataset_split_seed}\n")
        f.write(f"SVR kernel: {args.kernel}\n")
        if args.task_type == 'regression':
            f.write(f"C: {args.C}, epsilon: {args.epsilon}\n\n")
            f.write(f"Final Results:\n")
            f.write(f"  Train MSE: {train_metrics['train_mse']:.4f}\n")
            f.write(f"  Valid MSE: {val_metrics['valid_mse']:.4f}\n")
            f.write(f"  Test MSE:  {test_metrics['test_mse']:.4f}\n")
        else:  # classification
            f.write(f"C: {args.C}\n\n")
            f.write(f"Final Results:\n")
            f.write(f"  Train Acc: {train_metrics['train_acc']:.4f}\n")
            f.write(f"  Valid Acc: {val_metrics['valid_acc']:.4f}\n")
            f.write(f"  Test Acc:  {test_metrics['test_acc']:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
