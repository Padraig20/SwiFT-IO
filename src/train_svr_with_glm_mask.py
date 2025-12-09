"""
Train SVR baseline with GLM significant voxel mask for emotion prediction

This script trains SVR using only GLM-identified significant voxels,
which reduces dimensionality while preserving task-relevant brain regions.

Two reduction modes:
- 'glm_pca': GLM sig voxels -> PCA -> SVR (temporal information preserved)
- 'glm_direct': GLM sig voxels -> time-average -> SVR (simpler baseline)

Usage:
    # GLM PCA + SVR
    python train_svr_with_glm_mask.py \
        --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
        --reduction_mode glm_pca \
        --pca_components 100 \
        --output_dir output/svr_glm_pca_seq20

    # GLM Direct + SVR
    python train_svr_with_glm_mask.py \
        --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
        --reduction_mode glm_direct \
        --output_dir output/svr_glm_direct_seq20
"""

import os
import sys
import argparse
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.data_module import fMRIDataModule
from baselines.svr_with_glm_mask import SVRWithGLMMask

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Data arguments (same as SwiFT-IO)
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to fMRI image data")
    parser.add_argument("--dataset_name", type=str, default="HBN",
                       help="Dataset name")
    parser.add_argument("--downstream_task", type=str, default="emotions",
                       help="Task: emotions, contents, features")
    parser.add_argument("--downstream_task_type", type=str, default="regression",
                       help="Task type")
    parser.add_argument("--input_type", type=str, default="movieDM",
                       choices=['movieDM', 'movieTP'], help="Movie type")
    parser.add_argument("--dataset_split_seed", type=int, default=2,
                       help="Random seed for data split (use 2 for stratified)")
    parser.add_argument("--stratified_params", nargs="+", default=['Age', 'Sex'],
                       type=str, help="Stratified split parameters")
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Training set proportion")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Validation set proportion")
    parser.add_argument("--bad_subj_path", type=str, default=None,
                       help="Path to bad subjects file")
    parser.add_argument("--adjust_hrf", action='store_true',
                       help="Use HRF-adjusted emotion labels")

    # GLM mask SVR arguments
    parser.add_argument("--sequence_length", type=int, default=20,
                       help="Length of fMRI sequence")
    parser.add_argument("--reduction_mode", type=str, default="glm_pca",
                       choices=['glm_pca', 'glm_direct'],
                       help="Reduction mode: glm_pca or glm_direct")
    parser.add_argument("--pca_components", type=int, default=100,
                       help="Number of PCA components (for glm_pca mode)")
    parser.add_argument("--mask_type", type=str, default="union",
                       help="Mask type: union (all emotions combined)")
    parser.add_argument("--glm_mask_dir", type=str, default=None,
                       help="Path to GLM mask directory (default: built-in)")

    # SVR arguments
    parser.add_argument("--kernel", type=str, default="rbf",
                       choices=['linear', 'rbf', 'poly'], help="SVR kernel type")
    parser.add_argument("--C", type=float, default=1.0,
                       help="SVR regularization parameter")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Epsilon in epsilon-SVR")
    parser.add_argument("--standardize", action='store_true', default=True,
                       help="Standardize features")

    # Data loading arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output/svr_glm_mask",
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name")

    # Wandb arguments
    parser.add_argument("--use_wandb", action='store_true', default=False,
                       help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="svr_glm_baseline",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity")

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
    parser.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print(f"SVR with GLM Significant Voxel Mask ({args.reduction_mode} mode)")
    print("="*80)
    print(f"Task: {args.downstream_task}")
    print(f"Input type: {args.input_type}")
    print(f"Split seed: {args.dataset_split_seed}")
    print(f"Stratified params: {args.stratified_params}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Reduction mode: {args.reduction_mode}")
    if args.reduction_mode == 'glm_pca':
        print(f"PCA components: {args.pca_components}")
        print(f"Final feature dim: {args.sequence_length * args.pca_components}")
    else:
        print(f"GLM direct (time-averaged)")
    print(f"Mask type: {args.mask_type}")
    print(f"SVR kernel: {args.kernel}, C: {args.C}, epsilon: {args.epsilon}")
    print("="*80)

    # ===== 1. Setup data module =====
    print("\n[Step 1] Setting up data module...")

    # Determine number of emotions
    if args.downstream_task == 'emotions':
        num_emotions = 7
        emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    else:
        raise ValueError(f"GLM mask is only for emotions task, got: {args.downstream_task}")

    # Create data module
    data_module = fMRIDataModule(**vars(args))
    data_module.setup(stage='fit')

    print(f"Train subjects: {len(data_module.train_dataset.data)}")
    print(f"Val subjects: {len(data_module.val_dataset.data)}")
    print(f"Test subjects: {len(data_module.test_dataset.data)}")

    # Get scaler from data module for non-zero metrics
    scaler = data_module.train_dataset.scaler

    # ===== Initialize wandb =====
    if args.use_wandb and WANDB_AVAILABLE:
        experiment_name = args.experiment_name or f"svr_{args.reduction_mode}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                'reduction_mode': args.reduction_mode,
                'mask_type': args.mask_type,
                'task': args.downstream_task,
                'sequence_length': args.sequence_length,
                'kernel': args.kernel,
                'C': args.C,
                'epsilon': args.epsilon,
                'pca_components': args.pca_components if args.reduction_mode == 'glm_pca' else None,
                'split_seed': args.dataset_split_seed,
                'stratified_params': args.stratified_params,
                'num_train_samples': len(data_module.train_dataset),
                'num_val_samples': len(data_module.val_dataset),
                'num_test_samples': len(data_module.test_dataset)
            }
        )
        print(f"\nWandB initialized: {wandb.run.name}")

    # ===== 2. Initialize SVR with GLM mask =====
    print("\n[Step 2] Initializing SVR with GLM mask...")

    svr = SVRWithGLMMask(
        num_emotions=num_emotions,
        sequence_length=args.sequence_length,
        reduction_mode=args.reduction_mode,
        pca_components=args.pca_components,
        mask_type=args.mask_type,
        glm_mask_dir=args.glm_mask_dir,
        kernel=args.kernel,
        C=args.C,
        epsilon=args.epsilon,
        standardize=args.standardize,
        use_wandb=args.use_wandb
    )

    # ===== 3. Train SVR =====
    print("\n[Step 3] Training SVR with GLM mask...")

    train_metrics = svr.fit(
        data_module.train_loader,
        scaler=scaler,
        output_dir=args.output_dir
    )

    # ===== 4. Evaluate on validation set with non-zero metrics =====
    print("\n[Step 4] Evaluating on validation set...")

    val_metrics = svr.evaluate(data_module.val_loader, mode='valid')

    # ===== 5. Evaluate on test set with non-zero metrics =====
    print("\n[Step 5] Evaluating on test set...")

    test_metrics = svr.evaluate(data_module.test_loader, mode='test')

    # ===== 6. Save results =====
    print("\n[Step 6] Saving results...")

    # Combine all metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "svr_glm_mask_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Save model
    model_path = os.path.join(args.output_dir, "svr_glm_mask_model.pkl")
    svr.save(model_path)

    # Save configuration
    config_path = os.path.join(args.output_dir, "svr_glm_mask_config.json")
    config = {
        'task': args.downstream_task,
        'input_type': args.input_type,
        'split_seed': args.dataset_split_seed,
        'stratified_params': args.stratified_params,
        'sequence_length': args.sequence_length,
        'reduction_mode': args.reduction_mode,
        'mask_type': args.mask_type,
        'pca_components': args.pca_components if args.reduction_mode == 'glm_pca' else None,
        'kernel': args.kernel,
        'C': args.C,
        'epsilon': args.epsilon,
        'standardize': args.standardize,
        'num_emotions': num_emotions,
        'emotion_names': emotion_names,
        'feature_dim': svr.feature_dim,
        'n_masked_voxels': svr.n_masked_voxels,
        'num_train_samples': len(data_module.train_dataset),
        'num_val_samples': len(data_module.val_dataset),
        'num_test_samples': len(data_module.test_dataset)
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # ===== 7. Print summary =====
    print("\n" + "="*80)
    print(f"SVR with GLM Mask Training Complete! ({args.reduction_mode} mode)")
    print("="*80)

    # Print key non-zero metrics
    print(f"\nKey Non-Zero Metrics (Test):")
    avg_nz_mae = test_metrics.get('test_avg_nonzero_mae', 'N/A')
    std_nz_mae = test_metrics.get('test_std_nonzero_mae', 'N/A')
    avg_nz_pearson = test_metrics.get('test_avg_nonzero_pearson', 'N/A')
    std_nz_pearson = test_metrics.get('test_std_nonzero_pearson', 'N/A')

    if isinstance(avg_nz_mae, float):
        print(f"  Avg Non-Zero MAE:     {avg_nz_mae:.4f} +/- {std_nz_mae:.4f}")
        print(f"  Avg Non-Zero Pearson: {avg_nz_pearson:.4f} +/- {std_nz_pearson:.4f}")

    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)

    # Save summary
    summary_path = os.path.join(args.output_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"SVR with GLM Mask Training Summary ({args.reduction_mode} mode)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Task: {args.downstream_task}\n")
        f.write(f"Sequence length: {args.sequence_length}\n")
        f.write(f"Reduction mode: {args.reduction_mode}\n")
        f.write(f"Mask type: {args.mask_type}\n")
        f.write(f"GLM masked voxels: {svr.n_masked_voxels:,}\n")
        f.write(f"Feature dimension: {svr.feature_dim:,}\n\n")
        f.write(f"Key Test Metrics:\n")
        if isinstance(avg_nz_mae, float):
            f.write(f"  Avg Non-Zero MAE: {avg_nz_mae:.4f} +/- {std_nz_mae:.4f}\n")
            f.write(f"  Avg Non-Zero Pearson: {avg_nz_pearson:.4f} +/- {std_nz_pearson:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")

    # ===== Finish wandb =====
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("\nWandB run finished")


if __name__ == "__main__":
    main()
