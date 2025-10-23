"""
Evaluate SVR baseline from checkpoints

This script loads individual checkpoint files and evaluates on train/val/test sets.
Used when training was interrupted before final evaluation.

Usage:
    python src/eval_svr_from_checkpoint.py \
        --checkpoint_dir output/svr_reduction_pca \
        --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
        --dataset_split_seed 777
"""

import os
import sys
import pickle
import json
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.data_module import fMRIDataModule
from baselines.svr_with_reduction import SVRWithReduction


def load_from_checkpoints(checkpoint_dir: str):
    """
    Load SVR model from individual checkpoint files

    Expects:
    - pca_model_checkpoint.pkl (if PCA method)
    - svr_emotion_0_checkpoint.pkl, svr_emotion_1_checkpoint.pkl, ...
    - train_data_checkpoint.pkl (optional, for metadata)
    """
    print(f"\n{'='*80}")
    print(f"Loading SVR model from checkpoints: {checkpoint_dir}")
    print("="*80)

    # Load PCA model
    pca_path = os.path.join(checkpoint_dir, 'pca_model_checkpoint.pkl')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA checkpoint not found: {pca_path}")

    with open(pca_path, 'rb') as f:
        pca_checkpoint = pickle.load(f)

    # Extract PCA model from checkpoint dict
    if isinstance(pca_checkpoint, dict):
        pca_model = pca_checkpoint['pca_model']
        print(f"✓ Loaded PCA model from {pca_path}")
        print(f"  - Variance explained: {pca_checkpoint.get('variance_explained', 'N/A')}")
    else:
        # If it's just the model directly
        pca_model = pca_checkpoint
        print(f"✓ Loaded PCA model from {pca_path}")

    # Find all SVR emotion checkpoint files
    svr_files = sorted([f for f in os.listdir(checkpoint_dir)
                       if f.startswith('svr_emotion_') and f.endswith('_checkpoint.pkl')])

    if not svr_files:
        raise FileNotFoundError(f"No SVR emotion checkpoints found in {checkpoint_dir}")

    num_emotions = len(svr_files)
    print(f"✓ Found {num_emotions} emotion model checkpoints")

    # Load each emotion model and scaler
    models = {}
    scalers = {}

    for i, filename in enumerate(svr_files):
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)

            # Extract model and scaler
            if isinstance(checkpoint, dict):
                models[i] = checkpoint['model']
                scalers[i] = checkpoint['scaler']
            else:
                # If checkpoint is just the model
                models[i] = checkpoint
                scalers[i] = None

            print(f"  ✓ Loaded emotion {i} model")
        except Exception as e:
            print(f"  ✗ Failed to load emotion {i} model: {e}")
            print(f"    Skipping emotion {i}")
            continue

    # Try to load train data checkpoint for metadata
    train_data_path = os.path.join(checkpoint_dir, 'train_data_checkpoint.pkl')
    metadata = {}
    if os.path.exists(train_data_path):
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
            if isinstance(train_data, dict):
                metadata = train_data.get('metadata', {})
        print(f"✓ Loaded metadata from train data checkpoint")

    # Reconstruct SVR object (use actual number of loaded models)
    num_loaded = len(models)
    print(f"\n✓ Successfully loaded {num_loaded} out of {num_emotions} emotion models")

    svr = SVRWithReduction(
        num_emotions=num_loaded,  # Use actual number of loaded models
        sequence_length=metadata.get('sequence_length', 30),
        reduction_method='pca',
        pca_components=metadata.get('pca_components', 100),
        roi_atlas=None,
        kernel=metadata.get('kernel', 'rbf'),
        C=metadata.get('C', 1.0),
        epsilon=metadata.get('epsilon', 0.1),
        standardize=metadata.get('standardize', True),
        use_wandb=False
    )

    # Set loaded components
    svr.pca_model = pca_model
    svr.models = models
    svr.scalers = scalers
    svr.fitted = True
    svr.feature_dim = metadata.get('feature_dim', 3000)

    print(f"\n✓ SVR model reconstructed successfully")
    print(f"  - Num emotions: {num_emotions}")
    print(f"  - Feature dim: {svr.feature_dim}")
    print(f"  - Reduction: {svr.reduction_method}")
    print("="*80)

    return svr


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoint files")

    # Data arguments (same as training script)
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
    parser.add_argument("--dataset_split_seed", type=int, default=777,
                       help="Random seed for data split")
    parser.add_argument("--sequence_length", type=int, default=30,
                       help="Length of fMRI sequence")
    parser.add_argument("--stride_between_seq", type=int, default=1,
                       help="Stride between sequences")
    parser.add_argument("--stride_within_seq", type=int, default=1,
                       help="Stride within sequence")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Training set proportion")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Validation set proportion")
    parser.add_argument("--bad_subj_path", type=str, default=None,
                       help="Path to bad subjects file")
    parser.add_argument("--adjust_hrf", action='store_true',
                       help="Use HRF-adjusted emotion labels")
    parser.add_argument("--stratified_params", nargs="+", default=None, type=str,
                       help="Stratified split parameters")

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

    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for metrics (default: checkpoint_dir/eval_metrics.json)")

    args = parser.parse_args()

    # ===== 1. Load SVR model from checkpoints =====
    svr = load_from_checkpoints(args.checkpoint_dir)

    # ===== 2. Setup data module =====
    print("\n[Step 1] Setting up data module...")

    data_module = fMRIDataModule(**vars(args))

    data_module.setup(stage='fit')

    print(f"\nDataset sizes:")
    print(f"  Train: {len(data_module.train_dataset)}")
    print(f"  Val: {len(data_module.val_dataset)}")
    print(f"  Test: {len(data_module.test_dataset)}")

    # Update emotion names if available
    if hasattr(data_module, 'emotion_names'):
        svr.emotion_names = data_module.emotion_names

    # ===== 3. Evaluate on train set =====
    print("\n[Step 2] Evaluating on training set...")
    train_metrics = svr.evaluate(data_module.train_loader, mode='train')

    # ===== 4. Evaluate on validation set =====
    print("\n[Step 3] Evaluating on validation set...")
    val_metrics = svr.evaluate(data_module.val_loader, mode='valid')

    # ===== 5. Evaluate on test set =====
    print("\n[Step 4] Evaluating on test set...")
    test_metrics = svr.evaluate(data_module.test_loader, mode='test')

    # ===== 6. Save all metrics =====
    print("\n[Step 5] Saving metrics...")

    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # Determine output file
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.checkpoint_dir, 'eval_metrics.json')

    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ All metrics saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Train - MSE: {train_metrics['train_mse']:.4f}, "
          f"MAE: {train_metrics['train_mae']:.4f}, "
          f"R2: {train_metrics['train_r2']:.4f}")
    print(f"Valid - MSE: {val_metrics['valid_mse']:.4f}, "
          f"MAE: {val_metrics['valid_mae']:.4f}, "
          f"R2: {val_metrics['valid_r2']:.4f}")
    print(f"Test  - MSE: {test_metrics['test_mse']:.4f}, "
          f"MAE: {test_metrics['test_mae']:.4f}, "
          f"R2: {test_metrics['test_r2']:.4f}")
    print("="*80)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
