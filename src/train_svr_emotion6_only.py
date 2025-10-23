"""
Train only emotion 6 for SVR PCA baseline

This script loads existing PCA model and trains only emotion 6 SVR model.
Used to complete training after emotion 6 checkpoint was corrupted due to disk space.

Usage:
    python src/train_svr_emotion6_only.py \
        --checkpoint_dir output/svr_reduction_pca \
        --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120
"""

import os
import sys
import pickle
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.data_module import fMRIDataModule
from baselines.svr_with_reduction import SVRWithReduction


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing PCA checkpoint")

    # Data arguments
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to fMRI image data")
    parser.add_argument("--dataset_name", type=str, default="HBN")
    parser.add_argument("--downstream_task", type=str, default="emotions")
    parser.add_argument("--downstream_task_type", type=str, default="regression")
    parser.add_argument("--input_type", type=str, default="movieDM")
    parser.add_argument("--dataset_split_seed", type=int, default=777)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--bad_subj_path", type=str, default=None)
    parser.add_argument("--adjust_hrf", action='store_true', default=False)
    parser.add_argument("--stratified_params", nargs="+", default=None, type=str)

    # SVR parameters
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pca_components", type=int, default=100)

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

    print("="*80)
    print("Training Emotion 6 Only for SVR PCA Baseline")
    print("="*80)

    # ===== 1. Load PCA model =====
    print("\n[Step 1] Loading PCA model from checkpoint...")

    pca_path = os.path.join(args.checkpoint_dir, 'pca_model_checkpoint.pkl')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA checkpoint not found: {pca_path}")

    with open(pca_path, 'rb') as f:
        pca_checkpoint = pickle.load(f)

    if isinstance(pca_checkpoint, dict):
        pca_model = pca_checkpoint['pca_model']
        print(f"✓ Loaded PCA model")
        print(f"  - Variance explained: {pca_checkpoint.get('variance_explained', 'N/A')}")
    else:
        pca_model = pca_checkpoint
        print(f"✓ Loaded PCA model")

    # ===== 2. Setup data module =====
    print("\n[Step 2] Setting up data module...")

    data_module = fMRIDataModule(**vars(args))
    data_module.setup(stage='fit')

    print(f"\nDataset sizes:")
    print(f"  Train: {len(data_module.train_dataset)}")
    print(f"  Val: {len(data_module.val_dataset)}")
    print(f"  Test: {len(data_module.test_dataset)}")

    # ===== 3. Initialize SVR =====
    print("\n[Step 3] Initializing SVR for emotion 6...")

    svr = SVRWithReduction(
        num_emotions=7,
        sequence_length=args.sequence_length,
        reduction_method='pca',
        pca_components=args.pca_components,
        roi_atlas=None,
        kernel=args.kernel,
        C=args.C,
        epsilon=args.epsilon,
        standardize=True,
        use_wandb=False
    )

    # Set the loaded PCA model
    svr.pca_model = pca_model
    svr.feature_dim = args.sequence_length * args.pca_components

    print(f"✓ SVR initialized with pre-fitted PCA model")
    print(f"  - Feature dim: {svr.feature_dim}")

    # ===== 4. Prepare training data =====
    print("\n[Step 4] Preparing training data...")

    X_train, Y_train = svr.prepare_data_from_dataloader(
        data_module.train_loader,
        mode='train'
    )

    print(f"✓ Training data prepared")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - Y_train shape: {Y_train.shape}")

    # ===== 5. Train emotion 6 only =====
    print("\n[Step 5] Training emotion 6 SVR model...")

    emotion_idx = 6
    result = svr._train_single_emotion(emotion_idx, X_train, Y_train)

    print(f"\n✓ Emotion 6 training completed!")
    print(f"  - Train MSE: {result['mse']:.4f}")
    print(f"  - Train MAE: {result['mae']:.4f}")
    print(f"  - Train R²: {result['r2']:.4f}")

    # Save the model and scaler
    svr.models[emotion_idx] = result['model']
    svr.scalers[emotion_idx] = result['scaler']

    # ===== 6. Save checkpoint =====
    print("\n[Step 6] Saving emotion 6 checkpoint...")

    checkpoint_path = os.path.join(args.checkpoint_dir, f'svr_emotion_{emotion_idx}_checkpoint.pkl')
    checkpoint = {
        'model': result['model'],
        'scaler': result['scaler']
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    checkpoint_size = os.path.getsize(checkpoint_path) / 1024**2
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"  - Size: {checkpoint_size:.1f} MB")

    # Save metrics
    metrics_path = os.path.join(args.checkpoint_dir, 'train_metrics_emotion_6.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_mse_6': result['mse'],
            'train_mae_6': result['mae'],
            'train_r2_6': result['r2']
        }, f, indent=2)

    print(f"✓ Metrics saved: {metrics_path}")

    print("\n" + "="*80)
    print("✓ Emotion 6 training completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
