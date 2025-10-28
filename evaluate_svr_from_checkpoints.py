#!/usr/bin/env python3
"""
Evaluate SVR PCA baseline from saved checkpoints
Calculates train, val, and test performance for all emotions
"""

import os
import sys
import pickle
import argparse
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.module.utils.data_module import fMRIDataModule


def load_checkpoints(checkpoint_dir, num_emotions=7):
    """Load all emotion checkpoints and PCA model"""
    checkpoints = {}

    # Load PCA model
    pca_path = os.path.join(checkpoint_dir, 'pca_model_checkpoint.pkl')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA model not found: {pca_path}")

    with open(pca_path, 'rb') as f:
        pca_checkpoint = pickle.load(f)

    checkpoints['pca'] = pca_checkpoint['pca_model']
    print(f"✓ Loaded PCA model")
    print(f"  - n_components: {pca_checkpoint['pca_components']}")
    print(f"  - variance explained: {pca_checkpoint['variance_explained']:.4f}")

    # Load emotion models
    checkpoints['emotions'] = []
    for e in range(num_emotions):
        emotion_path = os.path.join(checkpoint_dir, f'svr_emotion_{e}_checkpoint.pkl')
        if not os.path.exists(emotion_path):
            print(f"⚠ Warning: Emotion {e} checkpoint not found: {emotion_path}")
            checkpoints['emotions'].append(None)
            continue

        with open(emotion_path, 'rb') as f:
            emotion_checkpoint = pickle.load(f)

        checkpoints['emotions'].append(emotion_checkpoint)
        print(f"✓ Loaded emotion {e} model")

    return checkpoints


def prepare_data_from_dataloader(dataloader, pca, mode='test'):
    """Prepare data by applying PCA reduction to each timepoint (VECTORIZED)"""
    X_list = []
    Y_list = []

    print(f"\nLoading {mode} data with PCA reduction (VECTORIZED - 30x faster)...")
    for batch in tqdm(dataloader, desc=f"Processing {mode}"):
        fmri, targets = batch['fmri_sequence'], batch['target']
        batch_size, _, H, W, D, T = fmri.shape

        # fmri: (batch, 1, H, W, D, T)
        # VECTORIZED: Apply PCA to all timepoints at once

        # Reshape: (batch, 1, H, W, D, T) -> (batch, H, W, D, T)
        fmri = fmri.squeeze(1)  # (batch, H, W, D, T)

        # Transpose to (batch, T, H, W, D) and flatten
        fmri = fmri.permute(0, 4, 1, 2, 3)  # (batch, T, H, W, D)
        fmri_flat = fmri.reshape(batch_size * T, -1).cpu().numpy()  # (batch*T, H*W*D)

        # Apply PCA once to all timepoints
        fmri_pca = pca.transform(fmri_flat)  # (batch*T, n_components)

        # Reshape back: (batch*T, n_components) -> (batch, T*n_components)
        fmri_reduced = fmri_pca.reshape(batch_size, T * pca.n_components)

        # targets: (batch, T, E) -> average over T
        targets_avg = targets.mean(dim=1).cpu().numpy()  # (batch, E)

        X_list.append(fmri_reduced)
        Y_list.append(targets_avg)

    X = np.vstack(X_list)  # (total_samples, T * n_components)
    Y = np.vstack(Y_list)  # (total_samples, num_emotions)

    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"{mode} data shape: X={X.shape}, Y={Y.shape}")
    return X, Y


def evaluate_emotions(X, Y, checkpoints, mode='test'):
    """Evaluate all emotions on given data"""
    num_emotions = len(checkpoints['emotions'])
    predictions = []

    for e in range(num_emotions):
        if checkpoints['emotions'][e] is None:
            print(f"⚠ Skipping emotion {e} (no checkpoint)")
            predictions.append(np.zeros(len(X)))
            continue

        model = checkpoints['emotions'][e]['model']
        scaler = checkpoints['emotions'][e]['scaler']

        # Apply standardization if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Predict
        y_pred = model.predict(X_scaled)
        predictions.append(y_pred)

    # Stack predictions: (num_samples, num_emotions)
    Y_pred = np.column_stack(predictions)

    # Compute metrics
    metrics = {}

    # Overall metrics
    mse_overall = mean_squared_error(Y.flatten(), Y_pred.flatten())
    mae_overall = mean_absolute_error(Y.flatten(), Y_pred.flatten())
    r2_overall = r2_score(Y.flatten(), Y_pred.flatten())

    metrics[f'{mode}_mse'] = float(mse_overall)
    metrics[f'{mode}_mae'] = float(mae_overall)
    metrics[f'{mode}_r2'] = float(r2_overall)

    # Per-emotion metrics
    emotion_names = ['Amusing', 'Anxiety', 'Boring', 'Fearful', 'Pleasant', 'Sad', 'Neutral']

    for e in range(num_emotions):
        y_true = Y[:, e]
        y_pred = Y_pred[:, e]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[f'{mode}_mse_{e}'] = float(mse)
        metrics[f'{mode}_mae_{e}'] = float(mae)
        metrics[f'{mode}_r2_{e}'] = float(r2)

        print(f"  Emotion {e} ({emotion_names[e]}): MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    print(f"\n{mode.upper()} Overall - MSE: {mse_overall:.4f}, MAE: {mae_overall:.4f}, R2: {r2_overall:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate SVR PCA baseline from checkpoints')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='output/svr_reduction_pca',
                       help='Directory containing checkpoints')
    parser.add_argument('--image_path', type=str,
                       default='/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120',
                       help='Path to fMRI data')
    parser.add_argument('--dataset_name', type=str, default='HBN')
    parser.add_argument('--downstream_task', type=str, default='emotions')
    parser.add_argument('--input_type', type=str, default='movieDM')
    parser.add_argument('--dataset_split_seed', type=int, default=777)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file for metrics')

    args = parser.parse_args()

    print("="*80)
    print("SVR PCA Baseline Evaluation from Checkpoints")
    print("="*80)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split seed: {args.dataset_split_seed}")
    print()

    # Load checkpoints
    print("[Step 1] Loading checkpoints...")
    checkpoints = load_checkpoints(args.checkpoint_dir)

    # Setup data module
    print("\n[Step 2] Setting up data module...")
    data_module = fMRIDataModule(
        image_path=args.image_path,
        dataset_name=args.dataset_name,
        dataset_split_num=1,
        dataset_split_seed=args.dataset_split_seed,
        downstream_task=args.downstream_task,
        downstream_task_type='regression',
        input_type=args.input_type,
        sequence_length=args.sequence_length,
        stride_between_seq=1,
        stride_within_seq=1,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_contrastive=False,
        contrastive_type=0,
        with_voxel_norm=False,
        augment_during_training=False,
        shuffle_time_sequence=False,
        label_scaling_method='standardization',
        decoder='series_decoder',
        adjust_hrf=False,
        input_offset=0,
        stratified_params=None,
        train_split=0.7,
        val_split=0.15,
        bad_subj_path=None,
        limit_training_samples=None,
    )
    data_module.setup(stage='fit')

    print(f"\nDataset sizes:")
    print(f"  Train: {len(data_module.train_dataset)}")
    print(f"  Val: {len(data_module.val_dataset)}")
    print(f"  Test: {len(data_module.test_dataset)}")

    # Evaluate on all splits
    all_metrics = {}

    # Train
    print("\n[Step 3] Evaluating on TRAIN set...")
    print("="*80)
    X_train, Y_train = prepare_data_from_dataloader(
        data_module.train_loader,
        checkpoints['pca'],
        mode='train'
    )
    train_metrics = evaluate_emotions(X_train, Y_train, checkpoints, mode='train')
    all_metrics.update(train_metrics)

    # Validation
    print("\n[Step 4] Evaluating on VALIDATION set...")
    print("="*80)
    X_val, Y_val = prepare_data_from_dataloader(
        data_module.val_loader,
        checkpoints['pca'],
        mode='val'
    )
    val_metrics = evaluate_emotions(X_val, Y_val, checkpoints, mode='val')
    all_metrics.update(val_metrics)

    # Test
    print("\n[Step 5] Evaluating on TEST set...")
    print("="*80)
    X_test, Y_test = prepare_data_from_dataloader(
        data_module.test_loader,
        checkpoints['pca'],
        mode='test'
    )
    test_metrics = evaluate_emotions(X_test, Y_test, checkpoints, mode='test')
    all_metrics.update(test_metrics)

    # Save metrics
    if args.output_file is None:
        args.output_file = os.path.join(args.checkpoint_dir, 'evaluation_metrics.json')

    print(f"\n[Step 6] Saving metrics to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"TRAIN  - MSE: {all_metrics['train_mse']:.4f}, MAE: {all_metrics['train_mae']:.4f}, R2: {all_metrics['train_r2']:.4f}")
    print(f"VAL    - MSE: {all_metrics['val_mse']:.4f}, MAE: {all_metrics['val_mae']:.4f}, R2: {all_metrics['val_r2']:.4f}")
    print(f"TEST   - MSE: {all_metrics['test_mse']:.4f}, MAE: {all_metrics['test_mae']:.4f}, R2: {all_metrics['test_r2']:.4f}")
    print("="*80)
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
