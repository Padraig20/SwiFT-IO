#!/usr/bin/env python3
"""
Evaluate Time-Averaged + PCA SVR baseline from saved checkpoints
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

    print(f"Loading PCA model from {pca_path}...")
    with open(pca_path, 'rb') as f:
        pca_checkpoint = pickle.load(f)
        checkpoints['pca'] = pca_checkpoint['pca_model']
    print(f"✓ PCA model loaded (n_components={checkpoints['pca'].n_components})")

    # Load emotion SVR models
    checkpoints['emotions'] = []
    for e in range(num_emotions):
        emotion_path = os.path.join(checkpoint_dir, f'svr_emotion_{e}_checkpoint.pkl')
        if not os.path.exists(emotion_path):
            raise FileNotFoundError(f"Emotion {e} model not found: {emotion_path}")

        with open(emotion_path, 'rb') as f:
            emotion_checkpoint = pickle.load(f)

        checkpoints['emotions'].append(emotion_checkpoint)
        print(f"✓ Loaded emotion {e} model")

    return checkpoints


def prepare_data_from_dataloader(dataloader, pca, mode='test'):
    """
    Prepare time-averaged data with PCA reduction

    Unlike sequence-based PCA which applies PCA to each timepoint,
    this applies PCA to time-averaged fMRI volumes.
    """
    X_list = []
    Y_list = []
    skipped_count = 0

    print(f"\nLoading {mode} data with time-averaging + PCA reduction...")
    for batch in tqdm(dataloader, desc=f"Processing {mode}"):
        try:
            fmri, targets = batch['fmri_sequence'], batch['target']
            batch_size, _, H, W, D, T = fmri.shape

            # fmri: (batch, 1, H, W, D, T)
            # Step 1: Average over time dimension
            fmri = fmri.squeeze(1)  # (batch, H, W, D, T)
            fmri_avg = fmri.mean(dim=-1)  # (batch, H, W, D)

            # Step 2: Flatten spatial dimensions
            fmri_flat = fmri_avg.reshape(batch_size, -1).cpu().numpy()  # (batch, H*W*D)

            # Step 3: Apply PCA
            fmri_pca = pca.transform(fmri_flat)  # (batch, n_components)

            # targets: (batch, T, E) -> average over T
            targets_avg = targets.mean(dim=1).cpu().numpy()  # (batch, E)

            X_list.append(fmri_pca)
            Y_list.append(targets_avg)
        except Exception as e:
            skipped_count += 1
            print(f"\nError processing batch: {e}")
            import traceback
            traceback.print_exc()
            raise  # Raise immediately to see the error

    X = np.vstack(X_list)  # (total_samples, n_components)
    Y = np.vstack(Y_list)  # (total_samples, num_emotions)

    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"{mode} data shape: X={X.shape}, Y={Y.shape}")
    return X, Y


def evaluate_emotions(X, Y, checkpoints, emotion_names, mode='test'):
    """Evaluate all emotions"""
    num_emotions = len(checkpoints['emotions'])
    results = {
        'per_emotion': [],
        'overall': {}
    }

    all_mse = []
    all_mae = []
    all_r2 = []

    print(f"\n{'='*80}")
    print(f"{mode.upper()} SET EVALUATION")
    print(f"{'='*80}")

    for e in range(num_emotions):
        model = checkpoints['emotions'][e]['model']
        scaler = checkpoints['emotions'][e]['scaler']
        emotion_name = emotion_names[e]

        # Get true targets for this emotion
        y_true = Y[:, e]

        # Predict
        y_pred = model.predict(X)

        # Denormalize predictions and targets
        y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_denorm = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_true_denorm, y_pred_denorm)
        mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
        r2 = r2_score(y_true_denorm, y_pred_denorm)

        # Correlation
        corr = np.corrcoef(y_true_denorm, y_pred_denorm)[0, 1]

        emotion_results = {
            'emotion_idx': e,
            'emotion_name': emotion_name,
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'correlation': float(corr)
        }
        results['per_emotion'].append(emotion_results)

        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)

        print(f"Emotion {e} ({emotion_name}):")
        print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}")

    # Overall metrics
    overall_mse = np.mean(all_mse)
    overall_mae = np.mean(all_mae)
    overall_r2 = np.mean(all_r2)

    results['overall'] = {
        'mse': float(overall_mse),
        'mae': float(overall_mae),
        'r2': float(overall_r2)
    }

    print(f"\n{'='*80}")
    print(f"OVERALL {mode.upper()} METRICS:")
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  R²: {overall_r2:.4f}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Time-Averaged + PCA SVR models')

    # Checkpoint directory
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing SVR and PCA checkpoints')

    # Data arguments
    parser.add_argument("--image_path", type=str,
                       default='/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120',
                       help="Path to fMRI data")
    parser.add_argument("--dataset_name", type=str, default="HBN")
    parser.add_argument("--downstream_task", type=str, default="emotions")
    parser.add_argument("--downstream_task_type", type=str, default="regression")
    parser.add_argument("--input_type", type=str, default="movieDM")
    parser.add_argument("--dataset_split_seed", type=int, default=777)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    # Dummy arguments for data module compatibility
    parser.add_argument("--decoder", type=str, default="lstm_series_regression_head")
    parser.add_argument("--num_targets", type=int, default=7)
    parser.add_argument("--with_voxel_norm", action='store_true', default=False)
    parser.add_argument("--shuffle_time_sequence", action='store_true', default=False)
    parser.add_argument("--label_scaling_method", type=str, default="standardization")
    parser.add_argument("--use_contrastive", action='store_true', default=False)
    parser.add_argument("--contrastive_type", type=int, default=0)
    parser.add_argument("--limit_training_samples", type=int, default=None)
    parser.add_argument("--input_offset", type=int, default=0)
    parser.add_argument("--img_size", nargs="+", default=[96, 96, 96, 30], type=int)
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--stratified_params", nargs="+", default=None)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--bad_subj_path", type=str, default=None)
    parser.add_argument("--adjust_hrf", action='store_true', default=False)
    parser.add_argument("--augment_during_training", action='store_true', default=False)

    args = parser.parse_args()

    print("="*80)
    print("Time-Averaged + PCA SVR Evaluation")
    print("="*80)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split seed: {args.dataset_split_seed}")
    print()

    # Define emotion names
    emotion_names = ['Amusing', 'Anxiety', 'Boring', 'Fearful', 'Pleasant', 'Sad', 'Neutral']

    # Load checkpoints
    print("[Step 1] Loading checkpoints...")
    checkpoints = load_checkpoints(args.checkpoint_dir)
    print()

    # Setup data module
    print("[Step 2] Setting up data module...")
    data_module = fMRIDataModule(
        image_path=args.image_path,
        dataset_name=args.dataset_name,
        dataset_split_num=1,
        dataset_split_seed=args.dataset_split_seed,
        downstream_task=args.downstream_task,
        downstream_task_type=args.downstream_task_type,
        input_type=args.input_type,
        sequence_length=args.sequence_length,
        stride_between_seq=args.stride_between_seq,
        stride_within_seq=args.stride_within_seq,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        use_contrastive=args.use_contrastive,
        contrastive_type=args.contrastive_type,
        with_voxel_norm=args.with_voxel_norm,
        augment_during_training=args.augment_during_training,
        shuffle_time_sequence=args.shuffle_time_sequence,
        label_scaling_method=args.label_scaling_method,
        decoder=args.decoder,
        adjust_hrf=args.adjust_hrf,
        input_offset=args.input_offset,
        stratified_params=args.stratified_params,
        train_split=args.train_split,
        val_split=args.val_split,
        bad_subj_path=args.bad_subj_path,
        limit_training_samples=args.limit_training_samples,
    )
    data_module.setup(stage='fit')

    print(f"Train set: {len(data_module.train_dataset)} samples")
    print(f"Val set: {len(data_module.val_dataset)} samples")
    print(f"Test set: {len(data_module.test_dataset)} samples")
    print()

    # Use data_module's dataloaders (same as training script)
    # These use the default PyTorch collate function
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    test_loader = data_module.test_loader

    pca = checkpoints['pca']

    # Evaluate on all splits
    all_results = {}

    print("[Step 3] Evaluating on train set...")
    X_train, Y_train = prepare_data_from_dataloader(train_loader, pca, mode='train')
    train_results = evaluate_emotions(X_train, Y_train, checkpoints, emotion_names, mode='train')
    all_results['train'] = train_results

    print("[Step 4] Evaluating on validation set...")
    X_val, Y_val = prepare_data_from_dataloader(val_loader, pca, mode='val')
    val_results = evaluate_emotions(X_val, Y_val, checkpoints, emotion_names, mode='val')
    all_results['val'] = val_results

    print("[Step 5] Evaluating on test set...")
    X_test, Y_test = prepare_data_from_dataloader(test_loader, pca, mode='test')
    test_results = evaluate_emotions(X_test, Y_test, checkpoints, emotion_names, mode='test')
    all_results['test'] = test_results

    # Save results
    results_path = os.path.join(args.checkpoint_dir, 'evaluation_results.json')
    print(f"\n[Step 6] Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✓ Results saved!")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
