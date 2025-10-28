#!/usr/bin/env python3
"""
Train SVR on Time-Averaged fMRI data with PCA dimensionality reduction

This script:
1. Loads pre-saved time-averaged training data checkpoint
2. Applies PCA to reduce 884,736 voxels → N components
3. Trains SVR models for emotion prediction
4. Evaluates on validation and test sets
5. Saves checkpoints for later evaluation

This is the optimal approach combining:
- Time averaging (temporal reduction)
- PCA (spatial reduction)
- SVR (simple, interpretable baseline)

Usage:
    python train_svr_time_avg_with_pca.py \
        --checkpoint_path output/svr_reduction_time_avg_linear/train_data_checkpoint.pkl \
        --pca_components 100 \
        --output_dir output/svr_time_avg_pca_100 \
        --kernel rbf
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from module.utils.data_module import fMRIDataModule


def train_single_emotion(emotion_idx, X_train, Y_train, kernel='rbf', C=1.0, epsilon=0.1,
                         standardize=True, emotion_name='emotion'):
    """Train SVR for a single emotion"""
    print(f"\n  Training emotion {emotion_idx} ({emotion_name})...")

    # Get targets for this emotion
    y_train = Y_train[:, emotion_idx]

    # Standardize features
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = X_train

    # Train SVR
    model = SVR(kernel=kernel, C=C, epsilon=epsilon, cache_size=1000, verbose=False)
    model.fit(X_scaled, y_train)

    # Evaluate on train set
    y_pred = model.predict(X_scaled)

    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    return {
        'emotion_idx': emotion_idx,
        'emotion_name': emotion_name,
        'model': model,
        'scaler': scaler,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }


def prepare_data_from_dataloader(dataloader, pca_model, mode='val'):
    """Prepare validation/test data with time-avg + PCA reduction"""
    X_list = []
    Y_list = []

    print(f"\nLoading {mode} data with time-avg + PCA reduction...")
    for batch in tqdm(dataloader, desc=f"Processing {mode}"):
        fmri = batch['fmri_sequence']  # (batch, 1, 96, 96, 96, T)
        targets = batch['target']  # (batch, T, E)

        batch_size = fmri.shape[0]

        # Time-average: mean over T dimension
        fmri_avg = fmri.mean(dim=-1).squeeze(1)  # (batch, 96, 96, 96)

        # Flatten to (batch, 884736)
        fmri_flat = fmri_avg.reshape(batch_size, -1).cpu().numpy()

        # Apply PCA
        fmri_reduced = pca_model.transform(fmri_flat)

        # Average targets over time
        targets_avg = targets.mean(dim=1).cpu().numpy()  # (batch, E)

        X_list.append(fmri_reduced)
        Y_list.append(targets_avg)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"{mode} data shape: X={X.shape}, Y={Y.shape}")
    return X, Y


def evaluate_emotions(X, Y, models, scalers, emotion_names, mode='test'):
    """Evaluate all emotions on given data"""
    num_emotions = len(models)
    predictions = []

    for e in range(num_emotions):
        model = models[e]
        scaler = scalers[e]

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
    for e in range(num_emotions):
        y_true = Y[:, e]
        y_pred = Y_pred[:, e]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Correlation
        if len(np.unique(y_pred)) > 1:
            corr, _ = pearsonr(y_true, y_pred)
        else:
            corr = 0.0

        metrics[f'{mode}_mse_{e}'] = float(mse)
        metrics[f'{mode}_mae_{e}'] = float(mae)
        metrics[f'{mode}_r2_{e}'] = float(r2)
        metrics[f'{mode}_corrcoef_{e}'] = float(corr)

        print(f"  Emotion {e} ({emotion_names[e]}): MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Corr={corr:.4f}")

    print(f"\n{mode.upper()} Overall - MSE: {mse_overall:.4f}, MAE: {mae_overall:.4f}, R2: {r2_overall:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train SVR on Time-Averaged + PCA fMRI data')

    # Data arguments
    parser.add_argument('--checkpoint_path', type=str,
                       default='output/svr_reduction_time_avg_linear/train_data_checkpoint.pkl',
                       help='Path to time-averaged training data checkpoint')
    parser.add_argument('--image_path', type=str,
                       default='/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120',
                       help='Path to fMRI data (for val/test)')
    parser.add_argument('--dataset_name', type=str, default='HBN')
    parser.add_argument('--downstream_task', type=str, default='emotions')
    parser.add_argument('--input_type', type=str, default='movieDM')
    parser.add_argument('--dataset_split_seed', type=int, default=777)
    parser.add_argument('--sequence_length', type=int, default=30)

    # PCA arguments
    parser.add_argument('--pca_components', type=int, default=100,
                       help='Number of PCA components (100 or 1000 recommended)')
    parser.add_argument('--pca_batch_size', type=int, default=1000,
                       help='Batch size for IncrementalPCA')

    # SVR arguments
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly'])
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--standardize', action='store_true', default=True)

    # Data loading
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    # Output
    parser.add_argument('--output_dir', type=str, default='output/svr_time_avg_pca')

    # Required for data_module
    parser.add_argument('--decoder', type=str, default='series_decoder')
    parser.add_argument('--num_targets', type=int, default=7)
    parser.add_argument('--with_voxel_norm', action='store_true', default=False)
    parser.add_argument('--shuffle_time_sequence', action='store_true', default=False)
    parser.add_argument('--label_scaling_method', type=str, default='standardization')
    parser.add_argument('--use_contrastive', action='store_true', default=False)
    parser.add_argument('--contrastive_type', type=int, default=0)
    parser.add_argument('--limit_training_samples', type=int, default=None)
    parser.add_argument('--input_offset', type=int, default=0)
    parser.add_argument('--img_size', nargs='+', default=[96, 96, 96, 30], type=int)
    parser.add_argument('--stride_between_seq', type=int, default=1)
    parser.add_argument('--stride_within_seq', type=int, default=1)
    parser.add_argument('--downstream_task_type', type=str, default='regression')
    parser.add_argument('--stratified_params', nargs='+', default=None, type=str)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--bad_subj_path', type=str, default=None)
    parser.add_argument('--adjust_hrf', action='store_true', default=False)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SVR Training: Time-Averaged + PCA Reduction")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"PCA components: {args.pca_components}")
    print(f"SVR kernel: {args.kernel}")
    print(f"C: {args.C}, epsilon: {args.epsilon}")
    print("="*80)

    # Emotion names
    emotion_names = ['Amusing', 'Anxiety', 'Boring', 'Fearful', 'Pleasant', 'Sad', 'Neutral']
    num_emotions = 7

    # ===== Step 1: Load time-averaged training data =====
    print("\n[Step 1] Loading time-averaged training data checkpoint...")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    with open(args.checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    X_train_raw = checkpoint['X_train']  # (11344, 884736)
    Y_train = checkpoint['Y_train']  # (11344, 7)

    print(f"Loaded X_train: {X_train_raw.shape}, dtype: {X_train_raw.dtype}")
    print(f"Loaded Y_train: {Y_train.shape}")
    print(f"Original feature dimension: {X_train_raw.shape[1]:,} voxels")
    print(f"Memory: {X_train_raw.nbytes / 1024**3:.2f} GB")

    # Convert to float32 if needed
    if X_train_raw.dtype == np.float16:
        print("Converting X_train from float16 to float32 for PCA...")
        X_train_raw = X_train_raw.astype(np.float32)

    # ===== Step 2: Apply PCA to training data =====
    print(f"\n[Step 2] Applying PCA ({args.pca_components} components)...")

    pca_checkpoint_path = os.path.join(args.output_dir, 'pca_model_checkpoint.pkl')

    if os.path.exists(pca_checkpoint_path):
        print(f"Loading PCA model from checkpoint: {pca_checkpoint_path}")
        with open(pca_checkpoint_path, 'rb') as f:
            pca_checkpoint = pickle.load(f)
            pca_model = pca_checkpoint['pca_model']
            X_train = pca_checkpoint['X_train_pca']
        print(f"Loaded PCA model and transformed data")
        print(f"Variance explained: {pca_checkpoint['variance_explained']:.4f}")
    else:
        pca_model = IncrementalPCA(n_components=args.pca_components,
                                   batch_size=args.pca_batch_size)

        # Fit PCA in batches
        print("Fitting PCA in batches...")
        n_samples = X_train_raw.shape[0]
        for start in tqdm(range(0, n_samples, args.pca_batch_size), desc="PCA fitting"):
            end = min(start + args.pca_batch_size, n_samples)
            batch = X_train_raw[start:end]
            pca_model.partial_fit(batch)

        # Transform training data
        print("Transforming training data...")
        X_train = pca_model.transform(X_train_raw)

        variance_explained = pca_model.explained_variance_ratio_.sum()
        print(f"PCA complete! Variance explained: {variance_explained:.4f}")
        print(f"Reduced feature dimension: {X_train.shape[1]}")
        print(f"Memory: {X_train.nbytes / 1024**2:.1f} MB")

        # Save PCA checkpoint
        print(f"\nSaving PCA checkpoint to: {pca_checkpoint_path}")
        pca_checkpoint = {
            'pca_model': pca_model,
            'X_train_pca': X_train,
            'pca_components': args.pca_components,
            'variance_explained': variance_explained,
            'original_feature_dim': X_train_raw.shape[1]
        }
        with open(pca_checkpoint_path, 'wb') as f:
            pickle.dump(pca_checkpoint, f)
        print(f"PCA checkpoint saved!")

    # Clear raw data to save memory
    del X_train_raw

    # ===== Step 3: Train SVR for each emotion =====
    print("\n[Step 3] Training SVR models for each emotion...")

    n_jobs = min(int(os.environ.get('SLURM_CPUS_PER_TASK',
                                     os.environ.get('SLURM_CPUS_ON_NODE',
                                                    multiprocessing.cpu_count()))),
                 num_emotions, 3)

    print(f"Using {n_jobs} parallel workers...")

    # Parallel training
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_emotion)(
            e, X_train, Y_train,
            kernel=args.kernel, C=args.C, epsilon=args.epsilon,
            standardize=args.standardize, emotion_name=emotion_names[e]
        )
        for e in range(num_emotions)
    )

    # Collect models and metrics
    models = [None] * num_emotions
    scalers = [None] * num_emotions
    train_metrics = {}

    for result in results:
        e = result['emotion_idx']
        models[e] = result['model']
        scalers[e] = result['scaler']

        train_metrics[f'train_mse_{e}'] = result['mse']
        train_metrics[f'train_mae_{e}'] = result['mae']
        train_metrics[f'train_r2_{e}'] = result['r2']

        print(f"  Emotion {e} ({result['emotion_name']}): MSE={result['mse']:.4f}, "
              f"MAE={result['mae']:.4f}, R2={result['r2']:.4f}")

        # Save emotion checkpoint
        emotion_checkpoint_path = os.path.join(args.output_dir, f'svr_emotion_{e}_checkpoint.pkl')
        checkpoint = {
            'emotion_idx': e,
            'model': models[e],
            'scaler': scalers[e],
            'metrics': {
                'mse': result['mse'],
                'mae': result['mae'],
                'r2': result['r2']
            }
        }
        with open(emotion_checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    # Overall train metrics
    train_metrics['train_mse'] = np.mean([train_metrics[f'train_mse_{e}'] for e in range(num_emotions)])
    train_metrics['train_mae'] = np.mean([train_metrics[f'train_mae_{e}'] for e in range(num_emotions)])
    train_metrics['train_r2'] = np.mean([train_metrics[f'train_r2_{e}'] for e in range(num_emotions)])

    print(f"\nOverall Training - MSE: {train_metrics['train_mse']:.4f}, "
          f"MAE: {train_metrics['train_mae']:.4f}, R2: {train_metrics['train_r2']:.4f}")

    # ===== Step 4: Setup data module for val/test =====
    print("\n[Step 4] Setting up data module for validation/test...")

    data_module = fMRIDataModule(**vars(args))
    data_module.setup(stage='fit')

    print(f"Val subjects: {len(data_module.val_dataset)}")
    print(f"Test subjects: {len(data_module.test_dataset)}")

    # ===== Step 5: Evaluate on validation set =====
    print("\n[Step 5] Evaluating on VALIDATION set...")
    print("="*80)

    X_val, Y_val = prepare_data_from_dataloader(data_module.val_loader, pca_model, mode='val')
    val_metrics = evaluate_emotions(X_val, Y_val, models, scalers, emotion_names, mode='valid')

    # ===== Step 6: Evaluate on test set =====
    print("\n[Step 6] Evaluating on TEST set...")
    print("="*80)

    X_test, Y_test = prepare_data_from_dataloader(data_module.test_loader, pca_model, mode='test')
    test_metrics = evaluate_emotions(X_test, Y_test, models, scalers, emotion_names, mode='test')

    # ===== Step 7: Save all metrics =====
    print("\n[Step 7] Saving results...")

    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    metrics_path = os.path.join(args.output_dir, 'svr_time_avg_pca_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    config = {
        'method': 'time_avg_pca',
        'original_checkpoint': args.checkpoint_path,
        'pca_components': args.pca_components,
        'variance_explained': float(pca_model.explained_variance_ratio_.sum()),
        'kernel': args.kernel,
        'C': args.C,
        'epsilon': args.epsilon,
        'standardize': args.standardize,
        'num_emotions': num_emotions,
        'emotion_names': emotion_names,
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0]
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # ===== Summary =====
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nMethod: Time-Averaged + PCA ({args.pca_components} components)")
    print(f"Variance explained: {pca_model.explained_variance_ratio_.sum():.4f}")
    print(f"\nFinal Results:")
    print(f"  TRAIN  - MSE: {train_metrics['train_mse']:.4f}, MAE: {train_metrics['train_mae']:.4f}, R2: {train_metrics['train_r2']:.4f}")
    print(f"  VALID  - MSE: {val_metrics['valid_mse']:.4f}, MAE: {val_metrics['valid_mae']:.4f}, R2: {val_metrics['valid_r2']:.4f}")
    print(f"  TEST   - MSE: {test_metrics['test_mse']:.4f}, MAE: {test_metrics['test_mae']:.4f}, R2: {test_metrics['test_r2']:.4f}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
