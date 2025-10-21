"""
SVR (Support Vector Regression) with Dimensionality Reduction for fMRI emotion prediction

Three dimensionality reduction strategies:
1. PCA: Reduce each timepoint's spatial dims with PCA
2. ROI: Use brain atlas to average voxels into ROIs
3. Time-averaged: Average across time to get single static pattern

This provides comparison with standard SVR and SwiFT-IO using different feature engineering approaches.

Author: For comparison with SwiFT-IO
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from tqdm import tqdm
import torch
import pickle
import json

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import multiprocessing

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


class SVRWithReduction:
    """
    SVR (Support Vector Regression) baseline with dimensionality reduction
    for emotion prediction from 4D fMRI sequences

    Three reduction strategies:
    - 'pca': Apply PCA to each timepoint, concat across time
    - 'roi': Use ROI averaging (atlas-based)
    - 'time_avg': Average across time dimension

    This is designed for regression tasks (emotion prediction).
    """

    def __init__(self,
                 num_emotions: int = 7,
                 sequence_length: int = 30,
                 reduction_method: Literal['pca', 'roi', 'time_avg'] = 'time_avg',
                 pca_components: int = 100,
                 roi_atlas: str = 'aal',
                 roi_timeseries_path: str = '/scratch/HBN/9.2.movieDM_ROI_timeseries',
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 standardize: bool = True,
                 use_wandb: bool = False,
                 emotion_names: List[str] = None):
        """
        Args:
            num_emotions: Number of emotions to predict
            sequence_length: Length of fMRI sequence (default: 30)
            reduction_method: Dimensionality reduction method
                - 'pca': PCA on each timepoint -> concat
                - 'roi': ROI-based averaging
                - 'time_avg': Time-averaged (simplest)
            pca_components: Number of PCA components per timepoint (for 'pca' method)
            roi_atlas: Atlas name for ROI extraction (for 'roi' method)
            roi_timeseries_path: Path to precomputed ROI timeseries CSVs
            kernel: SVR kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR
            standardize: Whether to standardize features
            use_wandb: Whether to log to wandb
            emotion_names: List of emotion names for better logging
        """
        from sklearn.svm import SVR

        self.num_emotions = num_emotions
        self.sequence_length = sequence_length
        self.reduction_method = reduction_method
        self.pca_components = pca_components
        self.roi_atlas = roi_atlas
        self.roi_timeseries_path = roi_timeseries_path
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.standardize = standardize
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.emotion_names = emotion_names or [f'emotion_{i}' for i in range(num_emotions)]

        # Create models: one SVR per emotion (since this is regression)
        # Note: SVR itself doesn't support n_jobs, but we can use joblib for parallel training
        self.models = [
            SVR(kernel=kernel, C=C, epsilon=epsilon, cache_size=1000, verbose=False)
            for _ in range(num_emotions)
        ]

        # Scalers for feature standardization (one per emotion)
        self.scalers = [StandardScaler() for _ in range(num_emotions)] if standardize else None

        # Dimensionality reduction components
        if reduction_method == 'pca':
            # Use IncrementalPCA for memory efficiency (batch-wise fitting)
            self.pca_model = IncrementalPCA(n_components=pca_components, batch_size=1000)
            print(f"Initialized IncrementalPCA model with {pca_components} components")
            print(f"Using batch_size=1000 for memory efficiency")
            print(f"Final feature dimension: {sequence_length * pca_components}")
        elif reduction_method == 'roi':
            # Use precomputed ROI timeseries from CSV files
            self.roi_cache = {}  # Cache loaded ROI timeseries by subject
            self.num_rois = None  # Will be set when first CSV is loaded
            self.common_roi_columns = None  # Will store common ROI column names
            print(f"ROI-based reduction using precomputed timeseries")
            print(f"  Path: {roi_timeseries_path}")
            print(f"  Feature dimension will be: {sequence_length} timepoints × num_ROIs (determined from CSV)")
        elif reduction_method == 'time_avg':
            print("Time-averaged reduction (single static brain pattern)")
        else:
            raise ValueError(f"Unknown reduction_method: {reduction_method}")

        self.fitted = False
        self.feature_dim = None  # Will be set after processing first sample

    def find_common_roi_columns(self, use_all_files: bool = True, cache_file: str = None) -> List[str]:
        """
        Find common ROI columns across all subjects

        Args:
            use_all_files: If True, check ALL CSV files (recommended for robustness)
            cache_file: Path to cache file for storing/loading common ROI list

        Returns:
            List of common ROI column names (excluding 'TR')
        """
        import glob

        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"\n  Loading common ROIs from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                common_columns_sorted = cached_data['common_roi_columns']
                print(f"  Loaded {len(common_columns_sorted)} common ROIs from cache")
                return common_columns_sorted

        # Get all CSV files
        csv_files = sorted(glob.glob(os.path.join(self.roi_timeseries_path, '*.csv')))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.roi_timeseries_path}")

        # Use all files for robustness
        files_to_check = csv_files

        print(f"\n  Finding common ROIs across ALL {len(files_to_check)} subjects...")
        print(f"  This may take a minute, but ensures no KeyError during training...")

        # Collect column sets from each file with progress bar
        all_column_sets = []
        for filepath in tqdm(files_to_check, desc="  Scanning CSV files", ncols=80):
            df = pd.read_csv(filepath, nrows=1)  # Only read header, not entire file
            cols = set(df.columns) - {'TR'}
            all_column_sets.append(cols)

        # Find intersection
        common_columns = set.intersection(*all_column_sets)
        common_columns_sorted = sorted(list(common_columns))

        print(f"\n  Found {len(common_columns_sorted)} common ROIs across ALL subjects")
        print(f"  Sample ROIs: {common_columns_sorted[:5]}")

        # Save to cache if requested
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {
                'common_roi_columns': common_columns_sorted,
                'num_subjects_checked': len(files_to_check),
                'num_common_rois': len(common_columns_sorted)
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"  Saved common ROI list to cache: {cache_file}")

        return common_columns_sorted

    def load_roi_timeseries(self, subject_id: str) -> np.ndarray:
        """
        Load precomputed ROI timeseries for a subject
        Only uses common ROI columns to ensure consistent feature dimensions

        Args:
            subject_id: Subject ID (may include 'sub-' prefix or not)

        Returns:
            ROI timeseries array (num_TRs, num_common_ROIs)
        """
        # Check cache first
        if subject_id in self.roi_cache:
            return self.roi_cache[subject_id]

        # Find common ROI columns on first load
        if self.common_roi_columns is None:
            cache_file = os.path.join(self.roi_timeseries_path, '_common_roi_cache.json')
            self.common_roi_columns = self.find_common_roi_columns(
                use_all_files=True,
                cache_file=cache_file
            )
            self.num_rois = len(self.common_roi_columns)
            print(f"  Using {self.num_rois} common ROIs across all subjects")

        # Construct filename (add 'sub-' prefix if not already present)
        if not subject_id.startswith('sub-'):
            filename = f"sub-{subject_id}_movieDM_roi_temporal_activity.csv"
        else:
            filename = f"{subject_id}_movieDM_roi_temporal_activity.csv"
        filepath = os.path.join(self.roi_timeseries_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ROI timeseries not found: {filepath}")

        # Load CSV
        roi_data = pd.read_csv(filepath)

        # Select only common ROI columns (in consistent order)
        roi_timeseries = roi_data[self.common_roi_columns].values  # (750, num_common_ROIs)

        # Cache it
        self.roi_cache[subject_id] = roi_timeseries

        return roi_timeseries

    def reduce_features(self, fmri_seq: np.ndarray, subject_id: Optional[str] = None,
                       start_frame: Optional[int] = None) -> np.ndarray:
        """
        Apply dimensionality reduction to fMRI sequence

        Note: PCA must be fitted before calling this method (via fit_pca_on_train_data)

        Args:
            fmri_seq: (seq_len, 96, 96, 96) or (96, 96, 96, seq_len) array
            subject_id: Subject ID (required for 'roi' method, without 'sub-' prefix)
            start_frame: Starting frame index in full timeseries (required for 'roi' method)

        Returns:
            Reduced features
        """
        if self.reduction_method == 'roi':
            # ROI method: use precomputed timeseries
            if subject_id is None or start_frame is None:
                raise ValueError("subject_id and start_frame required for ROI reduction")

            # Load full ROI timeseries for this subject (750 TRs, 106 ROIs)
            roi_timeseries = self.load_roi_timeseries(subject_id)

            # Extract the sequence slice
            end_frame = start_frame + self.sequence_length
            roi_sequence = roi_timeseries[start_frame:end_frame, :]  # (seq_len, num_ROIs)

            # Flatten to single feature vector: (seq_len * num_ROIs,)
            features = roi_sequence.flatten()

            return features

        # For PCA and time_avg, we use the voxel data
        # Ensure shape is (seq_len, 96, 96, 96)
        if fmri_seq.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {fmri_seq.shape}")

        if fmri_seq.shape[-1] < fmri_seq.shape[0]:
            # (96, 96, 96, seq_len) -> (seq_len, 96, 96, 96)
            fmri_seq = np.transpose(fmri_seq, (3, 0, 1, 2))

        seq_len = fmri_seq.shape[0]

        if self.reduction_method == 'pca':
            # Apply single PCA to all timepoints
            reduced_timepoints = []

            for t in range(seq_len):
                # Flatten spatial dimensions: (96*96*96,)
                frame_flat = fmri_seq[t].reshape(-1)

                # Transform using fitted PCA
                reduced = self.pca_model.transform(frame_flat.reshape(1, -1))
                reduced_timepoints.append(reduced.flatten())

            # Concatenate across time: (seq_len * pca_components,)
            features = np.concatenate(reduced_timepoints)

        elif self.reduction_method == 'time_avg':
            # Time-averaged: simply average across time
            avg_frame = fmri_seq.mean(axis=0)  # (96, 96, 96)
            features = avg_frame.flatten()  # (96*96*96,)

        else:
            raise ValueError(f"Unknown reduction_method: {self.reduction_method}")

        return features

    def fit_pca_on_train_data(self, dataloader, output_dir: str = None):
        """
        Fit IncrementalPCA on all training data using batch-wise processing

        Memory-efficient: Processes data batch-by-batch without loading all into memory
        Uses IncrementalPCA.partial_fit() to incrementally learn PCA components

        Args:
            dataloader: Training dataloader
            output_dir: Directory to save PCA checkpoint (optional)
        """
        if self.reduction_method != 'pca':
            return

        # Check if PCA checkpoint already exists
        pca_checkpoint_path = os.path.join(output_dir, 'pca_model_checkpoint.pkl') if output_dir else None

        if pca_checkpoint_path and os.path.exists(pca_checkpoint_path):
            print("\n" + "="*80)
            print(f"Loading PCA model from checkpoint: {pca_checkpoint_path}")
            print("="*80)
            with open(pca_checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.pca_model = checkpoint['pca_model']
                total_frames = checkpoint['total_frames']
                variance_explained = checkpoint['variance_explained']
            print(f"Loaded PCA model fitted on {total_frames:,} timepoints")
            print(f"Variance explained: {variance_explained:.2%}")
            print("="*80)
            return

        print("\n" + "="*80)
        print("Fitting IncrementalPCA on training data (MEMORY-EFFICIENT)...")
        print("="*80)

        total_frames = 0

        # Process each batch incrementally
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Fitting PCA incrementally")):
            fmri_data = batch['fmri_sequence'].numpy()  # (B, 1, 96, 96, 96, S)

            # Remove channel dimension
            if fmri_data.ndim == 6 and fmri_data.shape[1] == 1:
                fmri_data = fmri_data.squeeze(1)

            batch_size = fmri_data.shape[0]

            # Collect frames from this batch only (temporary)
            batch_frames = []
            for b in range(batch_size):
                # Get sequence: (96, 96, 96, seq_len)
                fmri_seq = fmri_data[b]

                # Transpose to (seq_len, 96, 96, 96)
                if fmri_seq.shape[-1] < fmri_seq.shape[0]:
                    fmri_seq = np.transpose(fmri_seq, (3, 0, 1, 2))

                seq_len = fmri_seq.shape[0]

                # Flatten each timepoint
                for t in range(seq_len):
                    frame_flat = fmri_seq[t].reshape(-1)  # (884736,)
                    batch_frames.append(frame_flat)

            # Stack frames from this batch
            if batch_frames:
                X_batch = np.stack(batch_frames, axis=0)

                # Incrementally fit PCA on this batch
                self.pca_model.partial_fit(X_batch)

                total_frames += len(batch_frames)

                # Free memory
                del batch_frames
                del X_batch

        print(f"\nIncrementalPCA fitted on {total_frames:,} timepoints")

        # Report variance explained
        variance_explained = self.pca_model.explained_variance_ratio_.sum()
        print(f"Variance explained: {variance_explained:.2%}")
        print("="*80)

        # Save PCA checkpoint immediately after fitting
        if pca_checkpoint_path:
            print(f"\nSaving PCA model checkpoint to: {pca_checkpoint_path}")
            pca_checkpoint = {
                'pca_model': self.pca_model,
                'total_frames': total_frames,
                'variance_explained': variance_explained,
                'pca_components': self.pca_components
            }
            with open(pca_checkpoint_path, 'wb') as f:
                pickle.dump(pca_checkpoint, f)
            print(f"PCA checkpoint saved! (Size: {os.path.getsize(pca_checkpoint_path) / 1024**2:.1f} MB)")
            print("="*80)

    def _train_single_emotion(self, e: int, X_train: np.ndarray, Y_train: np.ndarray) -> Dict:
        """
        Train SVR for a single emotion (helper for parallel training)

        Args:
            e: Emotion index
            X_train: Training features (num_samples, feature_dim)
            Y_train: Training targets (num_samples, num_emotions)

        Returns:
            Dictionary with model, scaler, and metrics
        """
        y_train = Y_train[:, e]

        # Standardize features
        if self.standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            scaler = None
            X_train_scaled = X_train

        # Fit SVR
        self.models[e].fit(X_train_scaled, y_train)

        # Training predictions
        y_pred = self.models[e].predict(X_train_scaled)

        # Metrics
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        return {
            'emotion_idx': e,
            'model': self.models[e],
            'scaler': scaler,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def prepare_data_from_dataloader(self, dataloader, mode: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from PyTorch dataloader with dimensionality reduction

        Args:
            dataloader: SwiFT-IO dataloader
            mode: 'train', 'valid', or 'test'

        Returns:
            X: (total_samples, reduced_dim) features
            Y: (total_samples, num_emotions) targets
        """
        X_list = []
        Y_list = []

        print(f"\nLoading {mode} data with {self.reduction_method} reduction...")

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # batch['fmri_sequence']: (batch_size, 1, 96, 96, 96, seq_len)
            # batch['target']: (batch_size, seq_len, num_emotions)
            # batch['subject_name']: list of subject names
            # batch['TR']: list of start frame indices

            fmri_data = batch['fmri_sequence'].numpy()  # (B, 1, 96, 96, 96, S)
            target_data = batch['target'].numpy()  # (B, S, E)
            subject_names = batch['subject_name']  # List of subject IDs
            start_frames = batch['TR']  # List of start frame indices (or tensor)

            # Convert start_frames to numpy if it's a tensor
            if isinstance(start_frames, torch.Tensor):
                start_frames = start_frames.numpy()

            # Print shape on first batch for debugging
            if batch_idx == 0:
                print(f"  First batch fMRI shape: {fmri_data.shape}")
                print(f"  First batch target shape: {target_data.shape}")
                if self.reduction_method == 'roi':
                    print(f"  Subject names: {subject_names[:3]}...")
                    print(f"  Start frames: {start_frames[:3]}...")

            # Remove channel dimension: (B, 1, X, Y, Z, S) -> (B, X, Y, Z, S)
            if fmri_data.ndim == 6 and fmri_data.shape[1] == 1:
                fmri_data = fmri_data.squeeze(1)
            elif fmri_data.ndim == 5:
                pass
            else:
                raise ValueError(f"Unexpected fMRI data shape: {fmri_data.shape}")

            batch_size = fmri_data.shape[0]

            for b in range(batch_size):
                # Get single sequence: (96, 96, 96, seq_len)
                fmri_seq = fmri_data[b]

                # Get subject info for ROI method
                subject_id = subject_names[b] if self.reduction_method == 'roi' else None
                start_frame = int(start_frames[b]) if self.reduction_method == 'roi' else None

                # Apply dimensionality reduction
                features = self.reduce_features(fmri_seq, subject_id=subject_id, start_frame=start_frame)

                # Get targets: for time-averaged, we also average targets
                targets = target_data[b]  # (seq_len, num_emotions)

                if self.reduction_method == 'time_avg':
                    # Average targets across time
                    targets = targets.mean(axis=0)  # (num_emotions,)
                    X_list.append(features)
                    Y_list.append(targets)
                else:
                    # For PCA and ROI, each timepoint is a separate sample
                    # But we need to predict the middle timepoint (or mean)
                    # Let's use the mean target for simplicity
                    targets_mean = targets.mean(axis=0)  # (num_emotions,)
                    X_list.append(features)
                    Y_list.append(targets_mean)

        # Stack all samples
        X = np.stack(X_list, axis=0)  # (total_samples, reduced_dim)
        Y = np.stack(Y_list, axis=0)  # (total_samples, num_emotions)

        print(f"{mode} data shape: X={X.shape}, Y={Y.shape}")

        # Set feature dimension from first batch
        if self.feature_dim is None:
            self.feature_dim = X.shape[1]
            print(f"Feature dimension: {self.feature_dim:,}")

        return X, Y

    def fit(self, train_dataloader, output_dir: str = None) -> Dict[str, float]:
        """
        Fit SVR models on training data with checkpoint saving

        Args:
            train_dataloader: PyTorch dataloader for training set
            output_dir: Directory to save checkpoints (optional)

        Returns:
            Training metrics
        """
        print("\n" + "="*80)
        print(f"Training SVR with Dimensionality Reduction ({self.reduction_method} method)")
        print("="*80)

        # Step 1: Fit PCA on all training data (if using PCA)
        self.fit_pca_on_train_data(train_dataloader, output_dir=output_dir)

        # Step 2: Prepare training data (transform with fitted PCA)
        # Check if checkpoint exists
        data_checkpoint_path = os.path.join(output_dir, 'train_data_checkpoint.pkl') if output_dir else None

        if data_checkpoint_path and os.path.exists(data_checkpoint_path):
            print(f"\n  Loading training data from checkpoint: {data_checkpoint_path}")
            with open(data_checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                X_train = checkpoint['X_train']
                Y_train = checkpoint['Y_train']
                self.feature_dim = checkpoint['feature_dim']
            print(f"  Loaded X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        else:
            X_train, Y_train = self.prepare_data_from_dataloader(train_dataloader, mode='train')

            # Save data checkpoint
            if data_checkpoint_path:
                print(f"\n  Saving training data checkpoint to: {data_checkpoint_path}")
                checkpoint = {
                    'X_train': X_train,
                    'Y_train': Y_train,
                    'feature_dim': self.feature_dim
                }
                with open(data_checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"  Checkpoint saved! (Size: {os.path.getsize(data_checkpoint_path) / 1024**2:.1f} MB)")

        # Train models for each emotion
        train_metrics = {}

        # Get number of available CPUs
        n_jobs = multiprocessing.cpu_count()
        print(f"\nTraining SVR models for each emotion using {n_jobs} CPUs in parallel...")

        # Parallel training for all emotions
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._train_single_emotion)(e, X_train, Y_train)
            for e in range(self.num_emotions)
        )

        # Process results and update models/scalers
        for result in results:
            e = result['emotion_idx']
            self.models[e] = result['model']
            if self.standardize:
                self.scalers[e] = result['scaler']

            mse = result['mse']
            mae = result['mae']
            r2 = result['r2']

            train_metrics[f'train_mse_{e}'] = mse
            train_metrics[f'train_mae_{e}'] = mae
            train_metrics[f'train_r2_{e}'] = r2

            print(f"  Emotion {e} ({self.emotion_names[e]}): MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    f'train/{self.emotion_names[e]}/mse': mse,
                    f'train/{self.emotion_names[e]}/mae': mae,
                    f'train/{self.emotion_names[e]}/r2': r2,
                    'train/emotion_progress': (e + 1) / self.num_emotions
                })

            # Save checkpoint after each emotion
            if output_dir:
                emotion_checkpoint_path = os.path.join(output_dir, f'svr_emotion_{e}_checkpoint.pkl')
                checkpoint = {
                    'emotion_idx': e,
                    'model': self.models[e],
                    'scaler': self.scalers[e] if self.standardize else None,
                    'metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    }
                }
                with open(emotion_checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)

                # Also save intermediate metrics
                intermediate_metrics_path = os.path.join(output_dir, f'train_metrics_up_to_emotion_{e}.json')
                current_metrics = {k: v for k, v in train_metrics.items()}
                with open(intermediate_metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=2)

        self.fitted = True

        # Overall metrics
        train_metrics['train_mse'] = np.mean([train_metrics[f'train_mse_{e}']
                                               for e in range(self.num_emotions)])
        train_metrics['train_mae'] = np.mean([train_metrics[f'train_mae_{e}']
                                               for e in range(self.num_emotions)])
        train_metrics['train_r2'] = np.mean([train_metrics[f'train_r2_{e}']
                                              for e in range(self.num_emotions)])

        print(f"\nOverall Training - MSE: {train_metrics['train_mse']:.4f}, "
              f"MAE: {train_metrics['train_mae']:.4f}, R2: {train_metrics['train_r2']:.4f}")
        print("="*80)

        # Log overall training metrics to wandb
        if self.use_wandb:
            wandb.log({
                'train/overall_mse': train_metrics['train_mse'],
                'train/overall_mae': train_metrics['train_mae'],
                'train/overall_r2': train_metrics['train_r2']
            })

        return train_metrics

    def evaluate(self, dataloader, mode: str = 'test') -> Dict[str, float]:
        """
        Evaluate SVR on validation or test data

        Args:
            dataloader: PyTorch dataloader
            mode: 'valid' or 'test'

        Returns:
            Evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        print(f"\n{'='*80}")
        print(f"Evaluating on {mode} set")
        print("="*80)

        # Prepare data
        X_eval, Y_eval = self.prepare_data_from_dataloader(dataloader, mode=mode)

        # Predict for each emotion in parallel
        def predict_single_emotion(e):
            if self.standardize:
                X_eval_scaled = self.scalers[e].transform(X_eval)
            else:
                X_eval_scaled = X_eval
            return self.models[e].predict(X_eval_scaled)

        n_jobs = multiprocessing.cpu_count()
        all_predictions = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(predict_single_emotion)(e)
            for e in range(self.num_emotions)
        )

        Y_pred = np.stack(all_predictions, axis=1)  # (num_samples, num_emotions)

        # Compute metrics
        metrics = {}

        # Overall metrics
        mse_overall = mean_squared_error(Y_eval.flatten(), Y_pred.flatten())
        mae_overall = mean_absolute_error(Y_eval.flatten(), Y_pred.flatten())
        r2_overall = r2_score(Y_eval.flatten(), Y_pred.flatten())

        metrics[f'{mode}_mse'] = mse_overall
        metrics[f'{mode}_mae'] = mae_overall
        metrics[f'{mode}_r2'] = r2_overall

        # Per-emotion metrics
        for e in range(self.num_emotions):
            y_true_e = Y_eval[:, e]
            y_pred_e = Y_pred[:, e]

            mse_e = mean_squared_error(y_true_e, y_pred_e)
            mae_e = mean_absolute_error(y_true_e, y_pred_e)
            r2_e = r2_score(y_true_e, y_pred_e)

            # Pearson correlation
            if len(y_true_e) > 1:
                corr_e, _ = pearsonr(y_true_e, y_pred_e)
            else:
                corr_e = 0.0

            metrics[f'{mode}_mse_{e}'] = mse_e
            metrics[f'{mode}_mae_{e}'] = mae_e
            metrics[f'{mode}_r2_{e}'] = r2_e
            metrics[f'{mode}_corrcoef_{e}'] = corr_e

            print(f"  Emotion {e} ({self.emotion_names[e]}): MSE={mse_e:.4f}, MAE={mae_e:.4f}, "
                  f"R2={r2_e:.4f}, Corr={corr_e:.4f}")

            # Log per-emotion metrics to wandb
            if self.use_wandb:
                wandb.log({
                    f'{mode}/{self.emotion_names[e]}/mse': mse_e,
                    f'{mode}/{self.emotion_names[e]}/mae': mae_e,
                    f'{mode}/{self.emotion_names[e]}/r2': r2_e,
                    f'{mode}/{self.emotion_names[e]}/corr': corr_e
                })

        print(f"\nOverall {mode} - MSE: {mse_overall:.4f}, MAE: {mae_overall:.4f}, "
              f"R2: {r2_overall:.4f}")
        print("="*80)

        # Log overall metrics to wandb
        if self.use_wandb:
            wandb.log({
                f'{mode}/overall_mse': mse_overall,
                f'{mode}/overall_mae': mae_overall,
                f'{mode}/overall_r2': r2_overall
            })

        return metrics

    def save(self, save_path: str):
        """Save SVR models, scalers, and reduction components"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            'models': self.models,
            'scalers': self.scalers,
            'num_emotions': self.num_emotions,
            'sequence_length': self.sequence_length,
            'reduction_method': self.reduction_method,
            'pca_components': self.pca_components,
            'roi_atlas': self.roi_atlas,
            'roi_timeseries_path': self.roi_timeseries_path,
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'standardize': self.standardize,
            'feature_dim': self.feature_dim
        }

        # Add reduction-specific components
        if self.reduction_method == 'pca':
            state['pca_model'] = self.pca_model
        elif self.reduction_method == 'roi':
            state['num_rois'] = self.num_rois
            state['common_roi_columns'] = self.common_roi_columns

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"\nSVR model saved to {save_path}")

    def load(self, load_path: str):
        """Load SVR models, scalers, and reduction components"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.models = state['models']
        self.scalers = state['scalers']
        self.num_emotions = state['num_emotions']
        self.sequence_length = state['sequence_length']
        self.reduction_method = state['reduction_method']
        self.pca_components = state['pca_components']
        self.roi_atlas = state['roi_atlas']
        self.roi_timeseries_path = state.get('roi_timeseries_path', '/scratch/HBN/9.2.movieDM_ROI_timeseries')
        self.kernel = state['kernel']
        self.C = state['C']
        self.epsilon = state['epsilon']
        self.standardize = state['standardize']
        self.feature_dim = state['feature_dim']

        # Load reduction-specific components
        if self.reduction_method == 'pca':
            self.pca_model = state['pca_model']
        elif self.reduction_method == 'roi':
            self.roi_cache = {}  # Reset cache on load
            self.num_rois = state['num_rois']
            self.common_roi_columns = state.get('common_roi_columns', None)

        self.fitted = True

        print(f"\nSVR model loaded from {load_path}")
