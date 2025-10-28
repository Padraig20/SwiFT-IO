"""
SVR (Support Vector Regression) Baseline for fMRI emotion prediction

Uses the same input (4D fMRI sequences) as SwiFT-IO but with nilearn's SVR decoder.
This provides a fair comparison by using identical input data.

Key differences from SwiFT-IO:
- Model: SVR (sklearn-based) vs Swin4D Transformer
- Training: Scikit-learn fit() vs PyTorch backpropagation
- Features: Flattened voxels vs learned hierarchical representations

Author: For comparison with SwiFT-IO
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
import pickle
import json

from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr
from nilearn.maskers import NiftiMasker
import nibabel as nib


class SVRBaseline:
    """
    SVR-based baseline for emotion prediction from 4D fMRI sequences

    This model:
    1. Loads 4D fMRI sequences (same as SwiFT-IO)
    2. Flattens spatial dimensions to create feature vectors
    3. Trains one SVR model per emotion
    4. Predicts emotion timeseries

    Input: (batch, 96, 96, 96, seq_len) 4D fMRI volumes
    Output: (batch, seq_len, num_emotions) emotion predictions
    """

    def __init__(self,
                 num_emotions: int = 7,
                 sequence_length: int = 30,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 standardize: bool = True,
                 use_masking: bool = False,
                 cache_dir: str = None,
                 task_type: str = 'regression'):
        """
        Args:
            num_emotions: Number of emotions to predict
            sequence_length: Length of fMRI sequence (default: 30)
            kernel: SVR kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR (only for regression)
            standardize: Whether to standardize features
            use_masking: Whether to use brain masking (requires mask file)
            cache_dir: Directory to cache processed data
            task_type: 'regression' or 'classification'
        """
        self.num_emotions = num_emotions
        self.sequence_length = sequence_length
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.standardize = standardize
        self.use_masking = use_masking
        self.cache_dir = cache_dir
        self.task_type = task_type

        # Create models: one SVR/SVC per emotion
        if task_type == 'classification':
            self.models = [
                SVC(kernel=kernel, C=C, cache_size=1000, verbose=False, probability=True)
                for _ in range(num_emotions)
            ]
        else:  # regression
            self.models = [
                SVR(kernel=kernel, C=C, epsilon=epsilon, cache_size=1000, verbose=False)
                for _ in range(num_emotions)
            ]

        # Scalers for feature standardization (one per emotion)
        self.scalers = [StandardScaler() for _ in range(num_emotions)] if standardize else None

        # Brain masker for spatial masking (optional)
        self.masker = None
        if use_masking:
            print("Warning: use_masking=True but masker not initialized. Call set_masker() with a mask file.")

        self.fitted = False
        self.feature_dim = None  # Will be set after loading first sample

    def set_masker(self, mask_img_path: str):
        """Set brain masker from a mask file"""
        self.masker = NiftiMasker(mask_img=mask_img_path, standardize=False)
        print(f"Brain masker loaded from {mask_img_path}")

    def load_fmri_sequence(self, subject_path: str, start_frame: int,
                          sequence_length: int, stride_within_seq: int = 1) -> Optional[np.ndarray]:
        """
        Load 4D fMRI sequence from .pt files (same as SwiFT-IO dataset)

        Args:
            subject_path: Path to subject directory with frame_*.pt files
            start_frame: Starting frame index
            sequence_length: Number of frames to load
            stride_within_seq: Stride within sequence

        Returns:
            fMRI data: (96, 96, 96, seq_len) or None if files not found
        """
        frames = []

        for frame_idx in range(start_frame, start_frame + sequence_length, stride_within_seq):
            frame_path = os.path.join(subject_path, f'frame_{frame_idx}.pt')

            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return None

            # Load frame (should be 96x96x96)
            frame = torch.load(frame_path, weights_only=True).numpy()
            frames.append(frame)

        # Stack to (96, 96, 96, seq_len)
        fmri_seq = np.stack(frames, axis=-1)

        return fmri_seq

    def flatten_sequence(self, fmri_seq: np.ndarray) -> np.ndarray:
        """
        Flatten 4D fMRI sequence to feature vector

        Args:
            fmri_seq: (96, 96, 96, seq_len) or (seq_len, 96, 96, 96) array

        Returns:
            Flattened features: (seq_len, num_voxels)
        """
        # Check shape and transpose if needed
        if fmri_seq.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {fmri_seq.shape}")

        # If shape is (96, 96, 96, seq_len), transpose to (seq_len, 96, 96, 96)
        if fmri_seq.shape[-1] < fmri_seq.shape[0]:
            # Last dimension is smallest -> it's likely seq_len
            fmri_seq = np.transpose(fmri_seq, (3, 0, 1, 2))

        # Flatten spatial dimensions: (seq_len, 96*96*96)
        num_frames = fmri_seq.shape[0]
        features = fmri_seq.reshape(num_frames, -1)

        return features

    def prepare_data_from_dataloader(self, dataloader, mode: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from PyTorch dataloader

        Args:
            dataloader: SwiFT-IO dataloader
            mode: 'train', 'valid', or 'test'

        Returns:
            X: (total_samples, num_voxels) features
            Y: (total_samples, num_emotions) targets
        """
        X_list = []
        Y_list = []

        print(f"\nLoading {mode} data from dataloader...")

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # batch['fmri_sequence']: (batch_size, 1, 96, 96, 96, seq_len)
            # batch['target']: (batch_size, seq_len, num_emotions)

            fmri_data = batch['fmri_sequence'].numpy()  # (B, 1, 96, 96, 96, S)
            target_data = batch['target'].numpy()  # (B, S, E)

            # Print shape on first batch for debugging
            if batch_idx == 0:
                print(f"  First batch fMRI shape: {fmri_data.shape}")
                print(f"  First batch target shape: {target_data.shape}")

            # Remove the channel dimension: (B, 1, X, Y, Z, S) -> (B, X, Y, Z, S)
            # Handle different possible shapes
            if fmri_data.ndim == 6 and fmri_data.shape[1] == 1:
                fmri_data = fmri_data.squeeze(1)
            elif fmri_data.ndim == 5:
                # Already (B, X, Y, Z, S) format
                pass
            else:
                raise ValueError(f"Unexpected fMRI data shape: {fmri_data.shape}")

            batch_size = fmri_data.shape[0]

            for b in range(batch_size):
                # Get single sequence: (96, 96, 96, seq_len)
                fmri_seq = fmri_data[b]

                # Flatten to (seq_len, num_voxels)
                features = self.flatten_sequence(fmri_seq)

                # Get targets: (seq_len, num_emotions)
                targets = target_data[b]

                # Each timepoint is a separate sample
                X_list.append(features)
                Y_list.append(targets)

        # Concatenate all samples
        X = np.vstack(X_list)  # (total_samples, num_voxels)
        Y = np.vstack(Y_list)  # (total_samples, num_emotions)

        print(f"{mode} data shape: X={X.shape}, Y={Y.shape}")

        # Set feature dimension from first batch
        if self.feature_dim is None:
            self.feature_dim = X.shape[1]
            print(f"Feature dimension: {self.feature_dim:,} voxels")

        return X, Y

    def fit(self, train_dataloader) -> Dict[str, float]:
        """
        Fit SVR models on training data

        Args:
            train_dataloader: PyTorch dataloader for training set

        Returns:
            Training metrics
        """
        print("\n" + "="*80)
        print("Training SVR Baseline")
        print("="*80)

        # Prepare training data
        X_train, Y_train = self.prepare_data_from_dataloader(train_dataloader, mode='train')

        # Train models for each emotion
        train_metrics = {}

        print("\nTraining SVR models for each emotion...")
        for e in tqdm(range(self.num_emotions), desc="Training emotions"):
            y_train = Y_train[:, e]

            # Standardize features
            if self.standardize:
                X_train_scaled = self.scalers[e].fit_transform(X_train)
            else:
                X_train_scaled = X_train

            # Fit SVR
            self.models[e].fit(X_train_scaled, y_train)

            # Training predictions
            y_pred = self.models[e].predict(X_train_scaled)

            # Metrics
            if self.task_type == 'classification':
                acc = accuracy_score(y_train, y_pred)
                f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)

                train_metrics[f'train_acc_{e}'] = acc
                train_metrics[f'train_f1_{e}'] = f1
                train_metrics[f'train_precision_{e}'] = precision
                train_metrics[f'train_recall_{e}'] = recall

                print(f"  Emotion {e}: Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            else:  # regression
                mse = mean_squared_error(y_train, y_pred)
                mae = mean_absolute_error(y_train, y_pred)
                r2 = r2_score(y_train, y_pred)

                train_metrics[f'train_mse_{e}'] = mse
                train_metrics[f'train_mae_{e}'] = mae
                train_metrics[f'train_r2_{e}'] = r2

                print(f"  Emotion {e}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        self.fitted = True

        # Overall metrics
        if self.task_type == 'classification':
            train_metrics['train_acc'] = np.mean([train_metrics[f'train_acc_{e}']
                                                   for e in range(self.num_emotions)])
            train_metrics['train_f1'] = np.mean([train_metrics[f'train_f1_{e}']
                                                  for e in range(self.num_emotions)])
            train_metrics['train_precision'] = np.mean([train_metrics[f'train_precision_{e}']
                                                         for e in range(self.num_emotions)])
            train_metrics['train_recall'] = np.mean([train_metrics[f'train_recall_{e}']
                                                      for e in range(self.num_emotions)])

            print(f"\nOverall Training - Acc: {train_metrics['train_acc']:.4f}, "
                  f"F1: {train_metrics['train_f1']:.4f}, "
                  f"Precision: {train_metrics['train_precision']:.4f}, "
                  f"Recall: {train_metrics['train_recall']:.4f}")
        else:  # regression
            train_metrics['train_mse'] = np.mean([train_metrics[f'train_mse_{e}']
                                                   for e in range(self.num_emotions)])
            train_metrics['train_mae'] = np.mean([train_metrics[f'train_mae_{e}']
                                                   for e in range(self.num_emotions)])
            train_metrics['train_r2'] = np.mean([train_metrics[f'train_r2_{e}']
                                                  for e in range(self.num_emotions)])

            print(f"\nOverall Training - MSE: {train_metrics['train_mse']:.4f}, "
                  f"MAE: {train_metrics['train_mae']:.4f}, R2: {train_metrics['train_r2']:.4f}")
        print("="*80)

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

        # Predict for each emotion
        all_predictions = []

        for e in range(self.num_emotions):
            if self.standardize:
                X_eval_scaled = self.scalers[e].transform(X_eval)
            else:
                X_eval_scaled = X_eval

            y_pred = self.models[e].predict(X_eval_scaled)
            all_predictions.append(y_pred)

        Y_pred = np.stack(all_predictions, axis=1)  # (num_samples, num_emotions)

        # Compute metrics
        metrics = {}

        if self.task_type == 'classification':
            # Overall metrics
            acc_overall = accuracy_score(Y_eval.flatten(), Y_pred.flatten())
            f1_overall = f1_score(Y_eval.flatten(), Y_pred.flatten(), average='weighted', zero_division=0)
            precision_overall = precision_score(Y_eval.flatten(), Y_pred.flatten(), average='weighted', zero_division=0)
            recall_overall = recall_score(Y_eval.flatten(), Y_pred.flatten(), average='weighted', zero_division=0)

            metrics[f'{mode}_acc'] = acc_overall
            metrics[f'{mode}_f1'] = f1_overall
            metrics[f'{mode}_precision'] = precision_overall
            metrics[f'{mode}_recall'] = recall_overall

            # Per-emotion metrics
            for e in range(self.num_emotions):
                y_true_e = Y_eval[:, e]
                y_pred_e = Y_pred[:, e]

                acc_e = accuracy_score(y_true_e, y_pred_e)
                f1_e = f1_score(y_true_e, y_pred_e, average='weighted', zero_division=0)
                precision_e = precision_score(y_true_e, y_pred_e, average='weighted', zero_division=0)
                recall_e = recall_score(y_true_e, y_pred_e, average='weighted', zero_division=0)

                metrics[f'{mode}_acc_{e}'] = acc_e
                metrics[f'{mode}_f1_{e}'] = f1_e
                metrics[f'{mode}_precision_{e}'] = precision_e
                metrics[f'{mode}_recall_{e}'] = recall_e

                print(f"  Emotion {e}: Acc={acc_e:.4f}, F1={f1_e:.4f}, "
                      f"Precision={precision_e:.4f}, Recall={recall_e:.4f}")

            print(f"\nOverall {mode} - Acc: {acc_overall:.4f}, F1: {f1_overall:.4f}, "
                  f"Precision: {precision_overall:.4f}, Recall: {recall_overall:.4f}")
        else:  # regression
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

                print(f"  Emotion {e}: MSE={mse_e:.4f}, MAE={mae_e:.4f}, "
                      f"R2={r2_e:.4f}, Corr={corr_e:.4f}")

            print(f"\nOverall {mode} - MSE: {mse_overall:.4f}, MAE: {mae_overall:.4f}, "
                  f"R2: {r2_overall:.4f}")
        print("="*80)

        return metrics

    def save(self, save_path: str):
        """Save SVR models and scalers"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            'models': self.models,
            'scalers': self.scalers,
            'num_emotions': self.num_emotions,
            'sequence_length': self.sequence_length,
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'standardize': self.standardize,
            'feature_dim': self.feature_dim,
            'task_type': self.task_type
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"\nSVR model saved to {save_path}")

    def load(self, load_path: str):
        """Load SVR models and scalers"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.models = state['models']
        self.scalers = state['scalers']
        self.num_emotions = state['num_emotions']
        self.sequence_length = state['sequence_length']
        self.kernel = state['kernel']
        self.C = state['C']
        self.epsilon = state['epsilon']
        self.standardize = state['standardize']
        self.feature_dim = state['feature_dim']
        self.task_type = state.get('task_type', 'regression')  # Backward compatibility
        self.fitted = True

        print(f"\nSVR model loaded from {load_path}")
