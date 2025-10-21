"""
GLM (General Linear Model) Baseline for fMRI emotion prediction

Uses pre-extracted ROI timeseries from /scratch/HBN/9.2.movieDM_ROI_timeseries/
and existing train/val/test splits from SwiFT-IO data_module.

Author: Generated for comparison with SwiFT-IO
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from .roi_selector import EmotionROISelector


class GLMBaseline:
    """
    GLM-based baseline for emotion prediction from fMRI ROI timeseries

    Workflow:
    1. Load ROI timeseries CSV files for each subject
    2. Select emotion-related ROIs using EmotionROISelector
    3. Fit Ridge regression: ROI activity -> emotion labels
    4. Predict emotions on val/test sets
    """

    def __init__(self,
                 roi_timeseries_dir: str = "/scratch/HBN/9.2.movieDM_ROI_timeseries",
                 num_emotions: int = 7,
                 sequence_length: int = 20,
                 use_cross_validation: bool = True,
                 alpha: float = 1.0,
                 standardize: bool = True,
                 use_emotion_rois_only: bool = True):
        """
        Args:
            roi_timeseries_dir: Path to directory with ROI timeseries CSV files
            num_emotions: Number of emotions to predict (default: 7)
            sequence_length: Length of fMRI sequence to use
            use_cross_validation: Use RidgeCV for alpha selection
            alpha: Ridge regularization parameter (if not using CV)
            standardize: Whether to standardize ROI features
            use_emotion_rois_only: Use only emotion-related ROIs (via EmotionROISelector)
        """
        self.roi_timeseries_dir = roi_timeseries_dir
        self.num_emotions = num_emotions
        self.sequence_length = sequence_length
        self.use_cross_validation = use_cross_validation
        self.alpha = alpha
        self.standardize = standardize
        self.use_emotion_rois_only = use_emotion_rois_only

        # ROI selector
        self.roi_selector = EmotionROISelector() if use_emotion_rois_only else None

        # Models: one Ridge regression per emotion
        if use_cross_validation:
            alphas = np.logspace(-3, 3, 20)  # 10^-3 to 10^3
            self.models = [RidgeCV(alphas=alphas, cv=5) for _ in range(num_emotions)]
        else:
            self.models = [Ridge(alpha=alpha) for _ in range(num_emotions)]

        # Scalers for standardization
        self.scalers = [StandardScaler() for _ in range(num_emotions)] if standardize else None

        self.fitted = False
        self.selected_roi_names = None

    def load_subject_roi_timeseries(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Load ROI timeseries for a subject

        Args:
            subject_id: Subject ID (e.g., 'NDARAA947ZG5')

        Returns:
            DataFrame with columns: TR, ROI1, ROI2, ... or None if file not found
        """
        # Handle both with and without 'sub-' prefix
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        csv_path = os.path.join(self.roi_timeseries_dir,
                               f"{subject_id}_movieDM_roi_temporal_activity.csv")

        if not os.path.exists(csv_path):
            # Try without 'sub-' prefix
            subject_id = subject_id.replace('sub-', '')
            csv_path = os.path.join(self.roi_timeseries_dir,
                                   f"sub-{subject_id}_movieDM_roi_temporal_activity.csv")

            if not os.path.exists(csv_path):
                print(f"Warning: ROI timeseries not found for {subject_id}")
                return None

        df = pd.read_csv(csv_path)

        # Select emotion-related ROIs if specified
        if self.roi_selector is not None:
            df = self.roi_selector.select_rois(df)

        return df

    def prepare_training_data(self, subject_dict: Dict, emotion_labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from subject dictionary

        Args:
            subject_dict: Dict mapping subject_id -> (sex, target)
            emotion_labels: DataFrame with emotion labels (columns: frame, Anger, Happy, ...)

        Returns:
            X: (num_samples, num_rois) features
            Y: (num_samples, num_emotions) targets
        """
        X_list = []
        Y_list = []

        print(f"Loading ROI timeseries for {len(subject_dict)} subjects...")

        for subject_id in tqdm(subject_dict.keys()):
            # Load ROI timeseries
            df_roi = self.load_subject_roi_timeseries(subject_id)

            if df_roi is None:
                continue

            # Store selected ROI names (from first subject)
            if self.selected_roi_names is None:
                self.selected_roi_names = [col for col in df_roi.columns if col != 'TR']
                print(f"Selected {len(self.selected_roi_names)} ROIs")

            # Get ROI values (exclude TR column)
            roi_values = df_roi.drop(columns=['TR']).values  # (T, num_rois)

            # Skip subjects with different number of ROIs (missing ROIs)
            if roi_values.shape[1] != len(self.selected_roi_names):
                print(f"Warning: Skipping {subject_id} - has {roi_values.shape[1]} ROIs instead of {len(self.selected_roi_names)}")
                continue

            # Get emotion labels for this subject's timepoints
            # Assuming emotion_labels has same timepoints for all subjects
            num_frames = len(df_roi)

            # Align with emotion labels
            emotion_values = emotion_labels.iloc[:num_frames].values  # (T, num_emotions)

            # Create sequences of length sequence_length
            for start_idx in range(0, num_frames - self.sequence_length + 1):
                end_idx = start_idx + self.sequence_length

                # Extract sequence
                roi_seq = roi_values[start_idx:end_idx, :]  # (seq_len, num_rois)
                emotion_seq = emotion_values[start_idx:end_idx, :]  # (seq_len, num_emotions)

                # Each timepoint in sequence is a training sample
                X_list.append(roi_seq)
                Y_list.append(emotion_seq)

        # Concatenate all sequences
        X = np.vstack(X_list)  # (total_samples, num_rois)
        Y = np.vstack(Y_list)  # (total_samples, num_emotions)

        return X, Y

    def fit(self, train_subject_dict: Dict, emotion_labels: pd.DataFrame) -> Dict[str, float]:
        """
        Fit GLM models on training subjects

        Args:
            train_subject_dict: Dict of training subjects
            emotion_labels: DataFrame with emotion labels

        Returns:
            Training metrics
        """
        print("\n=== Training GLM Baseline ===")

        # Prepare training data
        X_train, Y_train = self.prepare_training_data(train_subject_dict, emotion_labels)

        print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
        print(f"Number of ROI features: {X_train.shape[1]}")

        # Train models for each emotion
        train_metrics = {}

        for e in tqdm(range(self.num_emotions), desc="Training emotion models"):
            y_train = Y_train[:, e]

            # Standardize features
            if self.standardize:
                X_train_scaled = self.scalers[e].fit_transform(X_train)
            else:
                X_train_scaled = X_train

            # Fit model
            self.models[e].fit(X_train_scaled, y_train)

            # Training predictions
            y_pred = self.models[e].predict(X_train_scaled)

            # Metrics
            mse = mean_squared_error(y_train, y_pred)
            mae = mean_absolute_error(y_train, y_pred)

            train_metrics[f'train_mse_emotion_{e}'] = mse
            train_metrics[f'train_mae_emotion_{e}'] = mae

            if self.use_cross_validation:
                best_alpha = self.models[e].alpha_
                print(f"  Emotion {e}: Best alpha={best_alpha:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        self.fitted = True

        # Overall metrics
        train_metrics['train_mse'] = np.mean([train_metrics[f'train_mse_emotion_{e}']
                                               for e in range(self.num_emotions)])
        train_metrics['train_mae'] = np.mean([train_metrics[f'train_mae_emotion_{e}']
                                               for e in range(self.num_emotions)])

        print(f"\nOverall Training MSE: {train_metrics['train_mse']:.4f}")
        print(f"Overall Training MAE: {train_metrics['train_mae']:.4f}")

        return train_metrics

    def evaluate(self, subject_dict: Dict, emotion_labels: pd.DataFrame, mode: str = 'test') -> Dict[str, float]:
        """
        Evaluate GLM on validation or test subjects

        Args:
            subject_dict: Dict of subjects to evaluate
            emotion_labels: DataFrame with emotion labels
            mode: 'valid' or 'test'

        Returns:
            Evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        print(f"\n=== Evaluating on {mode} set ===")

        # Prepare data
        X_eval, Y_eval = self.prepare_training_data(subject_dict, emotion_labels)

        print(f"{mode} data shape: X={X_eval.shape}, Y={Y_eval.shape}")

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

        # Overall metrics
        mse_overall = mean_squared_error(Y_eval.flatten(), Y_pred.flatten())
        mae_overall = mean_absolute_error(Y_eval.flatten(), Y_pred.flatten())

        metrics[f'{mode}_mse'] = mse_overall
        metrics[f'{mode}_mae'] = mae_overall

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
            metrics[f'{mode}_r2_score_{e}'] = r2_e
            metrics[f'{mode}_corrcoef_{e}'] = corr_e

            print(f"  Emotion {e}: MSE={mse_e:.4f}, MAE={mae_e:.4f}, R2={r2_e:.4f}, Corr={corr_e:.4f}")

        print(f"\nOverall {mode} MSE: {mse_overall:.4f}")
        print(f"Overall {mode} MAE: {mae_overall:.4f}")

        return metrics

    def save(self, save_path: str):
        """Save GLM models"""
        import pickle

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            'models': self.models,
            'scalers': self.scalers,
            'num_emotions': self.num_emotions,
            'sequence_length': self.sequence_length,
            'alpha': self.alpha,
            'standardize': self.standardize,
            'use_emotion_rois_only': self.use_emotion_rois_only,
            'selected_roi_names': self.selected_roi_names
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"GLM model saved to {save_path}")

    def load(self, load_path: str):
        """Load GLM models"""
        import pickle

        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.models = state['models']
        self.scalers = state['scalers']
        self.num_emotions = state['num_emotions']
        self.sequence_length = state['sequence_length']
        self.alpha = state['alpha']
        self.standardize = state['standardize']
        self.use_emotion_rois_only = state['use_emotion_rois_only']
        self.selected_roi_names = state['selected_roi_names']
        self.fitted = True

        print(f"GLM model loaded from {load_path}")
