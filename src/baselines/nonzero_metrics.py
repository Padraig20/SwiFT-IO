"""
Non-Zero Metrics Calculator for SVR Baseline

This module provides metrics calculation that exactly matches pl_classifier.py
for fair comparison between SwiFT-IO and SVR baselines.

Metrics include:
- Overall: mse, mae, corrcoef, r2, adjusted_mse, adjusted_mae
- Non-zero: nonzero_mae, nonzero_mse, nonzero_rmse, nonzero_pearson
- Detection: detection_tpr, detection_fpr, detection_precision, detection_f1, detection_auroc
- Magnitude-stratified: small_mae, medium_mae, large_mae
- Zero: zero_mae, zero_mean_pred, zero_std_pred

Author: For comparison with SwiFT-IO
Reference: src/module/pl_classifier.py lines 598-800
"""

import numpy as np
from typing import Dict, Optional, Union
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score
)
from scipy.stats import pearsonr


class NonZeroMetricsCalculator:
    """
    Calculate metrics matching pl_classifier.py's _evaluate_metrics function

    This ensures fair comparison between SwiFT-IO and SVR baselines by using
    exactly the same metric calculation methods.
    """

    EMOTION_NAMES = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    EPSILON = 1e-6  # Same as pl_classifier.py line 616
    DETECTION_THRESHOLD = 0.5  # Same as pl_classifier.py line 675

    def __init__(self,
                 scaler=None,
                 label_scaling_method: str = 'standardization'):
        """
        Initialize the metrics calculator

        Args:
            scaler: StandardScaler or MinMaxScaler from sklearn
                   (should have mean_, scale_ for standardization
                    or data_min_, data_max_ for minmax)
            label_scaling_method: 'standardization' or 'minmax'
        """
        self.scaler = scaler
        self.label_scaling_method = label_scaling_method

    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled values to original scale

        Same logic as pl_classifier.py lines 608-613

        Args:
            scaled_values: Normalized/scaled values

        Returns:
            Values in original scale
        """
        if self.scaler is None:
            return scaled_values

        if self.label_scaling_method == 'standardization':
            # target_original = target_np * self.scaler.scale_[0] + self.scaler.mean_[0]
            return scaled_values * self.scaler.scale_[0] + self.scaler.mean_[0]
        elif self.label_scaling_method == 'minmax':
            # target_original = target_np * (data_max - data_min) + data_min
            data_range = self.scaler.data_max_[0] - self.scaler.data_min_[0]
            return scaled_values * data_range + self.scaler.data_min_[0]
        else:
            return scaled_values

    def create_masks(self, target_original: np.ndarray) -> tuple:
        """
        Create zero/nonzero masks using original scale values

        Same logic as pl_classifier.py lines 616-617

        Args:
            target_original: Target values in original scale

        Returns:
            (mask_zero, mask_nonzero): Boolean masks
        """
        mask_zero = np.abs(target_original) < self.EPSILON
        mask_nonzero = ~mask_zero
        return mask_zero, mask_nonzero

    def calculate_metrics(self,
                         y_true_scaled: np.ndarray,
                         y_pred_scaled: np.ndarray,
                         emotion_idx: Optional[int] = None,
                         mode_str: str = 'test') -> Dict[str, float]:
        """
        Calculate all metrics matching pl_classifier.py

        Args:
            y_true_scaled: Ground truth (scaled/normalized)
            y_pred_scaled: Predictions (scaled/normalized)
            emotion_idx: Emotion index (0-6) for naming
            mode_str: 'valid' or 'test' for metric name prefix

        Returns:
            Dictionary of all metrics
        """
        # Get emotion name
        if emotion_idx is not None and emotion_idx < len(self.EMOTION_NAMES):
            emotion_name = self.EMOTION_NAMES[emotion_idx]
        else:
            emotion_name = f'emotion_{emotion_idx}' if emotion_idx is not None else 'all'

        metrics = {}

        # Convert to numpy if needed
        y_true = np.asarray(y_true_scaled).flatten()
        y_pred = np.asarray(y_pred_scaled).flatten()

        # 1. Inverse transform to original scale
        y_true_original = self.inverse_transform(y_true)
        y_pred_original = self.inverse_transform(y_pred)

        # 2. Create masks (using original scale!)
        mask_zero, mask_nonzero = self.create_masks(y_true_original)

        n_total = len(y_true)
        n_zero = mask_zero.sum()
        n_nonzero = mask_nonzero.sum()

        # ========================================================================
        # SAMPLE COUNTS
        # ========================================================================
        metrics[f'{mode_str}_n_total_{emotion_name}'] = float(n_total)
        metrics[f'{mode_str}_n_zero_{emotion_name}'] = float(n_zero)
        metrics[f'{mode_str}_n_nonzero_{emotion_name}'] = float(n_nonzero)
        metrics[f'{mode_str}_pct_zero_{emotion_name}'] = float(n_zero / n_total * 100) if n_total > 0 else 0.0

        # ========================================================================
        # OVERALL METRICS (scaled)
        # ========================================================================
        metrics[f'{mode_str}_mse_{emotion_name}'] = mean_squared_error(y_true, y_pred)
        metrics[f'{mode_str}_mae_{emotion_name}'] = mean_absolute_error(y_true, y_pred)

        if n_total > 1:
            try:
                corr, _ = pearsonr(y_true, y_pred)
                metrics[f'{mode_str}_corrcoef_{emotion_name}'] = corr
            except:
                metrics[f'{mode_str}_corrcoef_{emotion_name}'] = 0.0

            try:
                metrics[f'{mode_str}_r2_score_{emotion_name}'] = r2_score(y_true, y_pred)
            except:
                metrics[f'{mode_str}_r2_score_{emotion_name}'] = 0.0
        else:
            metrics[f'{mode_str}_corrcoef_{emotion_name}'] = 0.0
            metrics[f'{mode_str}_r2_score_{emotion_name}'] = 0.0

        # ========================================================================
        # ADJUSTED METRICS (original scale)
        # ========================================================================
        metrics[f'{mode_str}_adjusted_mse_{emotion_name}'] = mean_squared_error(y_true_original, y_pred_original)
        metrics[f'{mode_str}_adjusted_mae_{emotion_name}'] = mean_absolute_error(y_true_original, y_pred_original)

        # ========================================================================
        # NON-ZERO METRICS (핵심!)
        # Same as pl_classifier.py lines 629-659
        # ========================================================================
        if n_nonzero > 1:
            y_true_nz = y_true[mask_nonzero]
            y_pred_nz = y_pred[mask_nonzero]
            y_true_nz_orig = y_true_original[mask_nonzero]
            y_pred_nz_orig = y_pred_original[mask_nonzero]

            # Non-zero MAE, MSE, RMSE (scaled)
            nonzero_mae = mean_absolute_error(y_true_nz, y_pred_nz)
            nonzero_mse = mean_squared_error(y_true_nz, y_pred_nz)
            nonzero_rmse = np.sqrt(nonzero_mse)

            # Non-zero Pearson correlation
            try:
                nonzero_pearson, _ = pearsonr(y_true_nz, y_pred_nz)
            except:
                nonzero_pearson = 0.0

            # Adjusted non-zero metrics (original scale)
            nonzero_adjusted_mae = mean_absolute_error(y_true_nz_orig, y_pred_nz_orig)
            nonzero_adjusted_rmse = np.sqrt(mean_squared_error(y_true_nz_orig, y_pred_nz_orig))

            metrics[f'{mode_str}_nonzero_mae_{emotion_name}'] = nonzero_mae
            metrics[f'{mode_str}_nonzero_mse_{emotion_name}'] = nonzero_mse
            metrics[f'{mode_str}_nonzero_rmse_{emotion_name}'] = nonzero_rmse
            metrics[f'{mode_str}_nonzero_pearson_{emotion_name}'] = nonzero_pearson
            metrics[f'{mode_str}_nonzero_adjusted_mae_{emotion_name}'] = nonzero_adjusted_mae
            metrics[f'{mode_str}_nonzero_adjusted_rmse_{emotion_name}'] = nonzero_adjusted_rmse

        # ========================================================================
        # DETECTION METRICS (Binary Classification)
        # Same as pl_classifier.py lines 662-711
        # ========================================================================
        target_binary = (y_true_original > self.DETECTION_THRESHOLD).astype(int)
        pred_binary = (y_pred_original > self.DETECTION_THRESHOLD).astype(int)

        # Confusion matrix components
        TP = np.sum((target_binary == 1) & (pred_binary == 1))
        TN = np.sum((target_binary == 0) & (pred_binary == 0))
        FP = np.sum((target_binary == 0) & (pred_binary == 1))
        FN = np.sum((target_binary == 1) & (pred_binary == 0))

        eps = 1e-8
        TPR = TP / (TP + FN + eps)  # Sensitivity/Recall
        FPR = FP / (FP + TN + eps)  # False Positive Rate
        Precision = TP / (TP + FP + eps)
        F1 = 2 * Precision * TPR / (Precision + TPR + eps)
        Specificity = TN / (TN + FP + eps)

        # AUROC
        try:
            if len(np.unique(target_binary)) > 1:
                detection_auroc = roc_auc_score(target_binary, y_pred_original)
            else:
                detection_auroc = 0.0
        except:
            detection_auroc = 0.0

        metrics[f'{mode_str}_detection_tpr_{emotion_name}'] = float(TPR)
        metrics[f'{mode_str}_detection_fpr_{emotion_name}'] = float(FPR)
        metrics[f'{mode_str}_detection_precision_{emotion_name}'] = float(Precision)
        metrics[f'{mode_str}_detection_f1_{emotion_name}'] = float(F1)
        metrics[f'{mode_str}_detection_specificity_{emotion_name}'] = float(Specificity)
        metrics[f'{mode_str}_detection_auroc_{emotion_name}'] = float(detection_auroc)

        # ========================================================================
        # MAGNITUDE-STRATIFIED METRICS
        # Same as pl_classifier.py lines 714-767
        # ========================================================================
        if n_nonzero > 0:
            # Using scaled values for magnitude thresholds (as in pl_classifier.py)
            mask_small = (y_true > 0) & (y_true <= 1)
            mask_medium = (y_true > 1) & (y_true <= 5)
            mask_large = (y_true > 5)

            for name, mask in [('small', mask_small), ('medium', mask_medium), ('large', mask_large)]:
                n_samples = mask.sum()
                if n_samples > 0:
                    metrics[f'{mode_str}_{name}_mae_{emotion_name}'] = mean_absolute_error(
                        y_true[mask], y_pred[mask])
                    metrics[f'{mode_str}_{name}_mse_{emotion_name}'] = mean_squared_error(
                        y_true[mask], y_pred[mask])
                    metrics[f'{mode_str}_{name}_rmse_{emotion_name}'] = np.sqrt(
                        metrics[f'{mode_str}_{name}_mse_{emotion_name}'])
                    metrics[f'{mode_str}_n_{name}_{emotion_name}'] = float(n_samples)

        # ========================================================================
        # ZERO METRICS
        # Same as pl_classifier.py lines 769-790
        # ========================================================================
        if n_zero > 0:
            y_pred_zero = y_pred[mask_zero]

            # Zero MAE (how close to 0 are predictions on zero targets)
            zero_mae = np.abs(y_pred_zero).mean()
            zero_mean_pred = y_pred_zero.mean()
            zero_std_pred = y_pred_zero.std() if len(y_pred_zero) > 1 else 0.0

            # Adjusted zero metrics (original scale)
            y_pred_zero_orig = y_pred_original[mask_zero]
            zero_adjusted_mae = np.abs(y_pred_zero_orig).mean()

            metrics[f'{mode_str}_zero_mae_{emotion_name}'] = float(zero_mae)
            metrics[f'{mode_str}_zero_mean_pred_{emotion_name}'] = float(zero_mean_pred)
            metrics[f'{mode_str}_zero_std_pred_{emotion_name}'] = float(zero_std_pred)
            metrics[f'{mode_str}_zero_adjusted_mae_{emotion_name}'] = float(zero_adjusted_mae)

        return metrics

    def calculate_summary_metrics(self,
                                  all_emotion_metrics: Dict[str, float],
                                  mode_str: str = 'test') -> Dict[str, float]:
        """
        Calculate summary metrics across all emotions

        Same as pl_classifier.py lines 1019-1073

        Args:
            all_emotion_metrics: Dictionary with all per-emotion metrics
            mode_str: 'valid' or 'test'

        Returns:
            Dictionary with avg/std metrics
        """
        summary = {}

        # Collect non-zero metrics per emotion
        nonzero_maes = []
        nonzero_pearsons = []
        nonzero_rmses = []

        for emotion_name in self.EMOTION_NAMES:
            mae_key = f'{mode_str}_nonzero_mae_{emotion_name}'
            pearson_key = f'{mode_str}_nonzero_pearson_{emotion_name}'
            rmse_key = f'{mode_str}_nonzero_rmse_{emotion_name}'

            if mae_key in all_emotion_metrics:
                nonzero_maes.append(all_emotion_metrics[mae_key])
            if pearson_key in all_emotion_metrics:
                nonzero_pearsons.append(all_emotion_metrics[pearson_key])
            if rmse_key in all_emotion_metrics:
                nonzero_rmses.append(all_emotion_metrics[rmse_key])

        # Calculate averages and stds
        if nonzero_maes:
            summary[f'{mode_str}_avg_nonzero_mae'] = np.mean(nonzero_maes)
            summary[f'{mode_str}_std_nonzero_mae'] = np.std(nonzero_maes)

        if nonzero_pearsons:
            summary[f'{mode_str}_avg_nonzero_pearson'] = np.mean(nonzero_pearsons)
            summary[f'{mode_str}_std_nonzero_pearson'] = np.std(nonzero_pearsons)

        if nonzero_rmses:
            summary[f'{mode_str}_avg_nonzero_rmse'] = np.mean(nonzero_rmses)
            summary[f'{mode_str}_std_nonzero_rmse'] = np.std(nonzero_rmses)

        return summary

    def print_summary(self,
                      all_metrics: Dict[str, float],
                      mode_str: str = 'test'):
        """
        Print a formatted summary table

        Args:
            all_metrics: Dictionary with all metrics
            mode_str: 'valid' or 'test'
        """
        print("\n" + "="*120)
        print(f"{'Emotion':<10} | {'MAE':<8} | {'NZ-MAE':<8} | {'NZ-Pearson':<10} | {'AUROC':<8} | {'TPR':<8} | {'%Zero':<8}")
        print("-"*120)

        for emotion_name in self.EMOTION_NAMES:
            mae = all_metrics.get(f'{mode_str}_mae_{emotion_name}', float('nan'))
            nz_mae = all_metrics.get(f'{mode_str}_nonzero_mae_{emotion_name}', float('nan'))
            nz_pearson = all_metrics.get(f'{mode_str}_nonzero_pearson_{emotion_name}', float('nan'))
            auroc = all_metrics.get(f'{mode_str}_detection_auroc_{emotion_name}', float('nan'))
            tpr = all_metrics.get(f'{mode_str}_detection_tpr_{emotion_name}', float('nan'))
            pct_zero = all_metrics.get(f'{mode_str}_pct_zero_{emotion_name}', float('nan'))

            print(f"{emotion_name:<10} | {mae:>8.4f} | {nz_mae:>8.4f} | {nz_pearson:>10.4f} | "
                  f"{auroc:>8.4f} | {tpr:>8.4f} | {pct_zero:>7.1f}%")

        print("-"*120)

        # Print summary
        summary = self.calculate_summary_metrics(all_metrics, mode_str)
        if summary:
            avg_nz_mae = summary.get(f'{mode_str}_avg_nonzero_mae', float('nan'))
            std_nz_mae = summary.get(f'{mode_str}_std_nonzero_mae', float('nan'))
            avg_nz_pearson = summary.get(f'{mode_str}_avg_nonzero_pearson', float('nan'))
            std_nz_pearson = summary.get(f'{mode_str}_std_nonzero_pearson', float('nan'))

            print(f"{'AVERAGE':<10} |          | {avg_nz_mae:>8.4f} | {avg_nz_pearson:>10.4f} |")
            print(f"{'STD':<10} |          | {std_nz_mae:>8.4f} | {std_nz_pearson:>10.4f} |")

        print("="*120)
