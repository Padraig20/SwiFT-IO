#!/usr/bin/env python3
"""
Helper functions for Youden threshold implementation in pl_classifier.py

These functions should be integrated into LitClassifier class:
1. Add optimal_thresholds attribute in __init__
2. Add find_optimal_threshold_youden method
3. Call it in validation_epoch_end
4. Use optimal thresholds in _evaluate_metrics for test set
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score


def find_optimal_threshold_youden(targets, probabilities):
    """
    Find optimal threshold using Youden Index
    J = Sensitivity + Specificity - 1 = TPR - FPR

    Args:
        targets: Binary labels (0 or 1), can be numpy array or tensor
        probabilities: Predicted probabilities for class 1, can be numpy array or tensor

    Returns:
        optimal_threshold (float)
    """
    # Convert to numpy if tensors
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(probabilities, 'cpu'):
        probabilities = probabilities.cpu().numpy()

    # Flatten if needed
    targets = targets.flatten()
    probabilities = probabilities.flatten()

    # Remove NaN values
    valid_mask = ~np.isnan(targets) & ~np.isnan(probabilities)
    targets = targets[valid_mask]
    probabilities = probabilities[valid_mask]

    # Check if we have both classes
    unique_classes = np.unique(targets)
    if len(unique_classes) < 2:
        print(f"  Warning: Only one class present ({unique_classes}) - using default threshold 0.5")
        return 0.5

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, probabilities)

    # Youden index = TPR - FPR
    j_scores = tpr - fpr

    # Find optimal threshold
    optimal_idx = j_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    return float(optimal_threshold)


def calculate_optimal_thresholds_per_emotion(logits, targets, num_targets=7, num_classes=2):
    """
    Calculate optimal thresholds for each emotion using validation data

    Args:
        logits: Tensor of shape (batch, time, num_targets, num_classes)
        targets: Tensor of shape (batch, time, num_targets)
        num_targets: Number of emotion categories (default 7)
        num_classes: Number of classes (default 2 for binary)

    Returns:
        dict: {emotion_idx: optimal_threshold}
    """
    import torch
    import torch.nn.functional as F

    optimal_thresholds = {}

    for i in range(num_targets):
        logits_emotion = logits[:, :, i, :]  # (batch, time, num_classes)
        targets_emotion = targets[:, :, i]  # (batch, time)

        # Get probabilities for class 1
        probs = F.softmax(logits_emotion.to(dtype=torch.float32), dim=-1)
        probs_pos = probs[..., 1]  # Probability of class 1

        # Flatten
        probs_flat = probs_pos.flatten().cpu().numpy()
        targets_flat = targets_emotion.flatten().cpu().numpy()

        # Remove NaN
        valid_mask = ~np.isnan(targets_flat) & ~np.isnan(probs_flat)
        probs_flat = probs_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

        if len(targets_flat) == 0:
            print(f"  Emotion {i}: No valid samples - using default threshold 0.5")
            optimal_thresholds[i] = 0.5
            continue

        # Find optimal threshold
        opt_thr = find_optimal_threshold_youden(targets_flat, probs_flat)
        optimal_thresholds[i] = opt_thr

        print(f"  Emotion {i}: Optimal threshold = {opt_thr:.4f}")

    return optimal_thresholds


# ============================================================================
# INTEGRATION INSTRUCTIONS FOR pl_classifier.py
# ============================================================================
"""
1. In __init__ method (around line 106), add:

   self.optimal_thresholds = None  # Will be calculated on validation set


2. Add this method to LitClassifier class (after _evaluate_metrics):

   def _calculate_optimal_thresholds(self, subj_array, total_out):
       '''Calculate optimal thresholds using Youden Index on validation data'''
       if self.hparams.downstream_task_type != 'classification':
           return

       if self.hparams.decoder not in ['series_decoder', 'lstm_series_regression_head']:
           return

       if self.hparams.num_classes != 2:
           print("[INFO] Youden threshold only supported for binary classification")
           return

       subjects = np.unique(subj_array)

       subj_avg_logits = []
       subj_targets = []
       for subj in subjects:
           subj_logits = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]
           subj_avg_logits.append(subj_logits)
           subj_targets.append([total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj][0])

       subj_avg_logits = [i[0] for i in subj_avg_logits]
       subj_avg_logits = torch.stack(subj_avg_logits)
       subj_targets = torch.stack(subj_targets)

       print("\\n" + "="*80)
       print("CALCULATING OPTIMAL THRESHOLDS (Youden Index)")
       print("="*80)

       from einops import rearrange
       t = self.hparams.img_size[3]

       # Reshape: (b*t*ta, c) -> (b, t, ta, c)
       subj_avg_logits = rearrange(subj_avg_logits, '(b t ta) c -> b t ta c',
                                   t=t, ta=self.hparams.num_targets, c=self.hparams.num_classes)
       subj_targets = rearrange(subj_targets, '(b t ta) -> b t ta',
                                t=t, ta=self.hparams.num_targets)

       optimal_thresholds = {}

       for i in range(self.hparams.num_targets):
           logits_emotion = subj_avg_logits[:, :, i, :]  # (batch, time, num_classes)
           targets_emotion = subj_targets[:, :, i]  # (batch, time)

           # Get probabilities for class 1
           probs = F.softmax(logits_emotion.to(dtype=torch.float32), dim=-1)
           probs_pos = probs[..., 1]

           # Flatten
           probs_flat = probs_pos.flatten().cpu().numpy()
           targets_flat = targets_emotion.flatten().cpu().numpy()

           # Remove NaN
           valid_mask = ~np.isnan(targets_flat) & ~np.isnan(probs_flat)
           probs_flat = probs_flat[valid_mask]
           targets_flat = targets_flat[valid_mask]

           if len(targets_flat) == 0:
               print(f"  Emotion {i}: No valid samples - using default threshold 0.5")
               optimal_thresholds[i] = 0.5
               continue

           # Check if both classes present
           unique_classes = np.unique(targets_flat)
           if len(unique_classes) < 2:
               print(f"  Emotion {i}: Only one class ({unique_classes}) - using default threshold 0.5")
               optimal_thresholds[i] = 0.5
               continue

           # Calculate ROC curve
           from sklearn.metrics import roc_curve
           fpr, tpr, thresholds = roc_curve(targets_flat, probs_flat)

           # Youden index = TPR - FPR
           j_scores = tpr - fpr
           optimal_idx = j_scores.argmax()
           opt_thr = float(thresholds[optimal_idx])

           optimal_thresholds[i] = opt_thr
           print(f"  Emotion {i}: Optimal threshold = {opt_thr:.4f} (J={j_scores[optimal_idx]:.4f})")

       self.optimal_thresholds = optimal_thresholds
       print("="*80 + "\\n")


3. In validation_epoch_end method (around line 539, after _evaluate_metrics for valid):

   # Calculate optimal thresholds from validation set
   if mode == "valid":  # Only calculate on validation, not test
       self._calculate_optimal_thresholds(subj_valid, total_out_valid)


4. In _evaluate_metrics method (around line 369-392), replace the prediction line:

   OLD:
   predictions = probabilities.argmax(dim=-1)  # (b, temporal_size)

   NEW:
   # Use optimal threshold if available, otherwise use argmax (threshold=0.5)
   if self.optimal_thresholds and i in self.optimal_thresholds and mode == 'test':
       opt_thr = self.optimal_thresholds[i]
       predictions = (probabilities[..., 1] >= opt_thr).long()  # (b, temporal_size)
       print(f"  [INFO] Emotion {i}: Using optimal threshold {opt_thr:.4f}")
   else:
       predictions = probabilities.argmax(dim=-1)  # (b, temporal_size)

"""
