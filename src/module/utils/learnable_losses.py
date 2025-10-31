"""
Learnable Weighted Loss Functions for Sparse Regression Tasks

This module implements learnable loss functions designed for sparse, imbalanced
regression problems such as emotion prediction from fMRI data.

References:
- Uncertainty-based weighting: "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics" (Kendall et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PerEmotionLearnableWeightedMSE(nn.Module):
    """
    Per-Emotion Learnable Weighted MSE Loss (Option 2)

    Learns separate weights for zero and non-zero values for each emotion.
    This allows the model to automatically adjust the importance of different
    emotions based on their sparsity during training.

    Args:
        num_emotions (int): Number of emotions to predict (default: 7)
        init_zero_weight (float): Initial weight for zero values (default: 0.1)
        init_nonzero_weight (float): Initial weight for non-zero values (default: 5.0)

    Example:
        >>> loss_fn = PerEmotionLearnableWeightedMSE(num_emotions=7)
        >>> pred = torch.randn(32, 20, 7)  # (batch, time, emotions)
        >>> target = torch.randn(32, 20, 7)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, num_emotions=7, init_zero_weight=0.1, init_nonzero_weight=5.0):
        super().__init__()
        self.num_emotions = num_emotions

        # Learn weights in log space to ensure they're always positive
        self.log_zero_weights = nn.Parameter(
            torch.ones(num_emotions) * np.log(init_zero_weight)
        )
        self.log_nonzero_weights = nn.Parameter(
            torch.ones(num_emotions) * np.log(init_nonzero_weight)
        )

    def forward(self, pred, target):
        """
        Compute weighted MSE loss

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
            target: Ground truth, same shape as pred

        Returns:
            Scalar loss value
        """
        # Convert log weights to actual weights (always positive)
        zero_weights = torch.exp(self.log_zero_weights)  # (num_emotions,)
        nonzero_weights = torch.exp(self.log_nonzero_weights)  # (num_emotions,)

        # Reshape weights for broadcasting
        if pred.dim() == 3:  # (batch, seq_len, num_emotions)
            weight_shape = (1, 1, self.num_emotions)
        elif pred.dim() == 2:  # (batch, num_emotions)
            weight_shape = (1, self.num_emotions)
        else:
            raise ValueError(f"Expected pred to have 2 or 3 dimensions, got {pred.dim()}")

        zero_weights = zero_weights.view(*weight_shape)
        nonzero_weights = nonzero_weights.view(*weight_shape)

        # Create weight mask: use nonzero_weight where target != 0
        weight = torch.where(
            target != 0,
            nonzero_weights,
            zero_weights
        )

        # Compute weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = weight * mse
        loss = weighted_mse.mean()

        return loss

    def get_weights(self):
        """
        Get current learned weights for inspection

        Returns:
            dict: Dictionary with 'zero_weights' and 'nonzero_weights' as numpy arrays
        """
        return {
            'zero_weights': torch.exp(self.log_zero_weights).detach().cpu().numpy(),
            'nonzero_weights': torch.exp(self.log_nonzero_weights).detach().cpu().numpy()
        }


class UncertaintyWeightedMSE(nn.Module):
    """
    Uncertainty-based Weighted MSE Loss (Option 3)

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., CVPR 2018)
    Learns a log-variance (uncertainty) parameter for each emotion, which automatically
    balances the contribution of each emotion to the total loss.

    The loss for emotion i is: (1 / (2*sigma_i^2)) * MSE_i + log(sigma_i)
    - When sigma_i is large (high uncertainty), MSE_i is down-weighted
    - The log(sigma_i) term prevents sigma from going to infinity

    Args:
        num_emotions (int): Number of emotions to predict (default: 7)
        init_log_var (float): Initial log variance for all emotions (default: 0.0, i.e., variance=1.0)

    Example:
        >>> loss_fn = UncertaintyWeightedMSE(num_emotions=7)
        >>> pred = torch.randn(32, 20, 7)  # (batch, time, emotions)
        >>> target = torch.randn(32, 20, 7)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, num_emotions=7, init_log_var=0.0):
        super().__init__()
        self.num_emotions = num_emotions

        # Learn log variance (log(sigma^2)) for each emotion
        # Log variance allows negative values (when variance < 1)
        self.log_vars = nn.Parameter(torch.ones(num_emotions) * init_log_var)

    def forward(self, pred, target):
        """
        Compute uncertainty-weighted MSE loss

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
            target: Ground truth, same shape as pred

        Returns:
            Scalar loss value
        """
        # Compute per-emotion MSE
        if pred.dim() == 3:  # (batch, seq_len, num_emotions)
            # MSE for each emotion: mean over batch and time dimensions
            loss = 0
            for i in range(self.num_emotions):
                mse_i = F.mse_loss(pred[..., i], target[..., i], reduction='mean')
                # Precision = 1 / sigma^2 = exp(-log_var)
                precision = torch.exp(-self.log_vars[i])
                # Loss = precision * MSE + log_var (regularization term)
                loss += precision * mse_i + self.log_vars[i]

            # Average over emotions
            loss = loss / self.num_emotions

        elif pred.dim() == 2:  # (batch, num_emotions)
            loss = 0
            for i in range(self.num_emotions):
                mse_i = F.mse_loss(pred[:, i], target[:, i], reduction='mean')
                precision = torch.exp(-self.log_vars[i])
                loss += precision * mse_i + self.log_vars[i]

            loss = loss / self.num_emotions
        else:
            raise ValueError(f"Expected pred to have 2 or 3 dimensions, got {pred.dim()}")

        return loss

    def get_uncertainties(self):
        """
        Get current learned uncertainties (sigma) for inspection

        Returns:
            dict: Dictionary with 'log_vars' (log variance), 'vars' (variance),
                  and 'sigmas' (standard deviation) as numpy arrays
        """
        log_vars = self.log_vars.detach().cpu().numpy()
        vars = np.exp(log_vars)
        sigmas = np.sqrt(vars)

        return {
            'log_vars': log_vars,
            'vars': vars,
            'sigmas': sigmas
        }


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for Sparse Regression

    Applies focal weighting to MSE loss to emphasize hard samples (large errors).
    This helps the model focus on difficult predictions (non-zero emotion events)
    rather than easy predictions (zero values).

    The focal weight is: (1 + mse)^gamma
    - gamma > 1: Exponentially increases weight for larger errors
    - gamma = 0: Reduces to standard MSE

    Reference:
    - Inspired by Focal Loss (Lin et al., ICCV 2017) for classification
    - Adapted for regression tasks with continuous targets

    Args:
        gamma (float): Focusing parameter (default: 2.0)
                      Higher gamma = more focus on hard samples
        reduction (str): 'mean' or 'sum' or 'none' (default: 'mean')

    Example:
        >>> loss_fn = FocalMSELoss(gamma=2.0)
        >>> pred = torch.randn(32, 20, 7)  # (batch, time, emotions)
        >>> target = torch.randn(32, 20, 7)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute focal MSE loss

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
            target: Ground truth, same shape as pred

        Returns:
            Scalar loss value (if reduction='mean')
        """
        # Compute MSE per sample
        mse = (pred - target) ** 2

        # Apply focal weight: larger errors get exponentially larger weights
        focal_weight = (1.0 + mse) ** self.gamma

        # Weighted MSE
        focal_mse = focal_weight * mse

        # Reduction
        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        elif self.reduction == 'none':
            return focal_mse
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def __repr__(self):
        return f"FocalMSELoss(gamma={self.gamma}, reduction='{self.reduction}')"


class WeightedFocalMSELoss(nn.Module):
    """
    Weighted Focal MSE Loss combining focal weighting with zero/non-zero weighting

    Applies two types of weighting:
    1. Focal weight: (1 + mse)^gamma - emphasizes hard samples
    2. Sample weight: Higher weight for non-zero targets

    This is particularly effective for zero-inflated regression where we want to:
    - Focus on large errors (focal)
    - Focus on non-zero samples (sample weight)

    Args:
        gamma (float): Focal parameter (default: 2.0)
        zero_weight (float): Weight for zero targets (default: 1.0)
        nonzero_weight (float): Weight for non-zero targets (default: 5.0)
        reduction (str): 'mean' or 'sum' or 'none' (default: 'mean')

    Example:
        >>> loss_fn = WeightedFocalMSELoss(gamma=2.0, nonzero_weight=5.0)
        >>> pred = torch.randn(32, 20, 7)
        >>> target = torch.randn(32, 20, 7)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, gamma=2.0, zero_weight=1.0, nonzero_weight=5.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute weighted focal MSE loss

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
            target: Ground truth, same shape as pred

        Returns:
            Scalar loss value (if reduction='mean')
        """
        # Compute MSE per sample
        mse = (pred - target) ** 2

        # Apply focal weight
        focal_weight = (1.0 + mse) ** self.gamma

        # Apply zero/non-zero weight
        sample_weight = torch.where(
            target != 0,
            torch.tensor(self.nonzero_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.zero_weight, device=target.device, dtype=target.dtype)
        )

        # Combined weighting
        weighted_focal_mse = focal_weight * sample_weight * mse

        # Reduction
        if self.reduction == 'mean':
            return weighted_focal_mse.mean()
        elif self.reduction == 'sum':
            return weighted_focal_mse.sum()
        elif self.reduction == 'none':
            return weighted_focal_mse
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def __repr__(self):
        return (f"WeightedFocalMSELoss(gamma={self.gamma}, "
                f"zero_weight={self.zero_weight}, "
                f"nonzero_weight={self.nonzero_weight}, "
                f"reduction='{self.reduction}')")


class NormalizedFocalMSELoss(nn.Module):
    """
    Normalized Focal MSE Loss - Scale-Robust Focal Loss

    This variant normalizes the MSE by the target magnitude before applying focal weighting,
    making the loss scale-invariant and robust to different target ranges.

    Key improvement over standard FocalMSELoss:
    - Standard: focal_weight = (1 + mse)^gamma
    - Normalized: focal_weight = (1 + mse/(|target| + eps))^gamma

    Benefits:
    1. Scale-invariant: Works well across different emotion value ranges (e.g., Positive: 0-27, Sad: 0-5)
    2. Fair weighting: Large errors on small targets get same attention as large errors on large targets
    3. Prevents bias toward high-magnitude emotions

    Args:
        gamma (float): Focusing parameter (default: 1.0)
                      - gamma = 0: Equivalent to MSE
                      - gamma > 0: Emphasizes hard samples
        eps (float): Small constant for numerical stability (default: 1e-6)
        reduction (str): 'mean', 'sum', or 'none' (default: 'mean')

    Example:
        >>> loss_fn = NormalizedFocalMSELoss(gamma=1.0)
        >>> pred = torch.randn(32, 20, 7)
        >>> target = torch.randn(32, 20, 7).abs()  # Positive targets
        >>> loss = loss_fn(pred, target)

    Reference:
        Adapted from scale-robust focal loss variants for imbalanced regression
    """

    def __init__(self, gamma=1.0, eps=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute normalized focal MSE loss

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
            target: Ground truth, same shape as pred

        Returns:
            Scalar loss value (if reduction='mean')
        """
        # Compute MSE per sample
        mse = (pred - target) ** 2

        # Normalize MSE by target magnitude (key improvement!)
        normalized_mse = mse / (target.abs() + self.eps)

        # Apply focal weight on normalized MSE
        focal_weight = (1.0 + normalized_mse) ** self.gamma

        # Final loss: focal weight × original MSE
        focal_mse = focal_weight * mse

        # Reduction
        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        elif self.reduction == 'none':
            return focal_mse
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def __repr__(self):
        return f"NormalizedFocalMSELoss(gamma={self.gamma}, eps={self.eps}, reduction='{self.reduction}')"


class TweedieLoss(nn.Module):
    """
    Tweedie Loss for Zero-Inflated Regression

    The Tweedie distribution is designed for zero-inflated positive continuous data,
    making it perfect for sparse regression tasks where most targets are zero.

    The Tweedie distribution is a compound Poisson-Gamma distribution that naturally
    handles:
    - Point mass at zero (zero-inflation)
    - Positive continuous values (magnitude prediction)

    The power parameter p controls the variance-mean relationship:
    - p = 1: Poisson (count data)
    - 1 < p < 2: Compound Poisson-Gamma (zero-inflated continuous) ← Use this!
    - p = 2: Gamma (positive continuous)

    Reference:
    - "Optimizing Video Recommendation Systems: A Deep Dive into Tweedie Regression" (2024)
    - Industry validated: Tubi achieved +0.4% revenue, +0.15% watch time
    - Theoretical foundation: Lambert (1992), Dunn & Smyth (2005)

    Args:
        p (float): Power parameter (1 < p < 2, default: 1.5)
                  1.5 is a good starting point for zero-inflated data
        reduction (str): 'mean' or 'sum' or 'none' (default: 'mean')
        eps (float): Small constant for numerical stability (default: 1e-8)

    Example:
        >>> loss_fn = TweedieLoss(p=1.5)
        >>> pred = torch.randn(32, 20, 7).abs()  # Must be positive
        >>> target = torch.randn(32, 20, 7).abs()
        >>> target = torch.where(torch.rand_like(target) > 0.7, target, torch.zeros_like(target))
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, p=1.5, reduction='mean', eps=1e-8):
        super().__init__()
        if not (1.0 < p < 2.0):
            raise ValueError(f"Power parameter p must be in (1, 2), got {p}")

        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Compute Tweedie deviance loss

        The Tweedie deviance for observation (y, μ) is:
        d(y, μ) = -y * μ^(1-p) / (1-p) + μ^(2-p) / (2-p)

        This is derived from the log-likelihood of the Tweedie distribution.

        Args:
            pred: Predictions, shape (batch, seq_len, num_emotions) or (batch, num_emotions)
                  Must be non-negative (use softplus or exp activation)
            target: Ground truth, same shape as pred
                   Can contain zeros (zero-inflation) and positive values

        Returns:
            Scalar loss value (if reduction='mean')
        """
        # Ensure predictions are positive (add small epsilon for stability)
        pred = torch.clamp(pred, min=0) + self.eps

        # Compute Tweedie deviance
        # Term 1: -y * μ^(1-p) / (1-p)
        # Term 2: μ^(2-p) / (2-p)

        term1 = -target * torch.pow(pred, 1 - self.p) / (1 - self.p)
        term2 = torch.pow(pred, 2 - self.p) / (2 - self.p)

        deviance = term1 + term2

        # Reduction
        if self.reduction == 'mean':
            return deviance.mean()
        elif self.reduction == 'sum':
            return deviance.sum()
        elif self.reduction == 'none':
            return deviance
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def __repr__(self):
        return f"TweedieLoss(p={self.p}, reduction='{self.reduction}', eps={self.eps})"


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Testing Learnable Loss Functions")
    print("="*80)

    # Test data
    batch_size = 8
    seq_len = 20
    num_emotions = 7

    pred = torch.randn(batch_size, seq_len, num_emotions)
    target = torch.randn(batch_size, seq_len, num_emotions)
    # Make target sparse (simulate emotion data)
    target = torch.where(torch.rand_like(target) > 0.7, target, torch.zeros_like(target))

    print(f"\nTest data shape: {pred.shape}")
    print(f"Target sparsity: {(target == 0).float().mean().item():.2%}")

    # Test Option 2: Per-Emotion Learnable Weighted MSE
    print("\n" + "-"*80)
    print("Option 2: Per-Emotion Learnable Weighted MSE")
    print("-"*80)
    loss_fn_2 = PerEmotionLearnableWeightedMSE(num_emotions=num_emotions)
    loss_2 = loss_fn_2(pred, target)
    print(f"Loss: {loss_2.item():.4f}")
    print(f"Initial weights: {loss_fn_2.get_weights()}")

    # Test Option 3: Uncertainty-based Weighted MSE
    print("\n" + "-"*80)
    print("Option 3: Uncertainty-based Weighted MSE")
    print("-"*80)
    loss_fn_3 = UncertaintyWeightedMSE(num_emotions=num_emotions)
    loss_3 = loss_fn_3(pred, target)
    print(f"Loss: {loss_3.item():.4f}")
    print(f"Initial uncertainties: {loss_fn_3.get_uncertainties()}")

    # Test Focal MSE Loss
    print("\n" + "-"*80)
    print("Focal MSE Loss")
    print("-"*80)
    loss_fn_focal = FocalMSELoss(gamma=2.0)
    loss_focal = loss_fn_focal(pred, target)
    print(f"Loss: {loss_focal.item():.4f}")
    print(f"Gamma: {loss_fn_focal.gamma}")

    # Test Weighted Focal MSE Loss
    print("\n" + "-"*80)
    print("Weighted Focal MSE Loss")
    print("-"*80)
    loss_fn_wfocal = WeightedFocalMSELoss(gamma=2.0, nonzero_weight=5.0)
    loss_wfocal = loss_fn_wfocal(pred, target)
    print(f"Loss: {loss_wfocal.item():.4f}")
    print(f"Gamma: {loss_fn_wfocal.gamma}, Nonzero weight: {loss_fn_wfocal.nonzero_weight}")

    # Test Tweedie Loss
    print("\n" + "-"*80)
    print("Tweedie Loss")
    print("-"*80)
    # Make predictions and targets positive for Tweedie
    pred_pos = torch.abs(pred)
    target_pos = torch.abs(target)
    target_pos = torch.where(torch.rand_like(target_pos) > 0.7, target_pos, torch.zeros_like(target_pos))

    loss_fn_tweedie = TweedieLoss(p=1.5)
    loss_tweedie = loss_fn_tweedie(pred_pos, target_pos)
    print(f"Loss: {loss_tweedie.item():.4f}")
    print(f"Power parameter p: {loss_fn_tweedie.p}")
    print(f"Target sparsity: {(target_pos == 0).float().mean().item():.2%}")

    # Test backward pass
    print("\n" + "-"*80)
    print("Testing backward pass")
    print("-"*80)

    # Create fresh tensors with grad enabled
    pred_grad = torch.randn(batch_size, seq_len, num_emotions, requires_grad=True)
    target_grad = torch.randn(batch_size, seq_len, num_emotions)
    target_grad = torch.where(torch.rand_like(target_grad) > 0.7, target_grad, torch.zeros_like(target_grad))

    loss_2 = loss_fn_2(pred_grad, target_grad)
    loss_2.backward()
    print(f"✓ Option 2 backward pass successful")

    pred_grad = torch.randn(batch_size, seq_len, num_emotions, requires_grad=True)
    loss_3 = loss_fn_3(pred_grad, target_grad)
    loss_3.backward()
    print(f"✓ Option 3 backward pass successful")

    pred_grad = torch.randn(batch_size, seq_len, num_emotions, requires_grad=True)
    loss_focal = loss_fn_focal(pred_grad, target_grad)
    loss_focal.backward()
    print(f"✓ Focal MSE backward pass successful")

    pred_grad = torch.randn(batch_size, seq_len, num_emotions, requires_grad=True)
    loss_wfocal = loss_fn_wfocal(pred_grad, target_grad)
    loss_wfocal.backward()
    print(f"✓ Weighted Focal MSE backward pass successful")

    pred_grad = torch.abs(torch.randn(batch_size, seq_len, num_emotions, requires_grad=True))
    target_grad_pos = torch.abs(target_grad)
    target_grad_pos = torch.where(torch.rand_like(target_grad_pos) > 0.7, target_grad_pos, torch.zeros_like(target_grad_pos))
    loss_tweedie = loss_fn_tweedie(pred_grad, target_grad_pos)
    loss_tweedie.backward()
    print(f"✓ Tweedie Loss backward pass successful")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
