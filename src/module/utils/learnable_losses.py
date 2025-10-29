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

    # Test backward pass
    print("\n" + "-"*80)
    print("Testing backward pass")
    print("-"*80)
    loss_2.backward()
    print(f"✓ Option 2 backward pass successful")

    loss_3.backward()
    print(f"✓ Option 3 backward pass successful")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
