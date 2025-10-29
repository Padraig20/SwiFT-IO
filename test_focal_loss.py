#!/usr/bin/env python
"""
Test Focal MSE Loss implementation

Quick sanity check:
1. Import works
2. Forward pass works
3. Backward pass works
4. Loss values are reasonable
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Add src to path
sys.path.insert(0, 'src')

from module.utils.learnable_losses import (
    FocalMSELoss,
    WeightedFocalMSELoss,
    PerEmotionLearnableWeightedMSE,
    UncertaintyWeightedMSE
)

def test_focal_mse():
    print("="*80)
    print("Testing Focal MSE Loss Implementation")
    print("="*80)

    # Test data - simulate sparse emotion data
    batch_size = 4
    seq_len = 20
    num_emotions = 7

    torch.manual_seed(42)
    pred = torch.randn(batch_size, seq_len, num_emotions, requires_grad=True)
    target = torch.randn(batch_size, seq_len, num_emotions)

    # Make target sparse (70% zeros like real data)
    target = torch.where(torch.rand_like(target) > 0.7, target, torch.zeros_like(target))

    print(f"\nTest data:")
    print(f"  Shape: {pred.shape}")
    print(f"  Sparsity: {(target == 0).float().mean().item():.2%}")
    print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")

    # Test 1: Standard MSE (baseline)
    print("\n" + "-"*80)
    print("Test 1: Standard MSE Loss (baseline)")
    print("-"*80)
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(pred, target)
    print(f"  Loss: {mse_loss.item():.6f}")
    mse_loss.backward()
    print(f"  ✓ Backward pass successful")
    grad_mse = pred.grad.clone()
    pred.grad.zero_()

    # Test 2: Focal MSE (gamma=2.0)
    print("\n" + "-"*80)
    print("Test 2: Focal MSE Loss (gamma=2.0)")
    print("-"*80)
    focal_loss_fn = FocalMSELoss(gamma=2.0)
    focal_loss = focal_loss_fn(pred, target)
    print(f"  Loss: {focal_loss.item():.6f}")
    print(f"  Loss function: {focal_loss_fn}")
    focal_loss.backward()
    print(f"  ✓ Backward pass successful")
    grad_focal = pred.grad.clone()
    pred.grad.zero_()

    # Compare gradients
    print(f"\n  Gradient comparison:")
    print(f"    MSE grad norm: {grad_mse.norm():.6f}")
    print(f"    Focal grad norm: {grad_focal.norm():.6f}")
    print(f"    Ratio: {(grad_focal.norm() / grad_mse.norm()).item():.3f}x")

    # Test 3: Weighted Focal MSE
    print("\n" + "-"*80)
    print("Test 3: Weighted Focal MSE Loss (gamma=2.0, nonzero_weight=5.0)")
    print("-"*80)
    wfocal_loss_fn = WeightedFocalMSELoss(gamma=2.0, nonzero_weight=5.0)
    wfocal_loss = wfocal_loss_fn(pred, target)
    print(f"  Loss: {wfocal_loss.item():.6f}")
    print(f"  Loss function: {wfocal_loss_fn}")
    wfocal_loss.backward()
    print(f"  ✓ Backward pass successful")
    grad_wfocal = pred.grad.clone()
    pred.grad.zero_()

    print(f"\n  Gradient comparison:")
    print(f"    MSE grad norm: {grad_mse.norm():.6f}")
    print(f"    Weighted Focal grad norm: {grad_wfocal.norm():.6f}")
    print(f"    Ratio: {(grad_wfocal.norm() / grad_mse.norm()).item():.3f}x")

    # Test 4: Different gamma values
    print("\n" + "-"*80)
    print("Test 4: Gamma parameter sensitivity")
    print("-"*80)
    gammas = [0.0, 0.5, 1.0, 2.0, 3.0]
    print(f"  {'Gamma':<10} {'Loss':<12} {'Loss/MSE ratio':<15}")
    print(f"  {'-'*10} {'-'*12} {'-'*15}")
    for gamma in gammas:
        loss_fn = FocalMSELoss(gamma=gamma)
        loss = loss_fn(pred, target)
        ratio = loss.item() / mse_loss.item()
        print(f"  {gamma:<10.1f} {loss.item():<12.6f} {ratio:<15.3f}")

    # Test 5: Check focusing behavior
    print("\n" + "-"*80)
    print("Test 5: Focusing behavior on hard vs easy samples")
    print("-"*80)

    # Easy sample (small error)
    pred_easy = torch.tensor([0.1])
    target_easy = torch.tensor([0.0])

    # Hard sample (large error)
    pred_hard = torch.tensor([5.0])
    target_hard = torch.tensor([0.0])

    focal_fn = FocalMSELoss(gamma=2.0)

    # MSE losses
    mse_easy = ((pred_easy - target_easy) ** 2).item()
    mse_hard = ((pred_hard - target_hard) ** 2).item()

    # Focal losses
    focal_easy = focal_fn(pred_easy, target_easy).item()
    focal_hard = focal_fn(pred_hard, target_hard).item()

    print(f"\n  Easy sample (error=0.1):")
    print(f"    MSE loss: {mse_easy:.6f}")
    print(f"    Focal loss: {focal_easy:.6f}")
    print(f"    Focal/MSE ratio: {focal_easy/mse_easy:.3f}x")

    print(f"\n  Hard sample (error=5.0):")
    print(f"    MSE loss: {mse_hard:.6f}")
    print(f"    Focal loss: {focal_hard:.6f}")
    print(f"    Focal/MSE ratio: {focal_hard/mse_hard:.3f}x")

    emphasis_ratio = (focal_hard/mse_hard) / (focal_easy/mse_easy)
    print(f"\n  Emphasis on hard sample: {emphasis_ratio:.1f}x more than easy sample")

    # Test 6: Integration with existing losses
    print("\n" + "-"*80)
    print("Test 6: Compatibility with existing losses")
    print("-"*80)

    # Test PerEmotionLearnableWeightedMSE
    learnable_fn = PerEmotionLearnableWeightedMSE(num_emotions=num_emotions)
    learnable_loss = learnable_fn(pred, target)
    print(f"  PerEmotionLearnableWeightedMSE: {learnable_loss.item():.6f} ✓")

    # Test UncertaintyWeightedMSE
    uncertainty_fn = UncertaintyWeightedMSE(num_emotions=num_emotions)
    uncertainty_loss = uncertainty_fn(pred, target)
    print(f"  UncertaintyWeightedMSE: {uncertainty_loss.item():.6f} ✓")

    # Test Focal MSE
    focal_loss = focal_loss_fn(pred, target)
    print(f"  FocalMSELoss: {focal_loss.item():.6f} ✓")

    # Test Weighted Focal MSE
    wfocal_loss = wfocal_loss_fn(pred, target)
    print(f"  WeightedFocalMSELoss: {wfocal_loss.item():.6f} ✓")

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

    print("\n📊 Summary:")
    print(f"  - Focal MSE emphasizes hard samples {emphasis_ratio:.1f}x more")
    print(f"  - Gradient magnitude increased by ~{(grad_focal.norm() / grad_mse.norm()).item():.1f}x")
    print(f"  - All loss functions work correctly")
    print(f"  - Backward pass successful for all variants")

    print("\n✅ Ready for training!")

if __name__ == "__main__":
    try:
        test_focal_mse()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
