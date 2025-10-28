"""
Quick test script for SVR baseline

Tests basic functionality without full training
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
from baselines.svr_baseline import SVRBaseline

def test_svr_baseline():
    """Test SVR baseline with synthetic data"""

    print("="*80)
    print("Testing SVR Baseline")
    print("="*80)

    # Initialize SVR baseline
    print("\n[1] Initializing SVR baseline...")
    svr = SVRBaseline(
        num_emotions=7,
        sequence_length=30,
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
        standardize=True
    )
    print("✓ SVR baseline initialized successfully")

    # Test data loading and flattening
    print("\n[2] Testing data processing...")

    # Create synthetic 4D fMRI data
    batch_size = 2
    fmri_data = np.random.randn(96, 96, 96, 30)

    # Flatten sequence
    features = svr.flatten_sequence(fmri_data)
    print(f"  Original shape: (96, 96, 96, 30)")
    print(f"  Flattened shape: {features.shape}")
    print(f"  Expected: (30, {96*96*96})")

    if features.shape == (30, 96*96*96):
        print("✓ Data flattening works correctly")
    else:
        print("✗ Data flattening failed")
        return False

    # Test with small synthetic dataset
    print("\n[3] Testing training with synthetic data...")

    # Create small synthetic dataset
    num_samples = 100
    num_voxels = 1000  # Use fewer voxels for quick test
    num_emotions = 7

    X_train = np.random.randn(num_samples, num_voxels)
    Y_train = np.random.randn(num_samples, num_emotions)

    print(f"  Synthetic train data: X={X_train.shape}, Y={Y_train.shape}")

    # Create simple SVR with fewer features
    svr_test = SVRBaseline(
        num_emotions=num_emotions,
        sequence_length=30,
        kernel='linear',  # Use linear for speed
        C=1.0,
        epsilon=0.1,
        standardize=True
    )
    svr_test.feature_dim = num_voxels

    # Train on first emotion only (for speed)
    print("  Training SVR on first emotion (this may take a moment)...")

    if svr_test.standardize:
        X_train_scaled = svr_test.scalers[0].fit_transform(X_train)
    else:
        X_train_scaled = X_train

    svr_test.models[0].fit(X_train_scaled, Y_train[:, 0])

    # Predict
    y_pred = svr_test.models[0].predict(X_train_scaled)

    print(f"  Prediction shape: {y_pred.shape}")
    print(f"  Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

    if y_pred.shape == (num_samples,):
        print("✓ SVR training and prediction works correctly")
    else:
        print("✗ SVR prediction failed")
        return False

    # Test save/load
    print("\n[4] Testing save/load functionality...")

    test_path = "test_svr_model.pkl"
    svr_test.fitted = True
    svr_test.save(test_path)

    if os.path.exists(test_path):
        print(f"✓ Model saved to {test_path}")

        # Load model
        svr_test_loaded = SVRBaseline(num_emotions=num_emotions, sequence_length=30)
        svr_test_loaded.load(test_path)

        if svr_test_loaded.fitted:
            print("✓ Model loaded successfully")
        else:
            print("✗ Model loading failed")
            return False

        # Clean up
        os.remove(test_path)
        print("  Cleaned up test file")
    else:
        print("✗ Model save failed")
        return False

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nSVR baseline is ready to use.")
    print("\nTo train with real data, run:")
    print("  bash run_svr_baseline.sh")
    print("or")
    print("  python src/train_svr_baseline.py --image_path /scratch/HBN/9.2.movieDM_SwiFT ...")

    return True

if __name__ == "__main__":
    test_svr_baseline()
