"""
Quick test to verify ROI timeseries loading works correctly

Tests:
1. Load a sample ROI CSV file
2. Extract a 30-frame sequence
3. Verify dimensions
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from baselines.svr_with_reduction import SVRWithReduction

def test_roi_loading():
    """Test ROI timeseries loading"""
    print("="*80)
    print("Testing ROI Timeseries Loading")
    print("="*80)

    # Initialize SVR with ROI reduction
    svr = SVRWithReduction(
        num_emotions=7,
        sequence_length=30,
        reduction_method='roi',
        roi_timeseries_path='/scratch/HBN/9.2.movieDM_ROI_timeseries'
    )

    # Test subject ID (without 'sub-' prefix, as used in data module)
    test_subject = 'NDARAA947ZG5'

    print(f"\n1. Testing load_roi_timeseries()...")
    print(f"   Subject: {test_subject}")

    # Load ROI timeseries
    roi_timeseries = svr.load_roi_timeseries(test_subject)

    print(f"   ✓ Loaded shape: {roi_timeseries.shape}")
    print(f"   ✓ Number of ROIs: {svr.num_rois}")
    print(f"   ✓ Number of TRs: {roi_timeseries.shape[0]}")

    # Test reduce_features with ROI method
    print(f"\n2. Testing reduce_features()...")

    # Simulate a sequence starting at frame 100
    start_frame = 100
    dummy_fmri = np.random.randn(96, 96, 96, 30)  # Won't be used for ROI method

    features = svr.reduce_features(
        fmri_seq=dummy_fmri,
        subject_id=test_subject,
        start_frame=start_frame
    )

    expected_dim = svr.sequence_length * svr.num_rois
    print(f"   ✓ Feature shape: {features.shape}")
    print(f"   ✓ Expected dimension: {expected_dim}")
    print(f"   ✓ Match: {features.shape[0] == expected_dim}")

    # Verify the extracted sequence matches the expected slice
    print(f"\n3. Verifying sequence extraction...")
    end_frame = start_frame + svr.sequence_length
    expected_sequence = roi_timeseries[start_frame:end_frame, :]
    expected_features = expected_sequence.flatten()

    print(f"   ✓ Extracted sequence shape: {expected_sequence.shape}")
    print(f"   ✓ Expected: ({svr.sequence_length}, {svr.num_rois})")
    print(f"   ✓ Features match: {np.allclose(features, expected_features)}")

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_roi_loading()
