"""
Quick test to verify ROI common column extraction works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baselines.svr_with_reduction import SVRWithReduction

print("="*80)
print("Testing ROI Common Column Extraction")
print("="*80)

# Initialize SVR with ROI reduction
svr = SVRWithReduction(
    num_emotions=7,
    sequence_length=30,
    reduction_method='roi',
    roi_atlas='aal',
    roi_timeseries_path='/scratch/HBN/9.2.movieDM_ROI_timeseries',
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    standardize=True
)

print("\n[Step 1] Testing load_roi_timeseries on sample subjects...")
print("  (This will automatically find common ROIs on first load)")
test_subjects = ['NDARAA947ZG5', 'NDARAB458VK9', 'NDARAC349YUC']

for subj in test_subjects:
    roi_ts = svr.load_roi_timeseries(subj)
    print(f"  {subj}: shape = {roi_ts.shape}")

print(f"\n[Step 2] Found {svr.num_rois} common ROIs")
print(f"First 10 ROIs: {svr.common_roi_columns[:10]}")
print(f"Last 10 ROIs: {svr.common_roi_columns[-10:]}")

print("\n[Step 3] Verifying all subjects have same shape...")
all_same_shape = True
expected_shape = (750, svr.num_rois)

for subj in test_subjects:
    roi_ts = svr.load_roi_timeseries(subj)  # Uses cache
    if roi_ts.shape != expected_shape:
        print(f"  ERROR: {subj} has shape {roi_ts.shape}, expected {expected_shape}")
        all_same_shape = False

if all_same_shape:
    print(f"  ✓ All subjects have consistent shape: {expected_shape}")
    print(f"  ✓ Feature dimension: {svr.num_rois} × 30 = {svr.num_rois * 30}")
else:
    print(f"  ✗ Shape mismatch detected!")

print("\n" + "="*80)
print("Test Complete!")
print("="*80)
