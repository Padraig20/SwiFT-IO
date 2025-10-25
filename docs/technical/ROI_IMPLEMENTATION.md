# ROI Implementation for SVR Baseline

## Summary

Successfully modified the SVR with reduction baseline to use precomputed ROI timeseries data instead of voxel-based ROI extraction.

## Changes Made

### 1. Modified `SVRWithReduction` class (`src/baselines/svr_with_reduction.py`)

#### Added ROI timeseries loading:
- New parameter: `roi_timeseries_path` (default: `/scratch/HBN/9.2.movieDM_ROI_timeseries`)
- New method: `load_roi_timeseries(subject_id)` - loads precomputed CSV files
- ROI cache: stores loaded timeseries in memory for efficiency

#### Updated `reduce_features()` method:
- Now accepts optional `subject_id` and `start_frame` parameters
- For ROI method: loads full 750-TR timeseries and extracts the 30-frame sequence
- For PCA/time_avg: uses voxel data as before

#### Updated `prepare_data_from_dataloader()` method:
- Extracts `subject_name` and `TR` (start frame) from batch
- Passes these to `reduce_features()` for ROI method

#### Updated save/load methods:
- Now saves/loads `roi_timeseries_path`
- Resets ROI cache on load

### 2. Updated SLURM script (`sample_scripts/run_svr_roi.slurm`)

- Updated comments to reflect precomputed ROI source
- Corrected feature dimension: 3,270 (30 × 109 ROIs)

### 3. Created test script (`test_roi_loading.py`)

- Verifies ROI loading works correctly
- Tests sequence extraction
- All tests passed ✓

## ROI Data Structure

### CSV Files
- **Location**: `/scratch/HBN/9.2.movieDM_ROI_timeseries/`
- **Naming**: `sub-{SUBJECT_ID}_movieDM_roi_temporal_activity.csv`
- **Format**:
  - Column 0: TR index (0-749)
  - Columns 1-109: ROI values (109 ROIs from FreeSurfer parcellation)
  - Total: 750 TRs × 109 ROIs

### Subject ID Handling
- **Data module**: Uses subject IDs without 'sub-' prefix (e.g., 'NDARAA947ZG5')
- **CSV files**: Include 'sub-' prefix in filename
- **Solution**: Code automatically adds 'sub-' prefix when loading

## Feature Dimensions

| Method     | Dimension | Description |
|------------|-----------|-------------|
| time_avg   | 884,736   | Single static brain pattern (96³) |
| PCA        | 3,000     | 100 components × 30 timepoints |
| **ROI**    | **3,270** | **109 ROIs × 30 timepoints** |

## Usage

```python
from baselines.svr_with_reduction import SVRWithReduction

# Initialize with ROI method
svr = SVRWithReduction(
    num_emotions=7,
    sequence_length=30,
    reduction_method='roi',
    roi_timeseries_path='/scratch/HBN/9.2.movieDM_ROI_timeseries'
)

# Train (will automatically load ROI timeseries as needed)
svr.fit(train_dataloader)

# Evaluate
svr.evaluate(test_dataloader, mode='test')
```

## Testing

Run the test script to verify implementation:
```bash
conda activate swiftio
python test_roi_loading.py
```

## Training

Submit ROI baseline training job:
```bash
sbatch sample_scripts/run_svr_roi.slurm
```

Expected output directory: `output/svr_reduction_roi/`

## Advantages of Precomputed ROI Approach

1. **Efficiency**: No need to load full voxel data
2. **Consistency**: Uses the same FreeSurfer parcellation for all subjects
3. **Simplicity**: Direct CSV loading is faster than voxel-based ROI extraction
4. **Memory**: Much lower memory footprint (750 × 109 vs 96³ × 30)

## Next Steps

1. ✓ Implementation complete
2. ✓ Testing passed
3. ⏳ Run full training with `sbatch sample_scripts/run_svr_roi.slurm`
4. ⏳ Compare results with time_avg and PCA baselines
