# SVR with GLM Significant Voxel Mask - Implementation Guide

## Overview

This document describes the implementation of SVR baseline using GLM-identified significant voxels for emotion prediction from fMRI data.

## Architecture

```
4D fMRI Input (81×95×81×T)
    ↓
GLM Union Mask (44,303 sig voxels)
    ↓
┌─────────────────────────────────────┐
│                                     │
│  glm_pca mode        glm_direct    │
│  ↓                   mode ↓        │
│  PCA (100 comp)      Time-average  │
│  per timepoint                     │
│  ↓                   ↓             │
│  Concat across       Flatten       │
│  time                              │
│  ↓                   ↓             │
│  (T×100,)            (44303,)      │
└─────────────────────────────────────┘
    ↓
SVR (per emotion, 7 models)
    ↓
Predictions + Non-zero Metrics
```

## Key Files

| File | Description |
|------|-------------|
| `src/baselines/nonzero_metrics.py` | Non-zero metrics calculator (matches pl_classifier.py) |
| `src/baselines/svr_with_glm_mask.py` | SVRWithGLMMask class (glm_pca, glm_direct modes) |
| `src/baselines/svr_with_reduction.py` | Original SVR with PCA/ROI/time_avg (updated with NonZeroMetrics) |
| `src/train_svr_with_glm_mask.py` | Training script for GLM mask SVR |
| `sample_scripts/svr_baselines/*.sh` | SLURM submission scripts |

## GLM Mask Details

### Source
- Path: `/scratch/connectome/kimbo/GLM-Baseline-Test/results/full_analysis/smooth_motion/threshold_nonparam/sig_masks_for_ridge/`
- Generated from 2nd level GLM analysis on train set
- Uses split: `split_fixed_2_stratified_Age_Sex.txt`

### Mask Statistics
```
Union mask shape: (81, 95, 81)
Total voxels: 589,815
Significant voxels: 44,303 (~7.5%)

Per-emotion significant voxels:
  Anger:    16,053
  Happy:     1,959
  Fear:      1,427
  Sad:      14,193
  Excited:  16,093
  Positive:  8,261
  Negative: 37,060
```

## Non-Zero Metrics

Matches `pl_classifier.py` exactly (lines 598-800):

### Calculation Method
```python
# 1. Inverse transform to original scale
target_original = target_scaled * scaler.scale_[0] + scaler.mean_[0]

# 2. Create mask (epsilon = 1e-6)
mask_nonzero = np.abs(target_original) >= 1e-6

# 3. Calculate metrics on non-zero samples only
nonzero_mae = mean_absolute_error(y_true[mask_nonzero], y_pred[mask_nonzero])
```

### Metrics Computed
- **Overall**: mse, mae, corrcoef, r2, adjusted_mse, adjusted_mae
- **Non-zero**: nonzero_mae, nonzero_mse, nonzero_rmse, nonzero_pearson
- **Detection**: detection_tpr, detection_fpr, detection_precision, detection_f1, detection_auroc
- **Magnitude**: small_mae, medium_mae, large_mae
- **Zero**: zero_mae, zero_mean_pred, zero_std_pred

## Usage

### Training

```bash
# GLM PCA + SVR (recommended for temporal info)
python src/train_svr_with_glm_mask.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --reduction_mode glm_pca \
    --pca_components 100 \
    --dataset_split_seed 2 \
    --stratified_params Age Sex \
    --output_dir output/svr_glm_pca_seq20

# GLM Direct + SVR (simpler, faster)
python src/train_svr_with_glm_mask.py \
    --reduction_mode glm_direct \
    --output_dir output/svr_glm_direct_seq20
```

### SLURM Submission

```bash
sbatch sample_scripts/svr_baselines/train_svr_glm_pca_stratified.sh
sbatch sample_scripts/svr_baselines/train_svr_glm_direct_stratified.sh
```

## Comparison with Other Baselines

| Baseline | Features | Dimension | Description |
|----------|----------|-----------|-------------|
| Whole brain PCA | All voxels + PCA | T×100 = 2,000 | Full brain temporal |
| Whole brain time_avg | All voxels + avg | 589,815 | Full brain static |
| **GLM PCA** | Sig voxels + PCA | T×100 = 2,000 | Task-relevant temporal |
| **GLM Direct** | Sig voxels + avg | 44,303 | Task-relevant static |

## Output Files

```
output/svr_glm_[pca|direct]_seq20/
├── svr_glm_mask_metrics.json     # All metrics including non-zero
├── svr_glm_mask_model.pkl        # Trained model
├── svr_glm_mask_config.json      # Configuration
├── training_summary.txt          # Human-readable summary
├── glm_pca_model_checkpoint.pkl  # PCA checkpoint (glm_pca only)
├── glm_train_data_checkpoint.pkl # Data checkpoint
└── glm_svr_emotion_*_checkpoint.pkl  # Per-emotion checkpoints
```

## Implementation Notes

### fMRI Data Format
- Files: `.pt` PyTorch tensors (not NIfTI)
- Shape: (81, 95, 81, 1) per frame
- 750 frames per subject

### GLM Mask Format
- Files: `.nii.gz` NIfTI
- Shape: (81, 95, 81) - **matches fMRI exactly!**
- No resampling needed

### Memory Optimization
- IncrementalPCA for batch-wise fitting
- float32 conversion (50% memory reduction)
- Checkpoint saving for resume capability

## Date Created
2024-12-09
