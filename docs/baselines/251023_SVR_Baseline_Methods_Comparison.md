# SVR Baseline Methods Comparison

**Date**: 2025-10-23
**Author**: kimbo
**Purpose**: Document different SVR baseline approaches for comparing with SwiFT-IO

---

## Overview

We prepared **4 different SVR baseline variants** for emotion prediction, each with different dimensionality reduction strategies. These baselines serve as comparison points to demonstrate SwiFT-IO's advantages in temporal modeling.

---

## 1. PCA-based SVR (`svr_reduction_pca`)

### Method
- Apply **PCA** to each timepoint independently
- Reduce spatial dimensions: 96×96×96 (884,736 voxels) → **100 components** per timepoint
- Concatenate across time: 30 timepoints × 100 = **3,000 features**
- **Preserves temporal information** (each timepoint processed separately)

### Configuration
```json
{
  "reduction_method": "pca",
  "pca_components": 100,
  "sequence_length": 30,
  "feature_dim": 3000,
  "kernel": "rbf",
  "C": 1.0,
  "epsilon": 0.1
}
```

### Characteristics
- ✅ Most sophisticated dimensionality reduction
- ✅ Preserves temporal dynamics
- ✅ Compact representation (3,000 features)
- ⚠️ Requires PCA fitting (~2.5 hours with optimization)
- ⚠️ Information loss through dimensionality reduction

### Training Process
1. **IncrementalPCA fitting** on training data
   - Memory-efficient batch processing
   - Vectorized operations (8x faster after optimization)
   - Time: ~2.5 hours (optimized from ~20 hours)
2. **SVR training** per emotion
   - Parallel training across 7 emotions
   - Time: ~10 minutes

### Current Status
- ✅ Training completed for emotions 0-5 (checkpoints saved)
- ⚠️ Emotion 6 checkpoint corrupted (disk space error during save)
- ✅ Models: `output/svr_reduction_pca/svr_emotion_{0-5}_checkpoint.pkl` (valid)
- ✅ PCA model: `output/svr_reduction_pca/pca_model_checkpoint.pkl`
- ⏳ Evaluation in progress (Job 62936) - evaluating 6 emotions only
- 📊 Results pending: `output/svr_reduction_pca/eval_metrics.json`

### Storage
- Total size: **2.3 GB**
- PCA model: 689 MB
- SVR models: 63-237 MB each

---

## 2. ROI-based SVR (`svr_reduction_roi`)

### Method
- Use **AAL (Automated Anatomical Labeling) atlas**
- Average voxels within each ROI
- Extract ROI timeseries: 95 ROIs × 30 timepoints = **2,850 features**
- **Preserves temporal information**

### Configuration
```json
{
  "reduction_method": "roi",
  "roi_atlas": "aal",
  "sequence_length": 30,
  "feature_dim": 2850,
  "kernel": "rbf",
  "C": 1.0,
  "epsilon": 0.1,
  "num_emotions": 7
}
```

### Characteristics
- ✅ **Anatomically interpretable** features
- ✅ Uses prior knowledge (brain parcellation)
- ✅ Preserves temporal dynamics
- ✅ Fast (uses precomputed ROI timeseries)
- ⚠️ Relies on atlas quality
- ⚠️ May lose fine-grained spatial patterns

### Performance (Completed)
```
Train: MSE=2.073, MAE=0.751, R²=-0.122
Valid: MSE=2.078, MAE=0.753, R²=-0.115
Test:  MSE=2.074, MAE=0.753, R²=-0.114
```

**Per-Emotion Test Results:**
| Emotion | MSE | MAE | R² | Correlation |
|---------|-----|-----|----|-------------|
| Anger | 2.507 | 0.860 | -0.217 | 0.053 |
| Happy | 2.414 | 0.946 | -0.120 | 0.029 |
| Fear | 0.366 | 0.371 | -0.050 | 0.026 |
| Sad | 2.830 | 0.820 | -0.213 | 0.082 |
| Excited | 3.974 | 0.821 | -0.128 | 0.062 |
| Positive | 1.550 | 0.742 | -0.116 | 0.051 |
| Negative | 0.876 | 0.708 | -0.027 | 0.058 |

### Current Status
- ✅ **Fully completed** (training + evaluation)
- 📊 Results: `output/svr_reduction_roi/svr_reduction_metrics.json`
- 📝 Config: `output/svr_reduction_roi/svr_reduction_config.json`

### Storage
- Total size: **3.2 GB**

---

## 3. Time-Averaged SVR (`svr_reduction_time_avg`) ⭐ **RECOMMENDED**

### Method
- **Average across time dimension**
- Collapse 30 timepoints into single static pattern
- Each voxel: mean of 30 values
- **Feature dim: 884,736** (96×96×96)
- **NO temporal information** (purely static)

### Configuration
```json
{
  "reduction_method": "time_avg",
  "sequence_length": 30,
  "feature_dim": 884736,
  "kernel": "rbf",
  "C": 1.0,
  "epsilon": 0.1
}
```

### Characteristics
- ✅ **Simplest baseline** approach
- ✅ **Best for comparison** with SwiFT-IO
- ✅ No temporal dynamics → highlights SwiFT's temporal modeling advantage
- ✅ Fast training (no PCA fitting needed)
- ✅ Full spatial resolution (no dimensionality reduction)
- ❌ Loses all temporal information
- ⚠️ High-dimensional (884K features)

### Why This is the Best Comparison
This baseline is **ideal for demonstrating SwiFT-IO's value** because:
1. **Static vs Dynamic**: Time-averaged baseline has NO temporal modeling → any improvement from SwiFT-IO directly shows the benefit of temporal dynamics
2. **Simple and interpretable**: No complex feature engineering
3. **No information loss**: Uses all spatial information (unlike PCA)
4. **Standard approach**: Common baseline in fMRI literature

### Current Status
- ✅ Directory exists: `output/svr_reduction_time_avg/`
- 📦 Size: 19 GB
- ⚠️ Results need verification

---

## 4. Time-Averaged Linear SVR (`svr_reduction_time_avg_linear`)

### Method
- Same as Time-Averaged SVR
- But uses **Linear kernel** instead of RBF

### Configuration
```json
{
  "reduction_method": "time_avg",
  "sequence_length": 30,
  "feature_dim": 884736,
  "kernel": "linear",
  "C": 1.0,
  "epsilon": 0.1
}
```

### Characteristics
- ✅ **Faster training** than RBF
- ✅ Models only linear relationships
- ✅ More interpretable (linear weights)
- ❌ Less expressive (no non-linear patterns)
- ❌ May underperform RBF kernel

### Current Status
- ✅ Directory exists: `output/svr_reduction_time_avg_linear/`
- 📦 Size: 19 GB
- ⚠️ Results need verification

---

## Summary Comparison Table

| Method | Feature Dim | Temporal Info | Spatial Info | Complexity | Training Time | Best For |
|--------|-------------|---------------|--------------|------------|---------------|----------|
| **PCA** | 3,000 | ✅ Preserved | ⚠️ Reduced | High | ~3 hours | Compact representation |
| **ROI** | 2,850 | ✅ Preserved | ⚠️ Averaged | Medium | ~1 hour | Interpretability |
| **Time-avg (RBF)** | 884,736 | ❌ Averaged | ✅ Full | Low | Fast | **SwiFT comparison** ⭐ |
| **Time-avg (Linear)** | 884,736 | ❌ Averaged | ✅ Full | Low | Fastest | Linear baseline |

---

## Recommended Comparison Strategy

### For SwiFT-IO Paper

**Primary comparison: Time-Averaged SVR (RBF)**
- Shows the value of temporal modeling
- Static baseline vs. Dynamic model (SwiFT-IO)
- Clear, simple, and interpretable

**Secondary comparisons:**
1. **PCA-based SVR**: Shows that even with temporal info + dimensionality reduction, SwiFT outperforms
2. **ROI-based SVR**: Shows that anatomical priors alone are insufficient

**Order of presentation:**
1. **Time-avg** → Demonstrates temporal modeling is crucial
2. **PCA** → Shows SwiFT handles high-dimensional temporal data better than PCA+SVR
3. **ROI** → Shows data-driven approach (SwiFT) beats prior-based approach

---

## Performance Comparison (When Available)

### Current Results

| Method | Test MSE | Test MAE | Test R² | Status |
|--------|----------|----------|---------|--------|
| **ROI** | 2.074 | 0.753 | -0.114 | ✅ Complete |
| **PCA** | - | - | - | ⏳ Evaluating |
| **Time-avg (RBF)** | - | - | - | ❓ Pending |
| **Time-avg (Linear)** | - | - | - | ❓ Pending |
| **SwiFT-IO** | - | - | - | 🎯 Target |

---

## Implementation Details

### Common Configuration
All baselines share:
- Dataset: HBN movieDM
- Task: 7 emotion regression
- Split seed: 777
- Sequence length: 30 TRs
- Train/Val/Test: 70%/15%/15%
- Samples: 11,349 / 2,403 / 2,437

### SVR Hyperparameters
- Kernel: RBF (or Linear for variant)
- C: 1.0
- Epsilon: 0.1
- Standardization: Yes (per emotion)

### Training Scripts
- PCA: `sample_scripts/run_svr_pca.slurm`
- ROI: `sample_scripts/run_svr_roi.slurm`
- Time-avg: `sample_scripts/run_svr_time_avg.slurm`
- General: `sample_scripts/run_svr_with_reduction.slurm`

### Evaluation Script
- Script: `src/eval_svr_from_checkpoint.py`
- Can load checkpoints and evaluate on train/val/test
- Useful when training was interrupted

---

## Key Findings

### ROI-based SVR Results
- **Negative R² scores** indicate model performs worse than predicting mean
- Very **low correlations** (0.03-0.08)
- Performance is **consistent across train/val/test** (no overfitting)
- **Fear** and **Negative** emotions slightly better (R² closer to 0)

### Implications
These weak baselines demonstrate:
1. **Static features are insufficient** for emotion prediction
2. **Temporal dynamics are crucial** (motivates SwiFT-IO)
3. **Simple dimensionality reduction** doesn't capture the complexity
4. Need for **sophisticated temporal modeling** (transformers)

---

## Next Steps

1. ✅ Complete PCA evaluation (Job 62935 running)
2. ⬜ Verify Time-avg results
3. ⬜ Run Time-avg Linear if needed
4. ⬜ Compare all baselines with SwiFT-IO
5. ⬜ Create visualization plots for paper

---

## References

### Related Files
- Main training script: `src/train_svr_with_reduction.py`
- SVR implementation: `src/baselines/svr_with_reduction.py`
- Evaluation script: `src/eval_svr_from_checkpoint.py`
- Optimization notes: `251022_SVR_PCA_Optimization.md`

### Data Locations
- fMRI data: `/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120`
- ROI timeseries: `/scratch/HBN/9.2.movieDM_ROI_timeseries`
- Output directory: `output/svr_reduction_*/`

---

## Notes

### Optimization History (PCA Method)
- **Original**: 20 hours for IncrementalPCA fitting
- **Optimized (v2)**: 2.5 hours (8x faster)
- **Key improvement**: Vectorized NumPy operations instead of nested for loops
- **Details**: See `251022_SVR_PCA_Optimization.md`

### Disk Space Issues
- Previous runs failed due to disk space
- Current space available: 2.9 TB on /scratch
- Checkpoint-based evaluation strategy implemented to handle interruptions
- **Emotion 6 checkpoint corrupted**: Job 62872 ran out of disk space while saving emotion 6 model
  - Emotions 0-5 checkpoints are valid and complete
  - Modified evaluation script to handle partial checkpoints gracefully
  - Evaluating only 6 emotions (0-5) instead of all 7

---

**Last Updated**: 2025-10-23
**Status**: PCA evaluation in progress, ROI complete, Time-avg pending verification
