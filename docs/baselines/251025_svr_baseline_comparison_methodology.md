# SVR Baseline Comparison - Methodology and Interpretation Guide

**Date**: 2025-10-25
**Task**: 7-class emotion regression from fMRI (HBN movieDM)
**Dataset split seed**: 777
**Purpose**: Compare different spatial feature extraction methods for emotion prediction

---

## Overview

We implement three SVR-based baselines to understand the importance of:
1. **Spatial representation** (anatomical vs. data-driven)
2. **Temporal information** (sequence vs. time-averaged)
3. **Feature engineering** (domain knowledge vs. learned features)

---

## 1. SVR with ROI Reduction (AAL Atlas)

### Methodology

**Approach**: Knowledge-driven spatial feature extraction

**Feature Extraction Process**:
```
Input: 750 TRs → 25 non-overlapping sequences (30 TRs each)

For each sequence:
  For each TR:
    96×96×96 voxels → AAL Atlas (116 regions) → Average per region

  Result: 30 TRs × 116 regions = 3,480 features
```

**Feature Dimensions**:
- Per TR: 116 features (anatomical regions)
- Total: 3,480 features (30 TRs × 116)
- Compression ratio: 884,736 → 116 (0.013%)

**Key Characteristics**:
- ✅ **Interpretable**: Features correspond to known brain regions (e.g., amygdala, prefrontal cortex)
- ✅ **Neuroscience-grounded**: Uses anatomical knowledge
- ✅ **Robust to noise**: Averaging within regions reduces voxel-level noise
- ❌ **Atlas-dependent**: AAL may not be optimal for emotion prediction
- ❌ **Loses spatial patterns**: Only region averages, no within-region structure

**Training Data**:
- Subjects: 473 (train)
- Sequences: 11,349 (473 × ~24)

---

## 2. SVR with PCA Reduction (Data-Driven)

### Methodology

**Approach**: Data-driven spatial feature extraction

**Feature Extraction Process**:
```
PCA Training Phase:
  - Collect ALL timepoints from training set
  - 473 subjects × 24 sequences × 30 TRs = ~340,560 timepoints
  - Each timepoint: 884,736 voxels
  - Fit IncrementalPCA on this massive dataset
  - Learn 100 principal components (91.7% variance)

Inference Phase:
  For each sequence (30 TRs):
    For each TR:
      96×96×96 voxels → PCA transform → 100 features

    Concatenate: 30 TRs × 100 = 3,000 features
```

**Feature Dimensions**:
- Per TR: 100 features (principal components)
- Total: 3,000 features (30 TRs × 100)
- Compression ratio: 884,736 → 100 (0.011%)

**Key Characteristics**:
- ✅ **Data-driven**: Learns optimal spatial patterns from data
- ✅ **Atlas-independent**: No prior anatomical assumptions
- ✅ **Preserves variance**: 91.7% of spatial variance retained
- ✅ **Temporal structure preserved**: Each TR processed separately
- ❌ **Not interpretable**: Components don't correspond to anatomical regions
- ❌ **Data-dependent**: May overfit to training set patterns

**Training Data**:
- Subjects: 473 (train)
- Sequences: 11,344 (473 × ~24)
- Timepoints for PCA: ~340,560

---

## 3. SVR with Time-Averaging + PCA

### Methodology

**Approach**: Simplified baseline (removes temporal information)

**Feature Extraction Process**:
```
Time-averaging Phase:
  750 TRs → Average across time → 1 volume per subject
  96×96×96 voxels

PCA Phase:
  884,736 voxels → PCA → 100 features

Result: 100 features (no temporal dimension)
```

**Feature Dimensions**:
- Per subject: 100 features (no temporal dimension)
- Total: 100 features
- Compression ratio: 884,736 → 100 (0.011%)

**Key Characteristics**:
- ✅ **Simple**: Minimal preprocessing
- ✅ **Fast**: No sequential processing
- ✅ **Small feature space**: Only 100 features
- ❌ **Loses temporal information**: All dynamics averaged out
- ❌ **Fewer samples**: 473 subjects vs. 11,344 sequences

**Training Data**:
- Subjects: 473 (train)
- Samples: 473 (one per subject)

---

## Comparison Table

| Aspect | ROI SVR | PCA SVR | Time-Avg PCA SVR |
|--------|---------|---------|------------------|
| **Feature Dimension** | 3,480 | 3,000 | 100 |
| **Dimension Difference** | Baseline | -14% | -97% |
| **Spatial Representation** | 116 anatomical regions | 100 data-driven components | 100 components |
| **Temporal Information** | ✅ Preserved (30 TRs) | ✅ Preserved (30 TRs) | ❌ Lost (averaged) |
| **Interpretability** | ✅✅✅ High | ❌ Low | ❌ Low |
| **Training Samples** | 11,349 sequences | 11,344 sequences | 473 subjects |
| **PCA Training Data** | N/A | 340,560 timepoints | 473 volumes |
| **Prior Knowledge** | Anatomical atlas | None | None |
| **Compression** | 884K → 116 | 884K → 100 | 884K → 100 |

---

## Fairness of Comparison

### Feature Dimension Difference: 3,480 vs 3,000 (16% difference)

**Is this fair?**

✅ **Yes, sufficiently fair**:
1. 16% difference is relatively small (not orders of magnitude)
2. Both represent ~100-level compression per timepoint
3. Each method uses its "natural" dimensionality:
   - AAL atlas: 116 regions (standard in neuroscience)
   - PCA: 100 components (captures 91.7% variance, standard practice)
4. The difference is comparable feature dimensions, not drastically different

**Justification for keeping current dimensions**:
- Respects the standard configuration of each method
- 16% difference is unlikely to dominate performance differences
- Allows comparison while maintaining methodological integrity
- Can be described as "comparable feature dimensions (~3,000)" in publications

---

## Interpretation Scenarios

### Scenario 1: ROI SVR >> PCA SVR

**Interpretation**:
```
→ Anatomical knowledge is highly valuable for emotion prediction
→ AAL atlas regions capture emotion-relevant patterns
→ Data-driven PCA may learn noise or task-irrelevant patterns
→ Domain knowledge (neuroscience priors) is crucial

Implications for SwiFT:
→ Should incorporate anatomical constraints
→ Attention mechanisms should align with brain regions
→ Interpretability matters for neuroscience applications
```

**Example Result**: ROI R² = 0.15, PCA R² = -0.05

---

### Scenario 2: PCA SVR >> ROI SVR

**Interpretation**:
```
→ Emotion-related patterns don't align with anatomical boundaries
→ Brain activity for emotions is distributed across multiple regions
→ Data-driven approaches can discover novel patterns
→ AAL atlas is suboptimal for emotion prediction

Implications for SwiFT:
→ End-to-end learning is valuable (don't constrain to anatomy)
→ Learned representations can outperform hand-crafted features
→ Deep learning's flexibility is necessary
```

**Example Result**: ROI R² = -0.11, PCA R² = 0.10

---

### Scenario 3: PCA SVR ≈ ROI SVR (Both Poor)

**Interpretation**:
```
→ Spatial features alone are insufficient
→ Temporal dynamics are crucial for emotion prediction
→ Static patterns (even with 30 TRs) miss important information
→ Need models that capture temporal dependencies

Implications for SwiFT:
→ Temporal modeling (LSTM/Transformer) is essential
→ Recurrent or attention-based architectures needed
→ Sequence-to-sequence learning is justified
```

**Example Result**: ROI R² = -0.11, PCA R² = -0.09

---

### Scenario 4: Time-Avg PCA << Both Sequential Methods

**Interpretation**:
```
→ Temporal information is critical
→ Emotion prediction requires dynamics, not just static patterns
→ Time-averaging destroys important signal

Implications:
→ Confirms the need for sequential models
→ Justifies the complexity of SwiFT architecture
```

**Example Result**: Time-Avg R² = -0.25, ROI/PCA R² = -0.11

---

### Scenario 5: Time-Avg PCA ≈ Sequential Methods

**Interpretation (Surprising)**:
```
→ Temporal dynamics may not be as important as expected
→ Task difficulty is high regardless of temporal modeling
→ Spatial patterns dominate emotion prediction
→ OR: fMRI temporal resolution is too coarse

Implications:
→ Question the value of sequential modeling
→ May need different tasks or data
→ Suggests fundamental challenges in fMRI-based emotion prediction
```

**Example Result**: All methods R² ≈ -0.10

---

## Expected Results

Based on current partial results:

### ROI SVR (Complete)
- Test MSE: 2.0739
- Test MAE: 0.7527
- Test R²: **-0.1144**
- Status: ✅ Complete

### PCA SVR (Running - Job 63164)
- Expected completion: ~25 hours
- Prediction: Similar or slightly better than ROI
- Status: ⏳ In progress

### Time-Avg PCA SVR (Running - Job 63162)
- Expected completion: ~3 hours
- Prediction: Worse than sequential methods
- Status: ⏳ In progress

---

## Scientific Value

### 1. Understanding Spatial Representations
**Question**: Are anatomical regions or data-driven components better for emotion?

**Answer from comparison**:
- If ROI > PCA → Neuroscience knowledge is valuable
- If PCA > ROI → Need to discover new spatial patterns

### 2. Understanding Temporal Importance
**Question**: How important are temporal dynamics for emotion prediction?

**Answer from comparison**:
- If Sequential >> Time-Avg → Temporal modeling is essential
- If Sequential ≈ Time-Avg → Spatial patterns dominate

### 3. Baseline for Deep Learning
**Question**: Do we need complex models like SwiFT?

**Answer**:
- If SwiFT >> SVR baselines → Complexity is justified
- If SwiFT ≈ SVR baselines → Task may be too difficult or data insufficient
- If SwiFT < SVR baselines → Overfitting or architectural issues

### 4. Feature Engineering Insights
**Question**: What kind of features matter for emotion?

**Answer from analysis**:
- Per-emotion importance of regions (ROI)
- Per-emotion importance of components (PCA)
- Guides future feature engineering and model design

---

## Next Steps

1. ⏳ **Wait for results** (Job 63164 and 63162 completion)
2. 📊 **Compare all three methods** (train/val/test metrics)
3. 🔍 **Analyze per-emotion performance** (which emotions are easier/harder)
4. 📈 **Compare with LSTM baseline** (temporal modeling baseline)
5. 🧠 **Compare with SwiFT** (ultimate comparison)
6. 📝 **Update baseline_performance_comparison.md** with final results

---

## Technical Notes

### PCA Training Details
- **Method**: IncrementalPCA (memory-efficient)
- **Samples**: ~340,560 timepoints from train set
- **Components**: 100 (91.7% variance explained)
- **Checkpoint**: Saved for reproducibility
- **Training time**: Completed during model training

### SVR Training Details
- **Kernel**: RBF (non-linear)
- **C**: 1.0 (regularization)
- **Epsilon**: 0.1
- **Standardization**: Yes (per-feature)
- **Training**: Per-emotion models (7 separate SVRs)

### Evaluation Details
- **Metrics**: MSE, MAE, R², Pearson correlation
- **Splits**: Train (70%), Val (15%), Test (15%)
- **Seed**: 777 (reproducible)
- **Per-emotion**: Separate metrics for each of 7 emotions

---

*Last updated: 2025-10-25*
*Jobs running: 63164 (PCA SVR eval), 63162 (Time-Avg PCA training)*
