# Baseline Model Performance Comparison

**Dataset**: HBN movieDM
**Task**: 7-emotion regression (Anger, Happy, Fear, Sad, Excited, Positive, Negative)
**Evaluation**: Train/Val/Test split (70%/15%/15%) with seed 777
**Samples**: 11,349 train / 2,403 val / 2,437 test sequences
**Evaluation Date**: 2025-10-25

---

## Why This Stepwise Comparison Matters

### Scientific Rationale for Baseline Progression

We compare SwiFT-IO against progressively sophisticated baselines to **isolate the contribution of each architectural component**:

1. **SVR (Time-averaged)** → Shows the **value of temporal modeling**
   - Baseline: Averages 30 TRs into single vector (temporal dynamics lost)
   - If SwiFT-IO >> Time-averaged SVR → Temporal modeling is critical

2. **SVR (ROI/PCA with temporal info)** → Shows **spatial modeling matters beyond temporal**
   - Baseline: Preserves temporal info but uses hand-crafted spatial features (ROI averaging or PCA)
   - If SwiFT-IO >> ROI/PCA SVR → Learned hierarchical spatial representations add value

3. **LSTM** → Shows **transformer advantage over recurrent models**
   - Baseline: Sequential temporal modeling with recurrent architecture
   - If SwiFT-IO >> LSTM → Self-attention > sequential processing for fMRI

4. **SwiFT-IO** → **Our contribution**: 4D spatiotemporal transformer
   - Combines: 4D Swin attention + hierarchical patches + self-supervised pretraining
   - Expected to outperform all baselines by capturing complex spatiotemporal dynamics

**Key Research Question**: *Can learned 4D spatiotemporal representations outperform hand-crafted features and simpler deep models for continuous emotion prediction from fMRI?*

---

## Table 1: Overall Performance Comparison

| Model | Test MSE ↓ | Test MAE ↓ | Test R² ↑ | Mean Correlation ↑ | Status |
|-------|-----------|-----------|----------|-------------------|--------|
| **SVR (ROI-based)** | **2.074** | **0.753** | **-0.114** | **0.050** | ✅ Complete |
| **SVR (PCA-based)** | - | - | - | - | 🔄 Evaluating (Job 63164) |
| **SVR (Time-averaged PCA, RBF)** | - | - | - | - | 🔄 Evaluating (Job 63225) |
| **SVR (Time-averaged, Linear)** | - | - | - | - | ⏸️ Pending |
| **LSTM Encoder-Decoder** | **12.108** | **1.798** | **-0.135** | **-0.160** | ✅ Complete |
| **SwiFT-IO (opr6oq97)** | - | - | - | - | 🔄 Evaluating (Job 63160) |

**Legend**:
- ↓ Lower is better | ↑ Higher is better
- Mean Correlation: Averaged across 7 emotions
- Negative R²: Model performs worse than predicting the mean
- val: Validation set only (test evaluation pending)

---

## Table 2: Per-Emotion Performance Comparison

### 2.1 SVR (ROI-based) - Test Set

| Emotion | MSE ↓ | MAE ↓ | R² ↑ | Correlation ↑ |
|---------|-------|-------|------|---------------|
| Anger | 2.515 | 0.860 | -0.218 | 0.046 |
| Happy | 2.419 | 0.945 | -0.121 | 0.036 |
| Fear | 0.366 | 0.371 | -0.048 | 0.041 |
| Sad | 2.837 | 0.822 | -0.213 | 0.078 |
| Excited | 3.975 | 0.821 | -0.128 | 0.050 |
| Positive | 1.555 | 0.743 | -0.118 | 0.037 |
| Negative | 0.878 | 0.709 | -0.025 | 0.057 |
| **Mean** | **2.074** | **0.753** | **-0.114** | **0.050** |

**Key Observations:**
- Negative R² across all emotions → worse than mean baseline
- Best: Sad (corr=0.078), Negative (corr=0.057)
- Worst: Happy (corr=0.036), Positive (corr=0.037), Fear (corr=0.041)
- Fear has lowest MSE (0.366) but poor correlation

### 2.2 LSTM Baseline - Test Set

| Emotion | MSE ↓ | MAE ↓ | R² ↑ | Correlation ↑ |
|---------|-------|-------|------|---------------|
| Anger | 4.263 | 1.693 | -0.426 | -0.445 |
| Happy | 1.522 | 0.971 | -0.195 | -0.085 |
| Fear | 0.801 | 0.849 | 0.000 | NaN |
| Sad | 1.159 | 0.953 | 0.000 | NaN |
| Excited | 68.754 | 5.471 | -0.642 | -0.301 |
| Positive | 7.937 | 2.150 | -0.436 | 0.594 |
| Negative | 0.317 | 0.501 | -0.305 | 0.028 |
| **Mean** | **12.108** | **1.798** | **-0.135** | **-0.160** |

**Key Observations:**
- ⚠️ Very poor test performance (MSE=12.11, R²=-0.14)
- Excited: catastrophic MSE = 68.75 (extremely high error)
- Fear & Sad: NaN correlations (numerical instability)
- Negative R² across most emotions → worse than mean baseline
- Best emotion: Positive (corr=0.594), but still R²=-0.436
- Model struggles with variability prediction despite epoch 10 being "best" on validation

### 2.3 SwiFT-IO (opr6oq97) - Test Set

| Emotion | MSE ↓ | MAE ↓ | R² ↑ | Correlation ↑ |
|---------|-------|-------|------|---------------|
| Anger | - | - | - | - |
| Happy | - | - | - | - |
| Fear | - | - | - | - |
| Sad | - | - | - | - |
| Excited | - | - | - | - |
| Positive | - | - | - | - |
| Negative | - | - | - | - |
| **Mean** | - | - | - | - |

**Status:** 🎯 Currently evaluating 103 test subjects (Job 63160, 14/103 complete)

---

## Table 3: Model Characteristics

| Model | Feature Dimension | Temporal Modeling | Spatial Processing | Training Time | Interpretability |
|-------|------------------|-------------------|-------------------|---------------|------------------|
| **SVR (ROI)** | 2,850 (95 ROIs × 30 TRs) | Concatenated | ROI averaging | Medium (~2h) | High (ROI-based) |
| **SVR (PCA)** | 3,000 (100 PC × 30 TRs) | Preserved | PCA compression | Slow (~3h) | Medium (PC weights) |
| **SVR (Time-avg, RBF)** | 884,736 (full voxels) | **Averaged out** | Full resolution | Fast (~1h) | Low (RBF kernel) |
| **SVR (Time-avg, Linear)** | 884,736 (full voxels) | **Averaged out** | Full resolution | Fast (~30m) | Medium (linear weights) |
| **LSTM** | Learned embedding | LSTM cells | CNN pooling (16³) | Long (~10h, GPU) | Low (black box) |
| **SwiFT-IO** | Self-attention | **4D Swin Transformer** | **4D patches** | Very long (~20h, multi-GPU) | Medium (attention maps) |

---

## Key Findings

### 1. SVR (ROI-based) Performance
- **First completed baseline**: Provides initial benchmark
- **Still negative R²** (-0.114): Performs worse than mean baseline
- **Low correlations** (0.026-0.082): Weak predictive power
- **Best emotions**: Sad (r=0.082), Excited (r=0.062), Negative (r=0.058)
- **Worst emotions**: Fear (r=0.026), Happy (r=0.029)
- **Conclusion**: ROI averaging helps but still insufficient; temporal dynamics not well captured

### 2. Emotion-Specific Patterns
- **Fear** has lowest MSE (0.366) - easier to predict
- **Sad** has highest MSE (2.830) - harder to predict
- **Negative and Sad** show highest correlations (0.058, 0.082)
- **Fear and Happy** show lowest correlations (0.026, 0.029)

### 3. Overall Implications
- **Negative R²** indicates model performs worse than predicting the mean
- **Low correlations** (0.026-0.082): Weak predictive power overall
- **ROI aggregation alone is insufficient**: Need for better temporal modeling
- **Motivation for SwiFT-IO**: Current baseline fails to capture spatiotemporal dynamics

---

## Implications for SwiFT-IO

### Expected Improvements Over Baselines

SwiFT-IO's 4D spatiotemporal transformer architecture should outperform baselines through:

1. **vs. SVR (Time-averaged)** → **Temporal modeling is essential**
   - SVR limitation: Averages out all temporal dynamics (30 TRs → 1 vector)
   - SwiFT-IO advantage: Preserves and models temporal evolution with 4D attention
   - **Expected gain**: Large improvement demonstrates emotions change over time

2. **vs. SVR (ROI/PCA)** → **Learned spatial features > hand-crafted**
   - SVR limitation: Fixed spatial features (anatomical ROIs or PCA)
   - SwiFT-IO advantage: Hierarchical learned representations via 4D Swin patches
   - **Expected gain**: Moderate improvement shows learned features capture task-relevant patterns

3. **vs. LSTM** → **Self-attention > sequential processing**
   - LSTM limitation: Sequential bottleneck, limited long-range dependencies
   - SwiFT-IO advantage: Parallel attention across space AND time
   - **Expected gain**: Moderate improvement demonstrates transformer benefits for fMRI

4. **Overall Performance Prediction**
   - Based on current baselines (MSE ~2.0, R² < 0):
   - **Target**: MSE < 1.0, R² > 0.3, Correlation > 0.6
   - **Minimum viable**: MSE < 1.5, R² > 0.1, Correlation > 0.4

### Comparison Strategy (Progressive Complexity)

| Comparison | Tests Hypothesis | Expected Result |
|------------|------------------|-----------------|
| SwiFT-IO vs. **SVR Time-avg** | Temporal modeling matters | **Large gap** (dynamic >> static) |
| SwiFT-IO vs. **SVR ROI/PCA** | Learned features >> hand-crafted | **Moderate gap** (hierarchical learning helps) |
| SwiFT-IO vs. **LSTM** | Transformers >> RNNs for fMRI | **Moderate gap** (attention > recurrence) |
| SwiFT-IO vs. **Baseline ensemble** | Overall contribution | **Consistent improvement across all metrics** |

### Statistical Testing Plan
- **Per-emotion paired t-test** (7 comparisons)
- **Bootstrap confidence intervals** (1000 iterations)
- **Effect size**: Cohen's d for each emotion
- **Multiple comparison correction**: Bonferroni or FDR
- **Significance threshold**: p < 0.05 (corrected)

---

## Next Steps

### Immediate (This Week)
1. ✅ Complete SVR PCA training (Job 63055 running)
2. ⬜ Evaluate SVR PCA on test set
3. ⬜ Train SVR Time-averaged (RBF and Linear)
4. ⬜ Update this table with all baseline results

### Short-term (Next Week)
1. ⬜ Train LSTM baseline
2. ⬜ Complete SwiFT-IO training/evaluation
3. ⬜ Statistical significance testing
4. ⬜ Create visualization plots

### Long-term (Paper Writing)
1. ⬜ Ablation studies (SwiFT-IO components)
2. ⬜ Interpretability analysis (attention maps)
3. ⬜ Cross-dataset validation
4. ⬜ Methods section draft

---

## References

### Data Location
- **fMRI data**: `/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120`
- **ROI timeseries**: `/scratch/HBN/9.2.movieDM_ROI_timeseries`
- **Output directory**: `output/*/`

### Code Location
- **Baseline implementations**: `src/baselines/`
- **Training scripts**: `src/train_*_baseline.py`
- **Evaluation scripts**: `src/eval_*_from_checkpoint.py`
- **SLURM scripts**: `sample_scripts/run_*.slurm`

### Documentation
- Baseline setup: `BASELINE_SETUP.md`
- Methods comparison: `251023_SVR_Baseline_Methods_Comparison.md`
- ROI results: `251020_SVR_ROI_BASELINE_RESULTS.md`
- PCA optimization: `251022_SVR_PCA_Optimization.md`

---

**Last Updated**: 2025-10-25 16:00 KST
**Status**: 1/6 baselines complete (SVR-ROI), 1 partial (LSTM validation only)
**Next Update**: After SwiFT-IO test evaluation completes (Job 63160)

---

## Notes on Excluded Results

### GLM Baseline (October 10)
- **Not included**: Different experimental setup
- Used only 34 emotion-related ROIs (not full brain)
- Sequence length: 20 (vs. current 30)
- Cannot be fairly compared with current baselines
- May re-run with consistent settings if needed
