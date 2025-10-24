# Baseline Model Performance Comparison

**Dataset**: HBN movieDM
**Task**: 7-emotion regression (Anger, Happy, Fear, Sad, Excited, Positive, Negative)
**Evaluation**: Train/Val/Test split (70%/15%/15%) with seed 777
**Samples**: 11,349 train / 2,403 val / 2,437 test sequences

---

## Table 1: Overall Performance Comparison

| Model | Test MSE ↓ | Test MAE ↓ | Test R² ↑ | Mean Correlation ↑ | Status |
|-------|-----------|-----------|----------|-------------------|--------|
| **SVR (ROI-based)** | **2.074** | **0.753** | **-0.114** | **0.050** | ✅ Complete |
| **SVR (PCA-based)** | - | - | - | - | 🔄 Training |
| **SVR (Time-averaged, RBF)** | - | - | - | - | ⏸️ Pending |
| **SVR (Time-averaged, Linear)** | - | - | - | - | ⏸️ Pending |
| **LSTM Encoder-Decoder** | - | - | - | - | ⏸️ Pending |
| **SwiFT-IO** | - | - | - | - | 🎯 Target |

**Notes**:
- ↓ indicates lower is better
- ↑ indicates higher is better
- Mean Correlation is averaged across 7 emotions
- Negative R² indicates model performs worse than predicting the mean

---

## Table 2: Per-Emotion Test Performance

### SVR (ROI-based)

| Emotion | MSE | MAE | R² | Correlation |
|---------|-----|-----|----|-------------|
| Anger (0) | 2.507 | 0.860 | -0.217 | 0.053 |
| Happy (1) | 2.414 | 0.946 | -0.120 | 0.029 |
| Fear (2) | 0.366 | 0.371 | -0.050 | 0.026 |
| Sad (3) | 2.830 | 0.820 | -0.213 | 0.082 |
| Excited (4) | 3.974 | 0.821 | -0.128 | 0.062 |
| Positive (5) | 1.550 | 0.742 | -0.116 | 0.051 |
| Negative (6) | 0.876 | 0.708 | -0.027 | 0.058 |
| **Mean** | **2.074** | **0.753** | **-0.114** | **0.050** |

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

### Expected Improvements
1. **Temporal Modeling**: 4D Swin Transformer vs. concatenated/averaged features
2. **Hierarchical Processing**: Multi-scale attention vs. single-scale features
3. **Learned Representations**: Self-supervised pretraining vs. hand-crafted features
4. **Global Context**: Self-attention vs. local/independent processing

### Comparison Strategy
1. **Primary baseline**: SVR (Time-averaged) - demonstrates value of temporal modeling (static vs dynamic)
2. **With temporal info**: SVR (PCA) and SVR (ROI) - shows even with temporal info, SwiFT outperforms
3. **Deep learning baseline**: LSTM - shows transformer advantage over RNNs

### Statistical Testing
- Paired t-test across 7 emotions
- Bootstrap confidence intervals
- Effect size calculation (Cohen's d)
- Cross-validated performance

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

**Last Updated**: 2025-10-24
**Status**: 1/5 baselines complete (SVR-ROI only)
**Next Update**: After SVR PCA completion

---

## Notes on Excluded Results

### GLM Baseline (October 10)
- **Not included**: Different experimental setup
- Used only 34 emotion-related ROIs (not full brain)
- Sequence length: 20 (vs. current 30)
- Cannot be fairly compared with current baselines
- May re-run with consistent settings if needed
