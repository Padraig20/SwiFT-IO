# SwiFT-IO Documentation

Documentation for SwiFT-IO project (v4-v9): 4D Swin Transformer for fMRI-based emotion prediction.

## 📁 Directory Structure

```
docs/
├── baselines/      # Baseline model implementations and comparisons
├── lstm/           # LSTM baseline documentation
├── analysis/       # Performance analysis and results
└── technical/      # Technical fixes and implementations
```

---

## 📊 Baselines (`baselines/`)

Baseline model implementations and performance comparisons:

- **`baseline_performance_table.md`** - 📈 Comprehensive performance comparison table (MSE, MAE, R², Correlation)
- `BASELINE_SETUP.md` - Initial baseline setup guide
- `SVR_BASELINE_READY.md` - SVR baseline ready summary
- `251014_SVR_BASELINE_RATIONALE.md` - Why we chose SVR as baseline
- `251014_SVM_vs_SVR_baseline.md` - SVM vs SVR comparison
- `251020_SVR_ROI_BASELINE_RESULTS.md` - ROI-based SVR results
- `251022_SVR_PCA_Optimization.md` - PCA optimization for SVR (8x speedup)
- `251023_SVR_Baseline_Methods_Comparison.md` - Time-averaged vs ROI vs PCA comparison
- `251025_svr_baseline_comparison_methodology.md` - Detailed methodology

---

## 🧠 LSTM (`lstm/`)

LSTM baseline model documentation:

- **`251014_LSTM_BASELINE.md`** - 📝 LSTM baseline implementation details
- `LSTM_BASELINE_ERROR_ANALYSIS.md` - Error analysis and debugging
- `LSTM_FIX_SUMMARY.md` - Bug fixes (valid_only flag, test evaluation)

---

## 📈 Analysis (`analysis/`)

Performance analysis and experimental results:

- **`251025_emotion_specific_performance_analysis.md`** - 🎯 Emotion-specific MSE analysis and top performers
- `251022_Baseline_Progress_Summary.md` - Overall baseline progress summary
- `251021_TRAINED_MODELS_SUMMARY.md` - Summary of all trained models

---

## 🔧 Technical (`technical/`)

Technical fixes and implementation details:

- `BUS_ERROR_FIX.md` - Bus error debugging and resolution
- `ROI_IMPLEMENTATION.md` - ROI extraction implementation details

---

## 🚀 Quick Reference

### Current Best Model
- **Run ID**: `opr6oq97`
- **Validation MSE**: 0.05
- **Architecture**: Swin4D Transformer + Perceiver Decoder
- **Test subjects**: 103

### Baseline Models Hierarchy

```
1. SVR (Time-averaged)     MSE: 0.1140  ← Shows value of temporal modeling
2. SVR (ROI-based)         MSE: 0.0923  ← Shows spatial modeling matters
3. SVR (PCA-based)         MSE: 0.0758  ← Shows dimensionality reduction helps
4. LSTM                    MSE: [TBD]   ← Shows transformer advantage
5. SwiFT-IO               MSE: 0.0500  ← Our contribution
```

### Key Findings

#### Emotion-Specific Performance (251025)
- **No "generalist" high performers**: Max 2 emotions per subject in top 5
- **Emotion difficulty ranking**:
  - Easiest: Excited (MSE: 0.288, very consistent)
  - Hardest: Fear (3.139), Sad (3.141)
  - Most variable: Sad (std: 0.105)

#### IG Map Baselines
- ✅ **First_10sec baseline**: Sparse, interpretable, auto-masks non-brain
- ❌ **Zeros baseline**: Dense, less interpretable, shows attribution outside brain

---

## 📝 File Naming Convention

- `YYMMDD_topic.md` - Dated documentation (e.g., `251025_emotion_specific_performance_analysis.md`)
- `TOPIC_SUMMARY.md` - Summary documents (e.g., `LSTM_FIX_SUMMARY.md`)
- `topic_name.md` - General documentation (e.g., `baseline_performance_table.md`)

---

## 🔗 Related Directories

- `/src/` - Source code
- `/igmap/` - Integrated Gradients mapping scripts
- `/analysis/4_IGmap/` - IG analysis results
- `/output/moviefmri/` - Model checkpoints
- `/logs/` - Job logs

---

**Last updated**: 2025-10-25
