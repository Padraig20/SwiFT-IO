# Phase 1B: Focal MSE & Tweedie Loss 평가 계획

**Date**: 2025-10-31
**Goal**: Phase 1B 실험 결과를 체계적으로 평가하고 best configuration 선정
**Timeline**: 이번 주말 (Phase 1B 완료 후)

---

## 📊 평가 Metrics

### Tier 1: Primary Metrics (필수)

문헌 조사 결과, zero-inflated regression에서는 **stratified evaluation**이 필수입니다.

#### 1. Non-zero MAE (per emotion) ⭐⭐⭐⭐⭐
- **정의**: Non-zero 샘플에서의 Mean Absolute Error
- **중요도**: 가장 중요! Magnitude prediction 능력의 핵심 지표
- **해석**: 낮을수록 좋음

#### 2. Non-zero Pearson Correlation (per emotion) ⭐⭐⭐⭐⭐
- **정의**: Non-zero 샘플에서 예측값과 실제값의 상관계수
- **중요도**: Magnitude 순서/관계를 얼마나 잘 잡는가
- **해석**: 높을수록 좋음 (0~1)

#### 3. Overall MAE (per emotion) ⭐⭐⭐⭐
- **정의**: 전체 샘플에서의 MAE
- **중요도**: Baseline 비교용 참고 지표
- **해석**: 낮을수록 좋음

### Tier 2: Secondary Metrics (강력 추천)

#### 4. Non-zero RMSE (per emotion)
- **정의**: Root Mean Squared Error on non-zero samples
- **용도**: Outlier sensitivity 확인

#### 5. Zero MAE (per emotion)
- **정의**: Zero 샘플에서의 MAE
- **용도**: 0을 얼마나 0 근처로 예측하는가

#### 6. Summary Metrics (across emotions)
- **Avg Non-zero MAE**: 7개 emotion 평균
- **Avg Non-zero Pearson**: 7개 emotion 평균

---

## 🔬 평가 절차

### Step 1: 데이터 준비

```python
# Phase 1B 완료된 실험들의 checkpoint 수집
experiments = {
    'baseline_mse': 'job_63790',  # Standard MSE baseline
    'focal_g0.5_lr1e-5': 'job_63746',
    'focal_g0.5_lr3e-5': 'job_63745',
    'focal_g0.5_lr5e-5': 'job_63744',
    'focal_g1.0_lr1e-5': 'job_63786',
    'focal_g1.0_lr3e-5': 'job_63748',
    'focal_g1.0_lr5e-5': 'job_63747',
    'tweedie_p1.2_lr1e-5': 'job_63787',
    'tweedie_p1.2_lr3e-5': 'job_63751',
    'tweedie_p1.2_lr5e-5': 'job_63750',
    'tweedie_p1.5_lr1e-5': 'job_63789',
    'tweedie_p1.5_lr3e-5': 'job_63754',
    'tweedie_p1.5_lr5e-5': 'job_63788',
}
```

### Step 2: TR-level Evaluation

**중요**: Sequence로 학습했지만 **TR-level로 평가**해야 합니다.

```python
def evaluate_tr_level(model, test_loader):
    """
    Sequence input → Flatten to TR-level → Evaluate

    Returns:
        results: Dict with stratified metrics
    """
    # Collect predictions (flatten sequences)
    all_preds = []
    all_targets = []

    for fmri_seq, labels in test_loader:
        preds = model(fmri_seq)  # (B, S, 7)

        # Flatten: (B, S, 7) → (B*S, 7)
        preds_flat = preds.view(-1, 7)
        labels_flat = labels.view(-1, 7)

        all_preds.append(preds_flat)
        all_targets.append(labels_flat)

    all_preds = torch.cat(all_preds).numpy()  # (N_TRs, 7)
    all_targets = torch.cat(all_targets).numpy()

    # Compute stratified metrics
    return compute_stratified_metrics(all_preds, all_targets)
```

### Step 3: Stratified Metrics Computation

```python
def compute_stratified_metrics(preds, targets, emotion_names):
    """
    Per-emotion stratified evaluation
    """
    results = {}

    for i, emo in enumerate(emotion_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]

        # Split zero vs non-zero
        mask_zero = (y_true == 0)
        mask_nonzero = ~mask_zero

        # Overall
        results[f'{emo}/overall/mae'] = mae(y_true, y_pred)
        results[f'{emo}/overall/rmse'] = rmse(y_true, y_pred)

        # Non-zero (핵심!)
        if mask_nonzero.sum() > 1:
            y_true_nz = y_true[mask_nonzero]
            y_pred_nz = y_pred[mask_nonzero]

            results[f'{emo}/nonzero/n'] = int(mask_nonzero.sum())
            results[f'{emo}/nonzero/mae'] = mae(y_true_nz, y_pred_nz)
            results[f'{emo}/nonzero/rmse'] = rmse(y_true_nz, y_pred_nz)
            results[f'{emo}/nonzero/pearson'] = pearsonr(y_true_nz, y_pred_nz)[0]

        # Zero
        if mask_zero.sum() > 0:
            results[f'{emo}/zero/mae'] = np.abs(y_pred[mask_zero]).mean()

    return results
```

### Step 4: 비교 분석

각 실험에 대해 평가 수행 후:

1. **Loss Type 비교**: Baseline MSE vs Focal MSE vs Tweedie
2. **Hyperparameter 비교**: γ values, p values, learning rates
3. **Per-emotion 분석**: 어떤 감정에서 개선이 있었나?

---

## 📋 예상 Results Table

### Table 1: Main Results (Test Set, Averaged Across Emotions)

| Model | Loss Type | Params | Overall MAE ↓ | Non-zero MAE ↓ | Non-zero Pearson ↑ | Avg Zero MAE ↓ |
|-------|-----------|--------|---------------|----------------|-------------------|----------------|
| **Baseline** | MSE | - | 0.85 | 2.45 | 0.32 | 0.12 |
| Focal MSE | Focal MSE | γ=0.5, lr=5e-5 | 0.82 | 2.28 | 0.38 | 0.14 |
| Focal MSE | Focal MSE | γ=1.0, lr=5e-5 | 0.79 | **2.15** | **0.42** | 0.16 |
| Focal MSE | Focal MSE | γ=1.0, lr=3e-5 | 0.80 | 2.18 | 0.41 | 0.15 |
| Tweedie | Tweedie | p=1.2, lr=5e-5 | 0.78 | 2.20 | 0.40 | 0.13 |
| Tweedie | Tweedie | p=1.5, lr=3e-5 | 0.77 | 2.12 | 0.43 | 0.14 |

**Bold**: Best performance in each column

**해석 예시**:
- Overall MAE는 비슷하지만 (0.77-0.85 범위)
- **Non-zero MAE에서 큰 개선**: 2.45 → 2.12 (~13% 개선)
- **Correlation 크게 향상**: 0.32 → 0.43 (+0.11, 34% 개선)
- → **Magnitude prediction이 실제로 개선됨!**

---

### Table 2: Per-Emotion Results (Best Model)

**Best Model**: Tweedie (p=1.5, lr=3e-5)

| Emotion | Sparsity | Non-zero N | Non-zero MAE ↓ | Non-zero Pearson ↑ | Overall MAE ↓ | Improvement vs Baseline |
|---------|----------|------------|----------------|-------------------|---------------|------------------------|
| **Positive** | 38.7% zero | 460 | 1.85 | 0.58 | 1.12 | +0.15 (↑) |
| **Negative** | 66.4% zero | 252 | 2.10 | 0.52 | 0.95 | +0.18 (↑) |
| **Anger** | 64.4% zero | 267 | 2.25 | 0.48 | 1.05 | +0.12 (↑) |
| **Happy** | 76.0% zero | 180 | 2.45 | 0.42 | 0.88 | +0.10 (↑) |
| **Fear** | 75.3% zero | 185 | 2.50 | 0.40 | 0.92 | +0.08 (↑) |
| **Sad** | 90.1% zero | 74 | 3.80 | 0.28 | 0.65 | +0.05 (↑) |
| **Excited** | 78.1% zero | 164 | 2.60 | 0.38 | 0.85 | +0.09 (↑) |
| **Average** | - | - | **2.51** | **0.44** | **0.92** | **+0.11** |

**Observations**:
- Positive (가장 balanced): 가장 좋은 성능
- Sad (가장 sparse): 성능 개선이 있지만 여전히 어려움
- 전반적으로 모든 emotion에서 개선

---

### Table 3: Hyperparameter Analysis (Focal MSE)

| Gamma | LR | Valid MSE | Non-zero MAE | Non-zero Pearson | Rank |
|-------|----|-----------|--------------|-----------------|------|
| 0.5 | 1e-5 | 3.208 | 2.35 | 0.36 | 5 |
| 0.5 | 3e-5 | 3.226 | 2.38 | 0.35 | 6 |
| 0.5 | 5e-5 | 3.233 | 2.40 | 0.34 | 7 |
| **1.0** | **5e-5** | **2.747** | **2.15** | **0.42** | **🥇1** |
| 1.0 | 3e-5 | 2.745 | 2.18 | 0.41 | 🥈2 |
| 1.0 | 1e-5 | 2.80 | 2.22 | 0.39 | 🥉3 |

**Finding**:
- γ=1.0 >> γ=0.5 (~15% 성능 향상)
- LR=5e-5와 3e-5는 비슷 (둘 다 안정적)

---

### Table 4: Loss Type Comparison (Best Config)

| Loss Type | Best Config | Non-zero MAE ↓ | Non-zero Pearson ↑ | Overall MAE ↓ | Training Time |
|-----------|-------------|----------------|-------------------|---------------|---------------|
| Baseline MSE | lr=5e-5 | 2.45 | 0.32 | 0.85 | 48h |
| Focal MSE | γ=1.0, lr=5e-5 | 2.15 (-12%) | 0.42 (+31%) | 0.79 | 48h |
| Tweedie | p=1.5, lr=3e-5 | **2.12** (-13%) | **0.43** (+34%) | **0.77** | 48h |

**Winner**: Tweedie (p=1.5)
- Non-zero MAE: 13% 개선
- Correlation: 34% 개선
- Training 시간 동일

---

## 🎯 Decision Criteria

### Best Model 선정 기준 (우선순위)

1. **Non-zero Pearson Correlation** (가장 중요!)
   - Magnitude 관계를 얼마나 잘 잡는가
   - 0.40 이상: Good, 0.50 이상: Excellent

2. **Non-zero MAE**
   - Absolute error on magnitude
   - Baseline 대비 10% 이상 개선 목표

3. **Overall MAE**
   - 참고용 (전체 성능)

4. **Training Stability**
   - Loss가 안정적으로 수렴하는가
   - NaN/Inf 발생 없음

### 성공 판단 기준

Phase 1B가 성공했다고 판단하려면:

✅ **Minimum Success Criteria**:
- Non-zero Pearson > 0.35 (Baseline 0.32 대비 개선)
- Non-zero MAE < 2.30 (Baseline 2.45 대비 6% 이상 개선)

✅ **Good Success**:
- Non-zero Pearson > 0.40 (+25% 개선)
- Non-zero MAE < 2.20 (-10% 개선)

✅ **Excellent Success**:
- Non-zero Pearson > 0.45 (+40% 개선)
- Non-zero MAE < 2.10 (-14% 개선)

---

## 📊 Visualization Plan

### Figure 1: Loss Comparison Heatmap
```
         | Overall MAE | Non-zero MAE | Non-zero Pearson |
---------|-------------|--------------|------------------|
Baseline |    0.85     |     2.45     |      0.32       |
Focal 0.5|    0.82     |     2.28     |      0.38       |
Focal 1.0|    0.79     |     2.15     |      0.42       |
Tweedie  |    0.77     |     2.12     |      0.43       |
```
Color: Green (better) → Red (worse)

### Figure 2: Per-Emotion Improvement
Bar plot showing improvement over baseline for each emotion

### Figure 3: Prediction Scatter Plots
- X-axis: True values (non-zero only)
- Y-axis: Predicted values
- One subplot per emotion
- Compare: Baseline vs Best Model

### Figure 4: Error Distribution
- Histogram of prediction errors
- Separate for zero vs non-zero samples

---

## 💻 Implementation Checklist

### Evaluation Script
- [ ] `evaluate_phase1b.py` 작성
  - [ ] TR-level flattening
  - [ ] Stratified metrics computation
  - [ ] Per-emotion analysis
  - [ ] Summary statistics

### Result Analysis
- [ ] Load all checkpoints
- [ ] Run evaluation on test set
- [ ] Generate comparison tables
- [ ] Create visualizations

### Documentation
- [ ] Results summary document
- [ ] Best model selection rationale
- [ ] Next steps recommendation

---

## 🚀 Timeline

### 금요일 (11/1)
- Phase 1B 재실행 jobs 완료 확인
- Evaluation script 작성

### 토요일 (11/2)
- 모든 실험 평가 실행
- Results table 생성
- Best model 선정

### 일요일 (11/3)
- Baseline 완료 확인 (job 63790)
- Full comparison with baseline
- Full training 시작 (best config, 40-50 epochs)

---

## 📝 Expected Outcome

**이번 주말 후 확보할 것**:

1. ✅ 13개 실험의 체계적 비교표
2. ✅ Best configuration 선정
3. ✅ Baseline 대비 개선 정도 정량화
4. ✅ Full training ready (best config)
5. ✅ 논문 Results section 초안

**다음 주**:
- Best model full training (40-50 epochs)
- Final evaluation & visualization
- 추가 분석 (필요 시)

---

**Created**: 2025-10-31
**Last Updated**: 2025-10-31
