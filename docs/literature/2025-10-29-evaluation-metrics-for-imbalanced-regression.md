# Evaluation Metrics for Sparse & Imbalanced Regression

**Date**: 2025-10-29
**Question**: Loss를 바꿨을 때 evaluation은 어떤 metric을 써야 하는가?
**Context**: Zero-inflated regression에서 magnitude prediction

---

## ❓ 핵심 질문

> **"Tweedie loss나 Zero-inflated model로 학습했는데,
> evaluation은 여전히 MSE를 써야 하나요?"**

**답**: ❌ **No!** Global MSE만 쓰면 개선이 제대로 보이지 않습니다.

---

## 🚨 왜 Global MSE/MAE가 문제인가?

### 문제 1: Majority (0) Dominance

```python
# 예시: Sad emotion (90% zero)
targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]  # 90% zero

# Model A: 항상 0 예측
preds_A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mse_A = 10.0  # 10^2 / 10

# Model B: Non-zero 잘 예측 (but 0을 조금 틀림)
preds_B = [0.5, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2, 0.3, 0.4, 9.5]
mse_B = 0.25 + ... + (0.5)^2 = ~0.52

# MSE: Model A (10.0) > Model B (0.52)
# But B가 실제로는 훨씬 좋은 모델!
```

**문제**: Global MSE는 0을 많이 틀려도 penalty가 작고, 큰 값 하나 틀리면 엄청 큰 penalty.

---

### 문제 2: Magnitude Prediction 무시

```python
# Emotion의 실제 magnitude 예측 비교
targets =  [0, 0, 5, 10, 15]

# Model A: 모든 non-zero를 8로 예측
preds_A = [0, 0, 8, 8, 8]
# Non-zero error: |8-5| + |8-10| + |8-15| = 3 + 2 + 7 = 12

# Model B: Non-zero magnitude를 잘 구분
preds_B = [0, 0, 4, 11, 14]
# Non-zero error: |4-5| + |11-10| + |14-15| = 1 + 1 + 1 = 3

# Global MSE로는 차이가 잘 안 보임
# But magnitude prediction 능력은 B >> A
```

---

## ✅ 올바른 Evaluation Strategy

### 전략 1: **Stratified Evaluation** ⭐⭐⭐⭐⭐

**핵심**: Zero samples와 Non-zero samples를 **분리해서** 평가

```python
def stratified_evaluation(y_true, y_pred):
    """Zero와 non-zero를 분리 평가"""
    # 1. Zero samples
    mask_zero = (y_true == 0)
    y_true_zero = y_true[mask_zero]
    y_pred_zero = y_pred[mask_zero]

    # 2. Non-zero samples
    mask_nonzero = ~mask_zero
    y_true_nonzero = y_true[mask_nonzero]
    y_pred_nonzero = y_pred[mask_nonzero]

    # Metrics
    results = {
        # Overall
        'overall_mse': mean_squared_error(y_true, y_pred),
        'overall_mae': mean_absolute_error(y_true, y_pred),

        # Zero samples (얼마나 0을 잘 예측하는가)
        'zero_mse': mean_squared_error(y_true_zero, y_pred_zero),
        'zero_mae': mean_absolute_error(y_true_zero, y_pred_zero),

        # Non-zero samples (magnitude 예측 능력) ← 가장 중요!
        'nonzero_mse': mean_squared_error(y_true_nonzero, y_pred_nonzero),
        'nonzero_mae': mean_absolute_error(y_true_nonzero, y_pred_nonzero),
        'nonzero_rmse': np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero)),

        # Correlation (magnitude 관계)
        'nonzero_pearson': pearsonr(y_true_nonzero, y_pred_nonzero)[0],
        'nonzero_spearman': spearmanr(y_true_nonzero, y_pred_nonzero)[0],
    }

    return results
```

**해석**:
- `overall_mse`: 전체적인 성능 (참고용)
- `nonzero_mse/mae`: **Magnitude prediction 능력** ← 핵심!
- `nonzero_pearson`: Magnitude 순서/관계를 얼마나 잘 잡는가

---

### 전략 2: **Density-Aware Weighted Metrics** ⭐⭐⭐⭐

**출처**: NeurIPS 2024 "Density Ratio Estimation"

**핵심**: Sample의 density에 따라 weight 조절

```python
def density_weighted_mae(y_true, y_pred, density_weights=None):
    """
    Density-aware weighted MAE
    density_weights: rare sample일수록 높은 weight
    """
    if density_weights is None:
        # Estimate density from data
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y_true)
        density = kde(y_true)
        # Inverse density = higher weight for rare samples
        density_weights = 1.0 / (density + 1e-8)
        # Normalize
        density_weights = density_weights / density_weights.sum()

    # Weighted MAE
    errors = np.abs(y_true - y_pred)
    weighted_mae = np.sum(density_weights * errors)

    return weighted_mae
```

**장점**: Rare samples (큰 값)에 자동으로 더 집중

---

### 전략 3: **Binned/Grouped Evaluation** ⭐⭐⭐⭐

**핵심**: Target value의 범위별로 나눠서 평가

```python
def binned_evaluation(y_true, y_pred, bins=None):
    """
    Target 범위별 성능 평가

    Example:
    - Bin 0: [0, 0]        (exact zero)
    - Bin 1: (0, 2]        (small)
    - Bin 2: (2, 5]        (medium)
    - Bin 3: (5, 10]       (large)
    - Bin 4: (10, inf)     (very large)
    """
    if bins is None:
        # Default bins for emotion data
        bins = [0, 0.001, 2, 5, 10, np.inf]

    bin_labels = ['zero', 'small', 'medium', 'large', 'xlarge']

    results = {}
    for i in range(len(bins) - 1):
        mask = (y_true > bins[i]) & (y_true <= bins[i+1])
        if mask.sum() == 0:
            continue

        y_true_bin = y_true[mask]
        y_pred_bin = y_pred[mask]

        results[bin_labels[i]] = {
            'n_samples': mask.sum(),
            'mae': mean_absolute_error(y_true_bin, y_pred_bin),
            'mse': mean_squared_error(y_true_bin, y_pred_bin),
            'mean_true': y_true_bin.mean(),
            'mean_pred': y_pred_bin.mean(),
        }

    return results
```

**출력 예시**:
```
Bin: zero      | n=676 | MAE=0.15 | Mean(pred)=0.10
Bin: small     | n=50  | MAE=0.80 | Mean(pred)=1.20
Bin: medium    | n=15  | MAE=1.50 | Mean(pred)=3.80
Bin: large     | n=7   | MAE=3.20 | Mean(pred)=8.50
Bin: xlarge    | n=2   | MAE=5.00 | Mean(pred)=18.0
```

→ 각 범위에서 얼마나 잘 예측하는지 명확히 보임!

---

## 📊 실제 논문들이 사용하는 Metrics

### 1. PRIME (ICML 2025)

**Evaluation Metrics**:
- **Stratified MSE**: Target value 범위별 MSE
- **Geometric Mean of MSE** (GM-MSE): 각 bin의 MSE를 geometric mean
- **Shot-based evaluation**: Few-shot, Medium-shot, Many-shot 분리

---

### 2. Tweedie Regression (Tubi, 2024)

**Evaluation Metrics**:
- **Mean watch duration** (non-zero only)
- **Revenue** (business metric): +0.4%
- **Total viewing time**: +0.15%
- **Conversion rate** (trade-off): -0.17%
- **Tweedie deviance** (다양한 p 값으로)

**중요 발견**: Overall conversion은 약간 떨어졌지만, revenue와 viewing time은 올라감!
→ 이게 진짜 목표였으므로 성공

---

### 3. Zero-Inflated Models (2024)

**Evaluation Metrics**:
- **Classification metrics** (zero vs non-zero):
  - Precision, Recall, F1
  - AUC-ROC, AUC-PR
- **Regression metrics** (non-zero only):
  - MAE, RMSE on non-zero
  - Pearson correlation

---

## 🎯 당신의 Emotion Decoding에 추천하는 Metrics

### Minimum Required Metrics (필수)

#### Per-Emotion Metrics:

**Non-zero samples (가장 중요!):**
- `nonzero_mae`: Magnitude 예측 오차
- `nonzero_rmse`: RMSE on non-zero
- `nonzero_pearson`: Correlation (순서/관계)
- `nonzero_spearman`: Rank correlation

**Zero samples:**
- `zero_mae`: 0을 얼마나 0 근처로 예측하는가
- `zero_mean_pred`: Zero samples의 평균 예측값

**Overall (참고용):**
- `overall_mae`: 전체 MAE
- `overall_rmse`: 전체 RMSE

#### Summary Metrics (across emotions):
- `avg_nonzero_mae`: 7개 emotion의 평균 non-zero MAE
- `avg_nonzero_pearson`: 평균 correlation

---

### Complete Evaluation Code

```python
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def comprehensive_evaluation(model, test_loader, emotion_names, device='cuda'):
    """Complete evaluation pipeline"""
    model.eval()
    all_preds = []
    all_targets = []

    # Collect predictions
    with torch.no_grad():
        for fmri_seq, emotion_labels in test_loader:
            fmri_seq = fmri_seq.to(device)
            preds = model(fmri_seq)
            all_preds.append(preds.cpu())
            all_targets.append(emotion_labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    results = {}

    # Per-emotion metrics
    for i, emo_name in enumerate(emotion_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        mask_zero = (y_true == 0)
        mask_nonzero = ~mask_zero

        # Overall
        results[f'{emo_name}/overall/mae'] = mean_absolute_error(y_true, y_pred)
        results[f'{emo_name}/overall/rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        # Non-zero (핵심!)
        if mask_nonzero.sum() > 1:
            y_true_nz = y_true[mask_nonzero]
            y_pred_nz = y_pred[mask_nonzero]

            results[f'{emo_name}/nonzero/n'] = int(mask_nonzero.sum())
            results[f'{emo_name}/nonzero/mae'] = mean_absolute_error(y_true_nz, y_pred_nz)
            results[f'{emo_name}/nonzero/rmse'] = np.sqrt(mean_squared_error(y_true_nz, y_pred_nz))
            results[f'{emo_name}/nonzero/pearson'] = pearsonr(y_true_nz, y_pred_nz)[0]
            results[f'{emo_name}/nonzero/spearman'] = spearmanr(y_true_nz, y_pred_nz)[0]

        # Zero samples
        if mask_zero.sum() > 0:
            y_pred_zero = y_pred[mask_zero]
            results[f'{emo_name}/zero/n'] = int(mask_zero.sum())
            results[f'{emo_name}/zero/mean_pred'] = y_pred_zero.mean()
            results[f'{emo_name}/zero/mae'] = np.abs(y_pred_zero).mean()

    # Summary
    nonzero_maes = [results[f'{emo}/nonzero/mae']
                    for emo in emotion_names
                    if f'{emo}/nonzero/mae' in results]
    nonzero_corrs = [results[f'{emo}/nonzero/pearson']
                     for emo in emotion_names
                     if f'{emo}/nonzero/pearson' in results]

    results['summary/avg_nonzero_mae'] = np.mean(nonzero_maes)
    results['summary/avg_nonzero_pearson'] = np.mean(nonzero_corrs)

    # Print
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)

    for emo_name in emotion_names:
        print(f"\n{emo_name}:")

        if f'{emo_name}/nonzero/mae' in results:
            print(f"  Non-zero (n={results[f'{emo_name}/nonzero/n']}): ")
            print(f"    MAE:  {results[f'{emo_name}/nonzero/mae']:.4f}")
            print(f"    RMSE: {results[f'{emo_name}/nonzero/rmse']:.4f}")
            print(f"    r:    {results[f'{emo_name}/nonzero/pearson']:.4f}")

        print(f"  Overall:")
        print(f"    MAE:  {results[f'{emo_name}/overall/mae']:.4f}")
        print(f"    RMSE: {results[f'{emo_name}/overall/rmse']:.4f}")

    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print(f"Avg Non-zero MAE:     {results['summary/avg_nonzero_mae']:.4f}")
    print(f"Avg Non-zero Pearson: {results['summary/avg_nonzero_pearson']:.4f}")
    print("="*70)

    return results, all_preds, all_targets
```

---

## 🎯 최종 추천 Metric Set

### Tier 1: 필수 (Must Report)

1. **Non-zero MAE** (per emotion)
   - Magnitude prediction 능력의 핵심 지표

2. **Non-zero Pearson Correlation** (per emotion)
   - Magnitude 순서/관계를 얼마나 잘 잡는가

3. **Overall MAE** (per emotion)
   - 참고용, baseline 비교

### Tier 2: 강력 추천 (Highly Recommended)

4. **Non-zero RMSE** (per emotion)
   - Outlier sensitivity 확인

5. **Zero MAE** (per emotion)
   - 0 근처를 얼마나 잘 예측하는가

6. **Averaged metrics** (across emotions)
   - 전체 모델 성능 요약

### Tier 3: Advanced (선택)

7. **Binned MAE** (per emotion, per bin)
   - 범위별 성능 분석

8. **Density-weighted MAE**
   - Rare sample에 더 집중

9. **Detection metrics** (Precision/Recall/F1)
   - 0 vs non-zero classification 능력

---

## 📝 Paper Writing Tip

### Results Section에 이렇게 쓰세요:

```markdown
## Results

### Evaluation Metrics

Following prior work on imbalanced regression [PRIME, 2025],
we evaluate model performance using **stratified metrics** that
separately assess zero and non-zero predictions.

For non-zero samples, we report:
- **MAE**: Mean Absolute Error on non-zero samples
- **Pearson r**: Correlation between predicted and true magnitudes

We also report overall MAE for comparison with baseline methods.

### Results

Table 1 shows our model significantly outperforms baselines on
**non-zero MAE** (-30% error) and **Pearson correlation** (+0.25),
indicating better magnitude prediction, while maintaining comparable
overall MAE.

| Model | Overall MAE ↓ | Non-zero MAE ↓ | Non-zero r ↑ |
|-------|---------------|----------------|--------------|
| Baseline (MSE) | 0.85 | 2.45 | 0.32 |
| Ours (Tweedie) | 0.82 | **1.71** | **0.57** |

Despite similar overall MAE, our model achieves substantially better
magnitude prediction on non-zero samples, which is critical for emotion
decoding applications.
```

---

## 🚨 Special Case: Sequence Input (fMRI Data)

### 문제 상황

**Your data structure**:
- Total TRs per subject: 750
- Sequence length: 20-30 TRs
- Number of sequences: ~30 per subject
- Target labels: Same emotion timeseries for all subjects

**우려사항**:
```
전체 750 TRs로 보면:
- Sad: 74 non-zero TRs (9.9%)
- 충분히 많음

But sequence 단위로 보면:
- 일부 sequences가 all-zero일 수 있음
- 특히 Sad (90.1% zero)
- Non-zero MAE를 계산할 수 없는 sequences 존재?
```

---

### 📊 분석 결과

```python
# Sequence-level sparsity analysis
Total TRs: 750
Sequence length: 25
Number of sequences: 30

Positive (38.7% zero):
  → Expected all-zero sequences: ~0 / 30
  → 문제 없음 ✅

Sad (90.1% zero):
  → Expected all-zero sequences: ~2-3 / 30
  → 일부 sequences가 all-zero 가능 ⚠️

추가 문제:
- Non-zero TRs는 clustering될 가능성 높음
  (영화에서 슬픈 장면은 특정 구간에 몰림)
- 실제로는 더 많은 all-zero sequences 존재 가능
```

---

### ✅ 해결책: Sample-level (TR-level) Evaluation

**핵심 아이디어**:
- **학습**: Sequence input (20-30 TRs)
  - Temporal context 활용
  - Batch processing 효율성
- **평가**: TR-level (모든 TRs flatten)
  - 실제 task 목표와 일치
  - Non-zero samples 충분히 확보

**왜 이게 정답인가**:
1. ✅ Non-zero samples 충분: Sad도 74 TRs 확보
2. ✅ All-zero sequence 문제 완전 해결
3. ✅ 평가가 실제 task와 일치 (TR마다 emotion 예측)
4. ✅ 논문들도 이렇게 함 (PRIME, Tweedie 등)

---

### 💻 Complete Implementation

```python
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_with_sequence_input(
    model,
    test_loader,
    emotion_names,
    device='cuda'
):
    """
    Sequence input → TR-level evaluation

    Args:
        model: Your fMRI emotion decoding model
        test_loader: DataLoader that returns:
            - fmri_seq: (batch, seq_len, C, H, W, D) or (batch, seq_len, features)
            - emotion_labels: (batch, seq_len, 7)
        emotion_names: ['Positive', 'Negative', 'Anger', 'Happy', 'Fear', 'Sad', 'Excited']
        device: 'cuda' or 'cpu'

    Returns:
        results: Dict of metrics
        all_preds: (n_total_TRs, 7) - Flattened predictions
        all_targets: (n_total_TRs, 7) - Flattened targets
    """
    model.eval()

    # ================================================================
    # Step 1: Collect predictions (flatten sequences to TR-level)
    # ================================================================
    all_preds = []
    all_targets = []

    print("Collecting predictions from sequences...")

    with torch.no_grad():
        for batch_idx, (fmri_seq, emotion_labels) in enumerate(test_loader):
            # Input shapes:
            # fmri_seq: (batch, seq_len, ...)
            # emotion_labels: (batch, seq_len, 7)

            fmri_seq = fmri_seq.to(device)
            preds = model(fmri_seq)  # (batch, seq_len, 7)

            # Flatten sequence dimension
            batch_size, seq_len, n_emotions = preds.shape
            preds_flat = preds.view(-1, n_emotions)  # (batch*seq_len, 7)
            labels_flat = emotion_labels.view(-1, n_emotions)  # (batch*seq_len, 7)

            all_preds.append(preds_flat.cpu())
            all_targets.append(labels_flat.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Concatenate all batches → All TRs flattened
    all_preds = torch.cat(all_preds).numpy()      # (n_total_TRs, 7)
    all_targets = torch.cat(all_targets).numpy()  # (n_total_TRs, 7)

    print(f"\n✅ Total TRs collected: {all_preds.shape[0]}")
    print(f"   Shape: {all_preds.shape}")

    # ================================================================
    # Step 2: Compute metrics (TR-level)
    # ================================================================
    results = {}

    print("\n" + "="*70)
    print("📊 Computing TR-level metrics...")
    print("="*70)

    for i, emo_name in enumerate(emotion_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        # Masks
        mask_zero = (y_true == 0)
        mask_nonzero = ~mask_zero

        n_total = len(y_true)
        n_zero = mask_zero.sum()
        n_nonzero = mask_nonzero.sum()

        # Overall metrics
        results[f'{emo_name}/overall/mae'] = mean_absolute_error(y_true, y_pred)
        results[f'{emo_name}/overall/rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        # Non-zero metrics (핵심!)
        if mask_nonzero.sum() > 1:
            y_true_nz = y_true[mask_nonzero]
            y_pred_nz = y_pred[mask_nonzero]

            results[f'{emo_name}/nonzero/n'] = int(mask_nonzero.sum())
            results[f'{emo_name}/nonzero/mae'] = mean_absolute_error(y_true_nz, y_pred_nz)
            results[f'{emo_name}/nonzero/rmse'] = np.sqrt(mean_squared_error(y_true_nz, y_pred_nz))
            results[f'{emo_name}/nonzero/pearson'] = pearsonr(y_true_nz, y_pred_nz)[0]
            results[f'{emo_name}/nonzero/spearman'] = spearmanr(y_true_nz, y_pred_nz)[0]

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true_nz - y_pred_nz) / (y_true_nz + 1e-8))) * 100
            results[f'{emo_name}/nonzero/mape'] = mape

            print(f"  {emo_name}: {n_nonzero} non-zero TRs found ✅")
        else:
            print(f"  {emo_name}: WARNING - No non-zero TRs! ⚠️")

        # Zero samples
        if mask_zero.sum() > 0:
            y_pred_zero = y_pred[mask_zero]
            results[f'{emo_name}/zero/n'] = int(mask_zero.sum())
            results[f'{emo_name}/zero/mean_pred'] = y_pred_zero.mean()
            results[f'{emo_name}/zero/mae'] = np.abs(y_pred_zero).mean()

        # Sparsity info
        results[f'{emo_name}/info/total'] = n_total
        results[f'{emo_name}/info/zero_pct'] = (n_zero / n_total) * 100
        results[f'{emo_name}/info/nonzero_pct'] = (n_nonzero / n_total) * 100

    # ================================================================
    # Step 3: Summary metrics (across emotions)
    # ================================================================
    nonzero_maes = [results[f'{emo}/nonzero/mae']
                    for emo in emotion_names
                    if f'{emo}/nonzero/mae' in results]
    nonzero_corrs = [results[f'{emo}/nonzero/pearson']
                     for emo in emotion_names
                     if f'{emo}/nonzero/pearson' in results]

    if len(nonzero_maes) > 0:
        results['summary/avg_nonzero_mae'] = np.mean(nonzero_maes)
        results['summary/std_nonzero_mae'] = np.std(nonzero_maes)

    if len(nonzero_corrs) > 0:
        results['summary/avg_nonzero_pearson'] = np.mean(nonzero_corrs)
        results['summary/std_nonzero_pearson'] = np.std(nonzero_corrs)

    # ================================================================
    # Step 4: Print results
    # ================================================================
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS (TR-level)")
    print("="*70)

    for emo_name in emotion_names:
        print(f"\n{emo_name}:")
        print(f"  Sparsity: {results[f'{emo_name}/info/zero_pct']:.1f}% zero, "
              f"{results[f'{emo_name}/info/nonzero_pct']:.1f}% non-zero")

        if f'{emo_name}/nonzero/mae' in results:
            print(f"  Non-zero (n={results[f'{emo_name}/nonzero/n']}): ")
            print(f"    MAE:     {results[f'{emo_name}/nonzero/mae']:.4f}")
            print(f"    RMSE:    {results[f'{emo_name}/nonzero/rmse']:.4f}")
            print(f"    Pearson: {results[f'{emo_name}/nonzero/pearson']:.4f}")
            print(f"    MAPE:    {results[f'{emo_name}/nonzero/mape']:.2f}%")

        print(f"  Overall:")
        print(f"    MAE:     {results[f'{emo_name}/overall/mae']:.4f}")
        print(f"    RMSE:    {results[f'{emo_name}/overall/rmse']:.4f}")

        if f'{emo_name}/zero/mean_pred' in results:
            print(f"  Zero samples (n={results[f'{emo_name}/zero/n']}):")
            print(f"    Mean pred: {results[f'{emo_name}/zero/mean_pred']:.4f}")
            print(f"    MAE:       {results[f'{emo_name}/zero/mae']:.4f}")

    print("\n" + "="*70)
    print("📋 SUMMARY (Across Emotions)")
    print("="*70)
    if 'summary/avg_nonzero_mae' in results:
        print(f"Avg Non-zero MAE:     {results['summary/avg_nonzero_mae']:.4f} "
              f"± {results['summary/std_nonzero_mae']:.4f}")
    if 'summary/avg_nonzero_pearson' in results:
        print(f"Avg Non-zero Pearson: {results['summary/avg_nonzero_pearson']:.4f} "
              f"± {results['summary/std_nonzero_pearson']:.4f}")
    print("="*70)

    return results, all_preds, all_targets
```

---

### 📝 Usage Example

```python
# ====================================================================
# In your evaluation script on remote server
# ====================================================================

import torch
from torch.utils.data import DataLoader

# Your model and dataset
model = YourEmotionDecodingModel()
model.load_state_dict(torch.load('best_model.pth'))
model = model.to('cuda')

# Test dataset
# Dataset should return:
#   - fmri_seq: (seq_len, C, H, W, D) or (seq_len, features)
#   - emotion_labels: (seq_len, 7)
test_dataset = YourEmotionDataset(split='test', seq_len=25)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Emotion names
emotion_names = ['Positive', 'Negative', 'Anger', 'Happy', 'Fear', 'Sad', 'Excited']

# Evaluate
results, all_preds, all_targets = evaluate_with_sequence_input(
    model=model,
    test_loader=test_loader,
    emotion_names=emotion_names,
    device='cuda'
)

# Save results
import json
with open('evaluation_results.json', 'w') as f:
    # Convert numpy types to Python types for JSON
    results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in results.items()}
    json.dump(results_json, f, indent=2)

# Save predictions for visualization
np.savez(
    'predictions.npz',
    preds=all_preds,
    targets=all_targets,
    emotion_names=emotion_names
)

print("\n✅ Evaluation complete!")
print("   Results saved to: evaluation_results.json")
print("   Predictions saved to: predictions.npz")
```

---

### 📊 Expected Output

```
Collecting predictions from sequences...
  Processed 10 batches...
  Processed 20 batches...
  ...

✅ Total TRs collected: 750
   Shape: (750, 7)

======================================================================
📊 Computing TR-level metrics...
======================================================================
  Positive: 460 non-zero TRs found ✅
  Negative: 252 non-zero TRs found ✅
  Anger: 267 non-zero TRs found ✅
  Happy: 180 non-zero TRs found ✅
  Fear: 185 non-zero TRs found ✅
  Sad: 74 non-zero TRs found ✅
  Excited: 164 non-zero TRs found ✅

======================================================================
📊 EVALUATION RESULTS (TR-level)
======================================================================

Positive:
  Sparsity: 38.7% zero, 61.3% non-zero
  Non-zero (n=460):
    MAE:     1.2345
    RMSE:    1.5678
    Pearson: 0.6543
    MAPE:    45.67%
  Overall:
    MAE:     0.9876
    RMSE:    1.2345
  ...

Sad:
  Sparsity: 90.1% zero, 9.9% non-zero
  Non-zero (n=74):
    MAE:     4.5678
    RMSE:    6.1234
    Pearson: 0.4321
    MAPE:    67.89%
  Overall:
    MAE:     1.2345
    RMSE:    3.4567
  ...

======================================================================
📋 SUMMARY (Across Emotions)
======================================================================
Avg Non-zero MAE:     2.3456 ± 1.2345
Avg Non-zero Pearson: 0.5432 ± 0.1234
======================================================================
```

---

### 🎯 Key Takeaways

1. ✅ **Sequence로 학습, TR로 평가**
   - Input: Sequences (temporal context)
   - Evaluation: Flatten to TRs (실제 task)

2. ✅ **Non-zero samples 충분**
   - Sad: 74 TRs (충분!)
   - All-zero sequence 문제 해결

3. ✅ **구현 간단**
   - `preds.view(-1, 7)` - Flatten
   - 나머지는 동일

4. ✅ **논문에서도 이 방식**
   - Standard practice in imbalanced regression
   - Sample-level metrics

---

## 🔗 References

1. **PRIME (ICML 2025)**: Geometric Mean MSE, Shot-based evaluation
2. **Tweedie (Tubi, 2024)**: Business metrics (revenue, viewing time)
3. **Density-Aware (NeurIPS 2024)**: Weighted MAE with density ratio
4. **Zero-Inflated**: Two-stage evaluation (classification + regression)

---

**Generated**: 2025-10-29
**Updated**: 2025-10-29 (Added sequence input section)
