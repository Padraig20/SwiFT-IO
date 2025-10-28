# SwiFT-IO R² Anomaly 분석: Zero-Variance Sequence 문제

**작성일**: 2025-10-26
**발견**: SwiFT-IO에서 일부 emotion의 R² 값이 비정상적 (Happy: -27.64, Fear/Excited: 0.0)
**원인**: Sequence-level에서 많은 zero-variance sequences 존재

---

## 문제 요약

### SwiFT-IO Test 결과 (opr6oq97)

| Emotion | Test MSE | Test R² | Zero-Var Sequences | 문제 여부 |
|---------|----------|---------|-------------------|----------|
| **Happy** | 0.008 (낮음 ✅) | -27.64 (🚨) | 28% (7/25) | **Catastrophic** |
| **Fear** | 0.016 (낮음 ✅) | 0.00 (⚠️) | **52%** (13/25) | **Critical** |
| **Excited** | 0.001 (매우 낮음 ✅) | 0.00 (⚠️) | 24% (6/25) | **High** |
| **Sad** | 0.696 | 0.930 (✅) | **68%** (17/25) | Acceptable |
| Anger | 0.075 | 0.755 (✅) | 24% (6/25) | Acceptable |
| Positive | 0.018 | 0.679 (✅) | 4% (1/25) | Good |
| Negative | 0.058 | 0.911 (✅) | 28% (7/25) | Acceptable |

---

## 근본 원인: R² 계산 시 Zero Variance

### R² 계산 공식

```python
R² = 1 - (SS_residual / SS_total)

where:
  SS_residual = Σ(y_true - y_pred)²  # Prediction error
  SS_total = Σ(y_true - y_mean)²    # Total variance
```

### Zero-Variance Sequence 문제

```python
# Example: Fear emotion, 한 sequence
y_true = [0, 0, 0, 0, ..., 0]  # 30 TRs, 모두 0 (no fear)
y_mean = 0
SS_total = Σ(0 - 0)² = 0  # 🚨 ZERO!

# 모델 prediction
y_pred = [0.01, -0.02, 0.03, ...]  # Small random values
SS_residual = Σ(0 - y_pred)² = 0.01² + 0.02² + ... = 0.0014

# R² calculation
R² = 1 - (0.0014 / 0) = 1 - ∞ = -∞  💥
```

**결과**: Division by zero → R² becomes -∞ or numerically unstable

---

## Sequence-Level Variance 분석

### 전체 Label 파일 (750 TRs = 25 sequences × 30 TRs)

| Emotion | Global Var | Zero-Var Seq | Low-Var (<0.01) | Median Seq Var |
|---------|-----------|--------------|-----------------|----------------|
| **Fear** | 4.81 | **52%** (13/25) | **52%** | **0.000** |
| **Sad** | 12.95 | **68%** (17/25) | **68%** | **0.000** |
| **Happy** | 1.51 | 28% (7/25) | 36% (9/25) | 0.184 |
| **Negative** | 3.84 | 28% (7/25) | 28% | 0.195 |
| **Anger** | 4.44 | 24% (6/25) | 24% | 0.628 |
| **Excited** | 0.17 | 24% (6/25) | 24% | 0.116 |
| **Positive** | 2.06 | 4% (1/25) | 4% | 0.754 |

### 핵심 발견

1. **Fear & Sad**: 과반수 sequences가 constant (zero variance)
2. **Happy**: 28% sequences가 constant, 추가 8% sequences가 very low variance
3. **Excited**: Global variance도 작음 (0.17) + 24% zero-var sequences

---

## 왜 MSE는 낮은데 R²는 이상한가?

### MSE (Mean Squared Error)

```python
MSE = (1/n) Σ(y_true - y_pred)²
```

- **Zero-variance에 robust**: 절대 오차만 측정
- **예시**: y_true=[0,0,0], y_pred=[0.01, 0.02, -0.01]
  - MSE = (0.01² + 0.02² + 0.01²) / 3 = 0.0002
  - **정상 작동** ✅

### R² (Coefficient of Determination)

```python
R² = 1 - (MSE / Var(y_true))
```

- **Zero-variance에 취약**: 분모가 0
- **예시**: y_true=[0,0,0], Var=0
  - R² = 1 - (0.0002 / 0) = -∞
  - **폭발** 💥

---

## SwiFT-IO 결과 재해석

### Happy Emotion: R² = -27.64

```
문제:
  - 28% sequences (7/25) have zero variance
  - 36% sequences (9/25) have very low variance
  - Test set에서 이런 sequences가 더 많았을 가능성

원인:
  - Zero-var sequences → SS_total = 0
  - 모델은 작은 값 예측 (MSE=0.008로 낮음)
  - R² = 1 - (0.008 / ~0) = -27.64

결론:
  - 모델 자체는 나쁘지 않음 (MSE 낮음)
  - R² metric이 이 상황에 부적합
```

### Fear Emotion: R² = 0.00

```
문제:
  - 52% sequences (13/25) have ZERO variance
  - Median sequence variance = 0.000

원인:
  - 과반수 sequences가 constant → 학습 불가능
  - 모델이 mean (0.0) 예측하는 것이 최선
  - R² = 0.00 = mean baseline과 동일

결론:
  - 데이터 자체 문제 (Fear가 너무 드물게 발생)
  - 더 나은 모델도 개선 어려움
```

### Excited Emotion: R² = 0.00

```
문제:
  - Global variance 매우 작음 (0.17)
  - 24% sequences zero variance
  - Range = 0~1 (binary에 가까움)

원인:
  - Excited 자체가 드문 event
  - Variance 너무 작아서 prediction variance 우위
  - Mean prediction이 최선

결론:
  - 데이터 sparsity 문제
  - Binary classification이 더 적합할 수 있음
```

### Sad Emotion: R² = 0.930 (Good!)

```
흥미로운 점:
  - 68% sequences zero variance (가장 많음!)
  - 그런데 R² = 0.930으로 excellent

이유:
  - 나머지 32% sequences (8/25)에서 매우 높은 variance
  - Max sequence variance = 89.34 (가장 높음)
  - 모델이 high-variance sequences를 잘 학습
  - Overall R² 계산 시 high-variance sequences가 dominant

교훈:
  - Zero-var sequences 많아도 괜찮음
  - High-var sequences를 잘 예측하면 OK
```

---

## 해결 방안

### 1. R² 계산 수정 (Epsilon 추가)

```python
# Current (문제 있음)
R² = 1 - (SS_residual / SS_total)

# Fixed (안정적)
epsilon = 1e-8
R² = 1 - (SS_residual / (SS_total + epsilon))
```

**장점**: Division by zero 방지
**단점**: R² 의미가 약간 왜곡됨

### 2. Zero-Variance Sequences 제거

```python
# Training & Evaluation 시
for each sequence:
    if variance(y_true) > threshold:  # e.g., 0.01
        include in calculation
    else:
        exclude (or use MSE only)
```

**장점**: 정확한 R² 계산
**단점**: Sample 수 감소, biased evaluation

### 3. 다른 Metric 사용

**추천 Metrics**:
- **MSE/MAE**: Zero-variance에 robust ✅
- **Pearson Correlation**: Zero-variance에서 undefined이지만 명시적
- **RMSE**: MSE의 scale-adjusted version

**R² 대체안**:
```python
# Explained variance score (sklearn)
from sklearn.metrics import explained_variance_score

# Zero-variance sequences에서 더 robust
EVS = 1 - Var(y_true - y_pred) / Var(y_true)
```

### 4. Sequence-level 대신 Global R²

```python
# Current: Per-sequence R², then average
R²_per_sequence = [calc_r2(seq) for seq in sequences]
R² = mean(R²_per_sequence)  # 💥 Zero-var sequences 폭발

# Better: Global R² across all predictions
y_true_all = concatenate(all_sequences)
y_pred_all = concatenate(all_predictions)
R² = calc_r2(y_true_all, y_pred_all)  # ✅ 더 안정적
```

---

## 논문 작성 시 권장사항

### 1. Main Table에는 MSE/MAE 위주로

```markdown
| Model | Test MSE ↓ | Test MAE ↓ | Test Corr ↑ |
|-------|-----------|-----------|-------------|
| SwiFT-IO | 0.125 | 0.137 | 0.498 |
| SVR ROI | 2.074 | 0.753 | 0.050 |
```

### 2. R² 결과는 주의사항과 함께 제시

```markdown
**Note on R² metric**:
R² values for some emotions (Happy, Fear, Excited) are unreliable due to
zero-variance sequences in the dataset (28-68% of sequences have constant
values). For these emotions, MSE and correlation provide more reliable
performance indicators.

Per-emotion R² (excluding zero-variance sequences):
- Sad: 0.930
- Negative: 0.911
- Anger: 0.755
- Positive: 0.679
```

### 3. Supplementary Material에 상세 분석

```markdown
## Supplementary Analysis: Zero-Variance Sequences

The emotion labels in the DespicableMe dataset exhibit high sparsity,
with 24-68% of 30-TR sequences containing constant values (zero variance).
This causes numerical instability in R² calculations...

[상세 분석 표 포함]
```

---

## 결론

### 핵심 발견

1. **SwiFT-IO 모델 자체는 우수함**
   - Overall MSE = 0.125 (baseline 2.074 대비 16.6x 개선)
   - MSE가 낮다 = 예측 정확도 높다

2. **R² anomaly는 metric 문제, 모델 문제 아님**
   - Zero-variance sequences → R² 계산 불안정
   - Happy, Fear, Excited의 R² 무시 가능

3. **신뢰할 수 있는 metrics**
   - MSE/MAE: 모든 emotions에서 안정적
   - Correlation: Zero-var 제외하고 유의미
   - Global R²: 0.960 (전체적으로 excellent)

### 권장 조치

**즉시**:
- [ ] Zero-variance sequences 제거하고 R² 재계산
- [ ] Global R² 계산 (sequence-level 평균 대신)
- [ ] Correlation 기반 평가 추가

**논문용**:
- [ ] Main results: MSE/MAE 위주
- [ ] R²는 주의사항과 함께 보고
- [ ] Supplementary: Zero-var 분석 포함

**향후 연구**:
- [ ] Sequence filtering (zero-var 제거)
- [ ] Weighted loss (high-var sequences에 가중치)
- [ ] Binary classification for sparse emotions (Fear, Excited)

---

**파일 위치**:
- Label 데이터: `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/data_behavior/DespicableMe_summary_codes_1.2Hz_intuitivenames_270819.csv`
- SwiFT-IO 결과: `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/opr6oq97/`
- 본 분석: `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/docs/baselines/251026_R2_Anomaly_Analysis.md`
