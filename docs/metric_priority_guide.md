# Metric Priority Guide: Sparse Emotion Prediction

**Purpose**: Classification과 Regression task에서 sparse, zero-inflated emotion 예측 시 어떤 metric을 우선적으로 봐야 하는지 정리
**Date**: 2025-11-18
**Code Reference**: `src/module/pl_classifier.py`

---

## 📌 Overview

이 프로젝트는 **sparse, zero-inflated emotion regression/classification** 문제를 다룹니다:
- 대부분의 timepoint에서 emotion = 0 (no peak)
- 일부 timepoint에서만 emotion > 0 (peak 발생)
- Peak detection + magnitude prediction이 모두 중요

---

## 🎯 CLASSIFICATION: Sparse Peak Detection

**Task**: "이 시점에 emotion peak가 있나/없나?" (Binary Classification)
**Code**: `pl_classifier.py` 431-521행

### 최우선순위 ⭐⭐⭐

#### 1. `{mode}_AUROC_{emotion_name}` (Line 517)
- **정의**: ROC curve 아래 면적
- **의미**: Peak 있음/없음을 얼마나 잘 구분하는가
- **중요도**: 가장 중요한 classification metric
- **해석**:
  - 0.5 = Random guess
  - 0.7-0.8 = Good
  - 0.8-0.9 = Excellent
  - 0.9+ = Outstanding

#### 2. `{mode}_balacc_{emotion_name}` (Line 516)
- **정의**: Balanced Accuracy
- **의미**: Imbalanced data에서도 robust한 정확도
- **중요도**: Peak class와 non-peak class의 균형잡힌 성능
- **해석**: 높을수록 좋음 (0~1)
- **장점**: Class imbalance에 영향을 덜 받음

#### 3. `{mode}_acc_{emotion_name}` (Line 515)
- **정의**: Overall Accuracy
- **의미**: 전체 예측 정확도
- **주의**: Imbalanced data에서는 misleading 가능
  - 예: 90% zero인 경우, 항상 0 예측해도 acc=0.9
- **용도**: 참고용

### 2순위 ⭐⭐

#### 4. Optimal Threshold (Youden Index)
- **정의**: TPR - FPR를 최대화하는 threshold
- **Code**: `_calculate_optimal_thresholds()` (Lines 769-897)
- **의미**: Sensitivity와 Specificity의 균형점
- **사용**: Test set에서 이 threshold 적용
- **출력 예시**:
  ```
  Emotion 0: Optimal threshold = 0.6234 (J=0.7123)
  Emotion 0: Sensitivity = 0.8456, Specificity = 0.8667
  ```

#### 5. Confusion Matrix 구성요소
- **코드상 직접 로깅 없음** (필요시 추가 분석)
- TP, TN, FP, FN
- Precision, Recall, F1 계산 가능

### Summary Metrics

- `{mode}_AUROC` (Line 521): 모든 emotion 평균
- `{mode}_balacc` (Line 520): 모든 emotion 평균
- `{mode}_acc` (Line 519): 모든 emotion 평균

---

## 📊 REGRESSION: Sparse Value Prediction

**Task**: "Emotion peak의 크기가 얼마인가?" (Regression)
**Code**: `pl_classifier.py` 524-767행

### 최우선순위 ⭐⭐⭐

#### 1. `{mode}_nonzero_adjusted_mae_{emotion_name}` (Line 628)
- **정의**: Original scale에서의 Non-zero MAE
- **의미**: Peak 값들의 실제 예측 오차 (normalized scale이 아닌 원본)
- **중요도**: **가장 중요!** Peak magnitude prediction의 핵심 지표
- **해석**: 낮을수록 좋음 (실제 emotion scale 단위)
- **예시**:
  - Sad: 0-27 scale → MAE 2.5 = 평균 2.5 포인트 오차
  - Positive: 0-7.4 scale → MAE 1.2 = 평균 1.2 포인트 오차

#### 2. `{mode}_nonzero_pearson_{emotion_name}` (Line 627)
- **정의**: Non-zero 샘플에서 예측값과 실제값의 Pearson correlation
- **의미**: Peak 크기의 순서/관계를 얼마나 잘 잡는가
- **중요도**: Magnitude 관계 파악 능력의 핵심
- **해석**:
  - 0.0-0.3 = Weak
  - 0.3-0.5 = Moderate
  - 0.5-0.7 = Strong
  - 0.7+ = Very strong

#### 3. `{mode}_detection_f1_{emotion_name}` (Line 669)
- **정의**: Zero vs Non-zero 구분 F1-score
- **의미**: Peak detection의 classification 성능
- **중요도**: Precision과 Recall의 조화평균
- **해석**: 높을수록 좋음 (0~1)
- **구성**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

#### 4. `{mode}_detection_auroc_{emotion_name}` (Line 671)
- **정의**: Zero vs Non-zero 구분 AUROC
- **의미**: Peak 발생 여부의 전반적 구분 능력
- **중요도**: Detection task의 종합 성능
- **해석**: Classification AUROC과 동일

### 2순위 ⭐⭐

#### 5. `{mode}_nonzero_adjusted_rmse_{emotion_name}` (Line 629)
- **정의**: Non-zero Root Mean Squared Error (original scale)
- **의미**: 큰 오차에 더 큰 패널티
- **용도**: Outlier sensitivity 확인
- **비교**: MAE vs RMSE → RMSE가 크면 큰 오차가 많음

#### 6. `{mode}_zero_adjusted_mae_{emotion_name}` (Line 750)
- **정의**: Zero 샘플에서 예측값의 평균 절대값
- **의미**: Zero를 얼마나 zero 근처로 예측하는가
- **중요도**: False positive 제어
- **해석**: 낮을수록 좋음 (이상적으로 0에 가까워야 함)

#### 7. Detection 상세 지표

##### `{mode}_detection_tpr_{emotion_name}` (Line 666)
- **정의**: True Positive Rate (Sensitivity, Recall)
- **의미**: 실제 peak를 얼마나 잘 찾아내는가
- **해석**: TPR = TP / (TP + FN)

##### `{mode}_detection_precision_{emotion_name}` (Line 668)
- **정의**: Precision
- **의미**: Peak라고 예측한 것 중 실제 peak 비율
- **해석**: Precision = TP / (TP + FP)

##### `{mode}_detection_specificity_{emotion_name}` (Line 670)
- **정의**: Specificity (True Negative Rate)
- **의미**: 실제 zero를 얼마나 잘 zero로 예측하는가
- **해석**: Specificity = TN / (TN + FP)

##### `{mode}_detection_fpr_{emotion_name}` (Line 667)
- **정의**: False Positive Rate
- **의미**: 실제 zero를 peak로 잘못 예측하는 비율
- **해석**: FPR = FP / (FP + TN) = 1 - Specificity

#### 8. Detection Threshold 정보
- **Default**: 0.5 (Line 636)
- **Binary labels**: target > threshold → 1, else → 0
- **Confusion matrix 기반 계산**

### 3순위 ⭐ (상세 분석용)

#### 9. Data Sparsity 파악

##### `{mode}_pct_zero_{emotion_name}` (Line 597)
- **정의**: Zero 샘플의 비율 (%)
- **용도**: 데이터가 얼마나 sparse한지 파악
- **예시**:
  - Positive: 38.7% zero
  - Sad: 90.1% zero (매우 sparse!)

##### `{mode}_n_total_{emotion_name}` (Line 594)
- 전체 샘플 수

##### `{mode}_n_zero_{emotion_name}` (Line 595)
- Zero 샘플 수

##### `{mode}_n_nonzero_{emotion_name}` (Line 596)
- Non-zero 샘플 수

#### 10. Magnitude-Stratified Metrics (Lines 677-727)

Peak 크기별로 세분화된 성능 분석:

##### Small peaks: 0 < x ≤ 1
- `{mode}_small_mae_{emotion_name}` (Line 696)
- `{mode}_small_mse_{emotion_name}` (Line 697)
- `{mode}_small_rmse_{emotion_name}` (Line 698)
- `{mode}_n_small_{emotion_name}` (Line 699)

##### Medium peaks: 1 < x ≤ 5
- `{mode}_medium_mae_{emotion_name}` (Line 710)
- `{mode}_medium_mse_{emotion_name}` (Line 711)
- `{mode}_medium_rmse_{emotion_name}` (Line 712)
- `{mode}_n_medium_{emotion_name}` (Line 713)

##### Large peaks: x > 5
- `{mode}_large_mae_{emotion_name}` (Line 724)
- `{mode}_large_mse_{emotion_name}` (Line 725)
- `{mode}_large_rmse_{emotion_name}` (Line 726)
- `{mode}_n_large_{emotion_name}` (Line 727)

**용도**: 특정 크기의 peak를 잘 예측하는지 확인

#### 11. Overall Metrics (참고용)

##### `{mode}_adjusted_mae_{emotion_name}` (Line 760)
- Zero 포함 전체 MAE (original scale)
- **주의**: Sparsity로 인해 misleading 가능

##### `{mode}_corrcoef_{emotion_name}` (Line 755)
- Zero 포함 전체 Pearson correlation
- **주의**: Zero가 많으면 낮게 나옴

##### `{mode}_r2_score_{emotion_name}` (Line 756)
- R² score (결정계수)
- **해석**: 모델이 설명하는 분산 비율

##### `{mode}_mse_{emotion_name}` (Line 757)
- Zero 포함 전체 MSE (normalized scale)

##### `{mode}_mae_{emotion_name}` (Line 758)
- Zero 포함 전체 MAE (normalized scale)

#### 12. Zero Prediction Metrics

##### `{mode}_zero_mae_{emotion_name}` (Line 747)
- Zero MAE (normalized scale)

##### `{mode}_zero_mean_pred_{emotion_name}` (Line 748)
- Zero 샘플에 대한 평균 예측값
- **이상적**: 0에 가까워야 함

##### `{mode}_zero_std_pred_{emotion_name}` (Line 749)
- Zero 샘플 예측값의 표준편차
- **이상적**: 낮을수록 좋음 (consistent하게 0 근처 예측)

### Summary Metrics (Lines 982-1034)

전체 감정에 대한 평균 성능:

#### `{mode}_avg_nonzero_mae` (Line 1009)
- 7개 emotion의 non-zero MAE 평균
- **가장 중요한 summary metric**

#### `{mode}_avg_nonzero_pearson` (Line 1015)
- 7개 emotion의 non-zero Pearson 평균

#### `{mode}_avg_nonzero_rmse` (Line 1021)
- 7개 emotion의 non-zero RMSE 평균

#### Standard Deviations
- `{mode}_std_nonzero_mae` (Line 1010)
- `{mode}_std_nonzero_pearson` (Line 1016)
- `{mode}_std_nonzero_rmse` (Line 1022)

---

## 📋 Quick Reference Checklists

### Classification Task 체크리스트

```python
# Sparse peak detection (binary classification)
priorities = {
    'tier1': [
        "AUROC_{emotion}",           # 1위: 구분 능력
        "balacc_{emotion}",          # 2위: 균형 정확도
        "acc_{emotion}",             # 3위: 전체 정확도
    ],
    'tier2': [
        "optimal_threshold",          # Youden Index
    ]
}
```

### Regression Task 체크리스트

```python
# Sparse value prediction
metrics = {
    'tier1': [
        "nonzero_adjusted_mae_{emotion}",    # Peak 예측 오차
        "nonzero_pearson_{emotion}",         # Peak 상관관계
        "detection_f1_{emotion}",            # Peak 발생 탐지 F1
        "detection_auroc_{emotion}",         # Peak 발생 탐지 AUROC
    ],

    'tier2': [
        "nonzero_adjusted_rmse_{emotion}",   # Outlier 민감도
        "zero_adjusted_mae_{emotion}",       # False positive 제어
        "detection_tpr_{emotion}",           # Sensitivity
        "detection_precision_{emotion}",     # Precision
        "detection_specificity_{emotion}",   # Specificity
    ],

    'tier3': [
        "pct_zero_{emotion}",                # Sparsity 파악
        "small/medium/large_mae_{emotion}",  # Magnitude별 분석
        "overall metrics",                   # 참고용 (misleading 가능)
    ],

    'summary': [
        "avg_nonzero_mae",                   # 전체 평균 (가장 중요)
        "avg_nonzero_pearson",
        "avg_nonzero_rmse",
    ]
}
```

---

## 🔍 Metric Interpretation Guide

### Regression: 좋은 모델의 기준

#### Minimum Success
- Non-zero Pearson > 0.35
- Non-zero MAE < baseline의 94%
- Detection AUROC > 0.70

#### Good Success
- Non-zero Pearson > 0.40
- Non-zero MAE < baseline의 90%
- Detection AUROC > 0.75

#### Excellent Success
- Non-zero Pearson > 0.45
- Non-zero MAE < baseline의 85%
- Detection AUROC > 0.80

### Classification: 좋은 모델의 기준

#### Good Performance
- AUROC > 0.80
- Balanced Accuracy > 0.75

#### Excellent Performance
- AUROC > 0.90
- Balanced Accuracy > 0.85

---

## 📊 Emotion-specific Considerations

### Emotion 특성 (예시)

| Emotion | Typical Sparsity | Difficulty | Key Metric |
|---------|------------------|------------|------------|
| **Positive** | ~40% zero | Easy | Nonzero Pearson |
| **Negative** | ~65% zero | Medium | Detection F1 |
| **Anger** | ~65% zero | Medium | Detection AUROC |
| **Happy** | ~75% zero | Hard | Nonzero MAE |
| **Fear** | ~75% zero | Hard | Nonzero MAE |
| **Sad** | ~90% zero | Very Hard | Detection metrics 우선 |
| **Excited** | ~80% zero | Hard | Balance detection + MAE |

**참고**: Sparsity가 높을수록 detection이 더 중요해짐

---

## ⚠️ Common Pitfalls

### 1. Overall MAE로만 평가
- ❌ Overall MAE가 낮다고 좋은 모델이 아님
- ✅ Zero를 많이 맞춰서 낮을 수 있음 (trivial solution)
- **해결**: Stratified metrics 필수

### 2. Accuracy 과신 (Classification)
- ❌ 90% accuracy라도 90% zero data에서 항상 0 예측하면 달성
- ✅ Balanced Accuracy, AUROC 확인 필수

### 3. Normalized scale로 해석
- ❌ MAE=0.5가 좋은지 나쁜지 알 수 없음
- ✅ Adjusted MAE (original scale) 확인

### 4. Magnitude 차이 무시
- ❌ 모든 emotion을 동일하게 평가
- ✅ Emotion별로 scale이 다름 (Sad: 0-27, Positive: 0-7.4)

---

## 💡 Best Practices

### 1. 항상 Stratified로 평가
- Zero vs Non-zero 분리
- Magnitude별 분리 (필요시)

### 2. Multiple metrics 종합 판단
- Single metric으로 판단 금지
- Tier 1 metrics를 모두 확인

### 3. Emotion별 개별 분석
- 평균만 보지 말고 각 emotion 확인
- 어떤 emotion이 어려운지 파악

### 4. Detection + Regression 동시 평가
- Regression task라도 detection 성능 중요
- Peak 발생 탐지 + 크기 예측 모두 확인

---

## 🔗 Related Documents

- `phase1b_evaluation_plan.md`: Phase 1B 평가 계획
- `pl_classifier.py`: Metric 구현 코드
- Emotion names: `['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']`

---

**Created**: 2025-11-18
**Author**: Analysis of pl_classifier.py
**Last Updated**: 2025-11-18
