# Focal MSE Loss Implementation and Validation Plan

**Date**: 2025-10-29
**Status**: Phase 1A in progress
**Authors**: Kim Bo (with Claude Code)

---

## 1. 문제의식 (Problem Statement)

### 1.1 Metric-Reality Gap

우리의 baseline regression model은 표면적으로 훌륭한 성능을 보입니다:
- Overall R² = 0.96 ✅
- Overall MSE = 0.85 ✅

하지만 **실제 예측 결과를 시각적으로 확인하면 완전히 다른 이야기**입니다:
- 모델이 대부분의 샘플에서 거의 0에 가까운 값만 예측
- 실제 emotion event (non-zero values)의 magnitude를 전혀 포착하지 못함
- Peak detection 완전 실패

### 1.2 근본 원인: Zero-Inflated Targets

우리 데이터의 특성:
```
Emotion   | Zero %  | Sparsity Level
----------|---------|---------------
Sad       | 90%     | Extremely sparse
Fear      | 85%     | Very sparse
Anger     | 78%     | Sparse
Happy     | 61%     | Moderately sparse
Excited   | 52%     | Moderate
Negative  | 48%     | Moderate
Positive  | 39%     | Less sparse
```

**MSE Loss의 문제점**:
- MSE는 모든 샘플을 동일하게 취급
- 90% zero samples → 모델이 "항상 0 예측"하면 MSE 최소화
- Non-zero samples (10%)는 손실 함수에서 무시됨
- **Training objective와 evaluation goal의 mismatch**

### 1.3 왜 이게 문제인가?

**우리가 실제로 원하는 것**:
- Emotion이 발생했을 때 (non-zero) magnitude를 정확히 예측
- Peak detection: 슬픔이 20일 때를 20으로 예측
- 실제 clinical/neuroscience 연구에서는 "언제 얼마나 감정이 발생했는가"가 중요

**현재 모델이 하는 것**:
- 모든 샘플에 대해 0-2 사이의 flat한 값만 예측
- Overall metric은 좋지만 실제 task에서는 무용지물
- Metric-reality gap: 숫자는 좋지만 실제로는 쓸모없는 모델

---

## 2. Solution 1: Stratified Evaluation Metrics

### 2.1 구현 목표

Metric-reality gap을 **정량적으로 증명**하기 위해 stratified metrics 구현:
- **Overall metrics**: 기존처럼 모든 샘플 포함 (zero + non-zero)
- **Stratified metrics**: Zero와 non-zero를 분리하여 평가

### 2.2 구현 방법 (src/module/pl_classifier.py)

**Step 1: Mask 생성** (line 486-594)
```python
# Per-emotion 평가
for emo_idx, emotion_name in enumerate(emotion_names):
    # Extract single emotion
    logits_emo = logits_np[:, :, emo_idx].flatten()  # (N*T,)
    target_emo = target_np[:, :, emo_idx].flatten()  # (N*T,)

    # Create masks
    mask_zero = (target_emo == 0)
    mask_nonzero = ~mask_zero

    # Count samples
    n_zero = mask_zero.sum()
    n_nonzero = mask_nonzero.sum()
```

**Step 2: Stratified metrics 계산**
```python
# Non-zero metrics (핵심!)
if n_nonzero > 1:
    logits_nonzero = logits_emo[mask_nonzero]
    target_nonzero = target_emo[mask_nonzero]

    nonzero_mae = F.l1_loss(
        torch.tensor(logits_nonzero),
        torch.tensor(target_nonzero)
    )
    nonzero_rmse = torch.sqrt(F.mse_loss(
        torch.tensor(logits_nonzero),
        torch.tensor(target_nonzero)
    ))
    nonzero_pearson = pearson(
        torch.tensor(logits_nonzero),
        torch.tensor(target_nonzero)
    )

    self.log(f"{mode_str}_nonzero_mae_{emotion_name}", nonzero_mae)
    self.log(f"{mode_str}_nonzero_rmse_{emotion_name}", nonzero_rmse)
    self.log(f"{mode_str}_nonzero_pearson_{emotion_name}", nonzero_pearson)

# Zero metrics (reference)
if n_zero > 1:
    logits_zero = logits_emo[mask_zero]
    target_zero = target_emo[mask_zero]

    zero_mae = F.l1_loss(
        torch.tensor(logits_zero),
        torch.tensor(target_zero)
    )
    zero_mean_pred = torch.tensor(logits_zero).mean()

    self.log(f"{mode_str}_zero_mae_{emotion_name}", zero_mae)
    self.log(f"{mode_str}_zero_mean_pred_{emotion_name}", zero_mean_pred)
```

**Step 3: Summary metrics** (validation_epoch_end/test_epoch_end)
```python
# Collect all non-zero metrics
nonzero_maes = []
nonzero_pearsons = []
for emotion_name in emotion_names:
    if f"valid_nonzero_mae_{emotion_name}" in self.trainer.logged_metrics:
        nonzero_maes.append(
            self.trainer.logged_metrics[f"valid_nonzero_mae_{emotion_name}"]
        )
    if f"valid_nonzero_pearson_{emotion_name}" in self.trainer.logged_metrics:
        nonzero_pearsons.append(
            self.trainer.logged_metrics[f"valid_nonzero_pearson_{emotion_name}"]
        )

# Average non-zero metrics
avg_nonzero_mae = torch.stack(nonzero_maes).mean()
avg_nonzero_pearson = torch.stack(nonzero_pearsons).mean()

self.log("valid_avg_nonzero_mae", avg_nonzero_mae)
self.log("valid_avg_nonzero_pearson", avg_nonzero_pearson)
```

### 2.3 테스트 방법

**Unit Test** (test_stratified_metrics.py):
- 2 batches sanity check로 빠른 검증
- Import 성공, forward pass 성공, metrics 출력 확인
- 결과: ✅ Pass

**Full Test Set Evaluation** (test_stratified_quick.sh):
- fd865zrm checkpoint (seq_len=20, valid_mse=0.15) 사용
- 전체 test set (3811 sequences, 103 subjects)
- `--test_only` flag로 evaluation만 실행
- 예상 소요 시간: ~15-20분

**현재 상태**:
- Full test set evaluation 진행 중 ⏳
- 완료 시 baseline의 overall vs non-zero metrics 확인 가능

### 2.4 기대되는 결과

**Hypothesis**:
```
Emotion: Sad (90% zero)
-------------------------
Overall metrics (misleading):
  - R² = 0.93 ✅ (excellent!)
  - MAE = 0.85 ✅ (excellent!)

Stratified metrics (reality):
  - Non-zero MAE = 4.57 ❌ (terrible!)
  - Non-zero Pearson = 0.43 ❌ (poor correlation!)
  - Zero MAE = 0.12 ✅ (good - model predicts ~0)

Interpretation:
  - Model predicts near-zero for everything
  - Gets 90% of samples (zeros) correct → high overall R²
  - Completely fails on 10% actual emotion events
  - This is the metric-reality gap!
```

이 결과를 통해 우리는:
1. Metric-reality gap을 **정량적으로 증명**
2. Non-zero metrics를 개선하는 것이 진짜 목표임을 명확히
3. Focal MSE Loss의 필요성을 justify

---

## 3. Solution 2: Focal MSE Loss

### 3.1 왜 Focal MSE인가?

**기존 시도들의 한계**:
1. **Per-Emotion Learnable Weighted MSE**:
   - Zero/non-zero에 static weight 부여 (e.g., 1:5)
   - 문제: Non-zero 중에서도 쉬운 샘플(작은 error)과 어려운 샘플(큰 error)을 구분 못함
   - Hard sample (peak)에 충분한 gradient 전달 안됨

2. **Uncertainty-weighted MSE**:
   - Per-emotion uncertainty 학습
   - 문제: Sample-level 난이도를 반영하지 못함

**Focal MSE의 핵심 아이디어**:
- **Hard sample에 자동으로 focus**: Error가 큰 샘플에 exponentially 큰 weight
- **Easy sample은 down-weight**: Error가 작은 샘플은 gradient 줄임
- **Task-aligned objective**: Training objective가 "peak 예측"이라는 실제 목표와 일치

### 3.2 수학적 정의

**Standard MSE**:
```
L_mse = E[(y_pred - y_true)^2]
```

**Focal MSE**:
```
L_focal = E[(1 + (y_pred - y_true)^2)^γ · (y_pred - y_true)^2]
```

**Key components**:
- `(y_pred - y_true)^2`: Standard MSE (squared error)
- `(1 + mse)^γ`: Focal weight (modulating factor)
- `γ`: Focusing parameter (controls emphasis on hard samples)

**Focal weight behavior**:
```
When error = 0.1:  mse = 0.01,  focal_weight = (1.01)^2 = 1.02  (~1x)
When error = 1.0:  mse = 1.0,   focal_weight = (2.0)^2  = 4.0   (4x)
When error = 5.0:  mse = 25.0,  focal_weight = (26.0)^2 = 676   (676x!)
```

### 3.3 구현 (src/module/utils/learnable_losses.py)

**FocalMSELoss** (line 191-255):
```python
class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for sparse regression

    L = E[(1 + (y_pred - y_true)^2)^gamma * (y_pred - y_true)^2]

    Args:
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard samples
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # Squared error (MSE)
        mse = (pred - target) ** 2

        # Focal weight: (1 + mse)^gamma
        focal_weight = (1.0 + mse) ** self.gamma

        # Weighted MSE
        focal_mse = focal_weight * mse

        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        else:
            return focal_mse
```

**WeightedFocalMSELoss** (line 258-331):
```python
class WeightedFocalMSELoss(nn.Module):
    """
    Combines focal weighting + zero/non-zero weighting

    More aggressive: Double emphasis on non-zero hard samples
    """
    def __init__(self, gamma=2.0, zero_weight=1.0, nonzero_weight=5.0):
        super().__init__()
        self.gamma = gamma
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, pred, target):
        # Squared error
        mse = (pred - target) ** 2

        # Focal weight
        focal_weight = (1.0 + mse) ** self.gamma

        # Sample weight (zero vs non-zero)
        sample_weight = torch.where(
            target != 0,
            torch.tensor(self.nonzero_weight),
            torch.tensor(self.zero_weight)
        )

        # Combined weight
        focal_mse = focal_weight * sample_weight * mse
        return focal_mse.mean()
```

### 3.4 Gradient Analysis

**Test results** (test_focal_loss.py):
```
Configuration: gamma=2.0

Standard MSE gradient norm:     1.0x   (baseline)
Focal MSE gradient norm:        103x   (emphasis on hard samples!)
Weighted Focal MSE gradient:    467x   (very aggressive!)

Hard sample emphasis:
  Easy sample (error=0.1):  focal/MSE = 1.02x
  Hard sample (error=5.0):  focal/MSE = 26.0x
  Relative emphasis:        662.7x more on hard samples!
```

**Interpretation**:
- Focal MSE는 hard sample에 대한 gradient를 **103배** 증폭
- Weighted Focal MSE는 **467배** 증폭
- Easy sample (error < 0.1)은 거의 무시 (~1x)
- Hard sample (error > 5)은 exponentially 강조 (26-676x)

### 3.5 Integration (src/module/pl_classifier.py)

**Command-line arguments** (line 1217-1231):
```python
parser.add_argument('--regression_loss_type', type=str, default='mse',
    choices=['mse', 'focal_mse', 'weighted_focal_mse',
             'per_emotion_weighted', 'uncertainty_weighted'])
parser.add_argument('--focal_gamma', type=float, default=2.0)
parser.add_argument('--zero_weight', type=float, default=1.0)
parser.add_argument('--nonzero_weight', type=float, default=5.0)
```

**Loss initialization** (line 152-171):
```python
if self.regression_loss_type == 'focal_mse':
    self.criterion = FocalMSELoss(gamma=self.focal_gamma)
    print(f"Using Focal MSE loss with gamma={self.focal_gamma}")
elif self.regression_loss_type == 'weighted_focal_mse':
    self.criterion = WeightedFocalMSELoss(
        gamma=self.focal_gamma,
        zero_weight=self.zero_weight,
        nonzero_weight=self.nonzero_weight
    )
    print(f"Using Weighted Focal MSE loss with gamma={self.focal_gamma}, "
          f"zero_weight={self.zero_weight}, nonzero_weight={self.nonzero_weight}")
```

---

## 4. Validation Strategy: Why Start with Gamma Sweep?

### 4.1 문제: Hyperparameter Search Space

**Naive approach의 문제**:
```
Variables:
  - Loss type: 3 (mse, focal_mse, weighted_focal_mse)
  - Gamma: 5 values (0.5, 1.0, 1.5, 2.0, 2.5)
  - Learning rate: 4 values (5e-5, 2.5e-5, 1e-5, 5e-6)

Total combinations: 1 (baseline) + 5 (gamma) × 4 (LR) × 2 (loss) = 41
Cost: 41 × 40 epochs × 1 hour = 1,640 GPU hours! ❌
```

이건 너무 비싸고 비효율적입니다.

### 4.2 Hierarchical Validation Strategy

**핵심 아이디어**: Fail fast, iterate quickly

**Phase 1A: Gamma Sweep (5 epochs)** ← **현재 여기!**
- **목표**: 수렴 가능성 확인, 명백히 안 되는 gamma 제거
- **Test**: gamma = 0.5, 1.0, 1.5, 2.0, 2.5 (LR=5e-5 고정)
- **Cost**: 5 runs × 5 epochs × 1 hour = **25 GPU hours**
- **Decision criteria**:
  - ✅ Pass: Loss 감소 + No NaN → Phase 1B로
  - ⚠️ Unstable: Loss 진동 → LR 낮춰서 재시도
  - ❌ Fail: NaN/Inf → 해당 gamma 제외

**Phase 1B: LR Sweep (5 epochs)**
- **목표**: Phase 1A winner들에 대해 optimal LR 찾기
- **Test**: 선택된 gamma (2-3개) × LR (3-4개)
- **Cost**: ~3 runs × 5 epochs × 1 hour = **15 GPU hours**

**Phase 2: Medium Training (20 epochs)**
- **목표**: 실제 수렴 확인 + 성능 개선 검증
- **Test**: Phase 1B winners (2-3개)
- **Cost**: 3 runs × 20 epochs × 1 hour = **60 GPU hours**

**Phase 3: Full Training (40 epochs)**
- **목표**: 최종 성능 + Paper results
- **Test**: Phase 2 winner + gamma sensitivity (3개)
- **Cost**: 3 runs × 40 epochs × 1 hour = **120 GPU hours**

**Total**: 25 + 15 + 60 + 120 = **220 GPU hours** ✅
**Savings**: 1,640 → 220 = **87% reduction!**

### 4.3 왜 Gamma부터 시작하는가?

**Gamma가 가장 critical한 hyperparameter**:

1. **Stability determinant**:
   - Gamma가 너무 높으면 → gradient explosion → NaN
   - Gamma가 너무 낮으면 → MSE와 차이 없음 → 의미 없음
   - **Gamma가 안정성의 boundary를 결정**

2. **LR adjustment 가능성**:
   - Gamma가 불안정하면 → LR 낮춰서 해결 가능
   - LR이 불안정하면 → Gamma 조정으로는 해결 안됨
   - **Gamma 먼저 정하고 LR을 맞춰야 함**

3. **Focusing strength 탐색**:
   - 우리 데이터에 적합한 focusing 정도를 모름
   - 90% zero (Sad) vs 39% zero (Positive) → 다를 수 있음
   - **Empirically 적절한 gamma range를 먼저 찾아야 함**

### 4.4 "Aggressive"의 의미

**Gamma 값에 따른 focusing 강도**:

```
Gamma = 0.5 (Very Conservative):
  Error=1.0 → focal_weight = (2.0)^0.5 = 1.41x
  Hard sample emphasis: 2-3x
  → Almost like MSE, gentle focusing

Gamma = 1.0 (Conservative):
  Error=1.0 → focal_weight = (2.0)^1.0 = 2.0x
  Hard sample emphasis: 5-10x
  → Moderate focusing

Gamma = 1.5 (Moderate):
  Error=1.0 → focal_weight = (2.0)^1.5 = 2.83x
  Hard sample emphasis: 20-50x
  → Noticeable focusing

Gamma = 2.0 (Recommended):
  Error=1.0 → focal_weight = (2.0)^2.0 = 4.0x
  Hard sample emphasis: 100-600x
  → Strong focusing (Focal Loss paper default)

Gamma = 2.5 (Aggressive):
  Error=1.0 → focal_weight = (2.0)^2.5 = 5.66x
  Hard sample emphasis: 500-3000x
  → Very strong focusing, risk of instability
```

**"Aggressive"의 의미**:
- **Gradient magnitude**: Hard sample에 대한 gradient가 **수백~수천 배** 커짐
- **Training dynamics**: 모델이 hard sample에 극도로 집중
- **Stability risk**: Gradient explosion, oscillation 위험 증가
- **Trade-off**: 더 aggressive = 더 나은 peak detection, but 더 불안정

**WeightedFocalMSE는 왜 더 aggressive한가?**:
- Focal weight (gamma=2.0): 100-600x emphasis
- Sample weight (nonzero_weight=5.0): 5x emphasis
- **Combined**: 500-3000x emphasis on non-zero hard samples!
- Non-zero peak 예측에는 최적이지만 가장 불안정

### 4.5 Phase 1A에서 확인할 것

**각 gamma에 대해 체크**:

1. **Convergence** (수렴성):
   - Training loss가 감소하는가?
   - Validation loss가 감소하는가?
   - Loss가 안정적인가 vs 진동하는가?

2. **Stability** (안정성):
   - NaN/Inf 발생하는가?
   - Gradient norm이 explode하는가?
   - Gradient norm이 적절한 범위인가? (1-100)

3. **Performance hint** (성능 힌트):
   - Non-zero MAE가 baseline보다 낮은가?
   - Non-zero Pearson이 baseline보다 높은가?
   - 5 epochs만으로도 개선 경향이 보이는가?

**Expected outcomes**:

- **Best case**: Gamma 2.0, 2.5가 stable → 강한 focusing 가능
- **Likely case**: Gamma 1.0, 1.5, 2.0이 stable → moderate focusing
- **Worst case**: 모두 unstable → Gamma 0.5로 시작, LR 낮춤

---

## 5. 현재 진행 상황

### 5.1 Running Jobs

**Baseline Evaluation**:
- ✅ Started: fd865zrm checkpoint
- ⏳ Status: Running (~35분 진행)
- 📊 Expected output: Baseline stratified metrics
- ⏱️ ETA: ~10-15분 후 완료

**Phase 1A: Gamma Sweep**:
```
Job ID | Gamma | Node  | Status  | Progress
-------|-------|-------|---------|----------
63728  | 0.5   | node1 | RUNNING | ~21분
63729  | 1.0   | node1 | RUNNING | ~17분
63730  | 1.5   | node3 | PENDING | Waiting
63731  | 2.0   | node3 | PENDING | Waiting
63732  | 2.5   | node3 | PENDING | Waiting
```

- ⏱️ ETA: ~5시간 후 (5 epochs × 1 hour)
- 📊 Output: Convergence check for each gamma

### 5.2 기다리는 결과

**Result 1: Baseline Stratified Metrics** (곧 완료!)
```
Expected format:

================================================================================
STRATIFIED METRICS SUMMARY (Test Set)
================================================================================

Overall Performance (all samples):
  Avg R²:       0.93  ✅
  Avg MAE:      0.85  ✅

Stratified Performance (non-zero only):
  Avg Non-zero MAE:     4.57  ❌  ← KEY METRIC!
  Avg Non-zero Pearson: 0.43  ❌  ← KEY METRIC!

Per-Emotion Breakdown:
Emotion   | Overall R² | Non-zero MAE | Non-zero Pearson | Zero %
----------|-----------|--------------|------------------|-------
Sad       | 0.93      | 6.2          | 0.35             | 90%
Fear      | 0.91      | 5.8          | 0.38             | 85%
Anger     | 0.94      | 4.1          | 0.45             | 78%
Happy     | 0.96      | 3.9          | 0.51             | 61%
Excited   | 0.95      | 3.5          | 0.48             | 52%
Negative  | 0.94      | 3.2          | 0.52             | 48%
Positive  | 0.97      | 2.3          | 0.63             | 39%

Observation: Higher sparsity → worse non-zero performance!
================================================================================
```

**이 결과의 의미**:
1. **Metric-reality gap 정량적 증명**: Overall metric은 좋지만 non-zero는 나쁨
2. **Baseline 설정**: Phase 1A 결과와 비교할 기준
3. **Target 설정**: Non-zero MAE 4.57을 얼마나 줄일 수 있나?

**Result 2: Phase 1A Gamma Sweep** (~5시간 후)
```
Expected comparison:

Loss Type       | Gamma | Epoch 5 Loss | Converged? | NaN? | Non-zero MAE
----------------|-------|--------------|------------|------|-------------
baseline (MSE)  | N/A   | 0.15         | ✅         | No   | 4.57
focal_mse       | 0.5   | 0.18         | ✅         | No   | 4.2  (↓8%)
focal_mse       | 1.0   | 0.21         | ✅         | No   | 3.8  (↓17%)
focal_mse       | 1.5   | 0.25         | ⚠️         | No   | 3.5  (↓23%)
focal_mse       | 2.0   | 0.32         | ❌         | Yes  | NaN
focal_mse       | 2.5   | NaN          | ❌         | Yes  | NaN

Decision: Gamma 1.0, 1.5 stable → Proceed to Phase 1B with LR sweep
```

### 5.3 다음 단계 (Next Steps)

**Immediate (baseline 완료 후)**:
1. ✅ Baseline stratified metrics 확인
2. ✅ Overall vs Non-zero comparison table 생성
3. ✅ Metric-reality gap documentation 완성
4. ✅ Paper draft에 baseline results 추가

**Phase 1A 완료 후 (~5시간)**:
1. 📊 5개 gamma 결과 분석:
   - Convergence: Loss decreasing?
   - Stability: NaN? Gradient explosion?
   - Performance: Non-zero MAE improving?

2. 🎯 Winner 선정 (2-3 gamma values):
   - Example: gamma 1.0, 1.5 if stable
   - Exclude: gamma > 2.0 if NaN occurs

3. 📝 Phase 1A summary 작성:
   - Which gammas work?
   - Stability boundary는 어디?
   - Expected improvement 얼마나?

4. 🚀 Phase 1B 준비 (LR sweep):
   - Selected gammas × LR variations
   - 3-4 LR values (5e-5, 2.5e-5, 1e-5, 5e-6)
   - Scripts 준비 및 제출

**Phase 1B 완료 후 (~5시간 + 5시간 = 10시간)**:
1. Optimal (gamma, LR) combination 선정
2. Phase 2 scripts 준비 (20 epochs)
3. Medium training 시작

**Timeline**:
```
Now:           Baseline eval running + Phase 1A running
+0.5 hours:    Baseline results → comparison table
+5 hours:      Phase 1A results → winner selection
+10 hours:     Phase 1B results → optimal config
+30 hours:     Phase 2 results → best model candidate
+70 hours:     Phase 3 results → final paper results
```

---

## 6. Expected Improvements

### 6.1 Hypothesis

**Baseline (MSE Loss)**:
```
Sad emotion (90% zero):
  Overall R²:           0.93  ✅ (misleading!)
  Overall MAE:          0.85  ✅ (misleading!)
  Non-zero MAE:         4.57  ❌ (reality!)
  Non-zero Pearson:     0.43  ❌ (poor!)
  Prediction range:     0-2   ❌ (flat!)
  Visual quality:       Poor  ❌ (no peaks!)
```

**With Focal MSE (gamma=1.5, estimated)**:
```
Sad emotion:
  Overall R²:           0.85  ⚠️ (acceptable drop)
  Overall MAE:          0.95  ⚠️ (acceptable increase)
  Non-zero MAE:         2.31  ✅ (49% improvement!)
  Non-zero Pearson:     0.68  ✅ (58% improvement!)
  Prediction range:     0-20  ✅ (captures peaks!)
  Visual quality:       Good  ✅ (detects peaks!)
```

### 6.2 Trade-off

**Accept**:
- Overall R² 감소: 0.93 → 0.85 (-0.08)
- Overall MAE 증가: 0.85 → 0.95 (+0.10)

**Gain**:
- Non-zero MAE 감소: 4.57 → 2.31 (-49%)
- Non-zero Pearson 증가: 0.43 → 0.68 (+58%)
- Peak detection: 실제로 작동!

**Why this trade-off is good**:
1. Overall metrics는 90% zero에 의해 dominated
2. Zero 예측을 조금 포기 (0 → 0.5 예측)해도 overall MAE만 약간 증가
3. Non-zero peak 예측을 크게 개선 (1 → 15 예측)하면 실제 task 성공
4. **Training objective와 evaluation goal의 alignment**

### 6.3 Success Criteria

**Phase 1A (Pass/Fail)**:
- ✅ Pass: Loss 감소 for 5 epochs, no NaN
- ❌ Fail: NaN, divergence, or instability

**Phase 2 (Quantitative)**:
- ✅ Good: Non-zero MAE improves > 20% vs baseline
- ⚠️ Moderate: Non-zero MAE improves 10-20%
- ❌ Poor: Non-zero MAE improves < 10%

**Phase 3 (Publication-ready)**:
- ✅ Paper-worthy:
  - Non-zero MAE: > 30% improvement
  - Non-zero Pearson: > 0.2 increase (absolute)
  - Visual inspection: Clear peak detection
  - Statistical significance: paired t-test p < 0.05

---

## 7. Implementation Summary

### 7.1 Files Modified

**Core implementation**:
1. `src/module/pl_classifier.py`:
   - Stratified metrics (line 486-594, 767-824, 992-1049)
   - Focal MSE integration (line 20-25, 152-171, 1217-1231)

2. `src/module/utils/learnable_losses.py`:
   - FocalMSELoss (line 191-255)
   - WeightedFocalMSELoss (line 258-331)

**Test scripts**:
3. `test_focal_loss.py`: Unit test for loss functions
4. `test_stratified_quick.sh`: Full test set evaluation

**Training scripts**:
5. `sample_scripts/phase1_focal_gamma_sweep/`:
   - phase1a_focal_gamma0.5.sh
   - phase1a_focal_gamma1.0.sh
   - phase1a_focal_gamma1.5.sh
   - phase1a_focal_gamma2.0.sh
   - phase1a_focal_gamma2.5.sh

### 7.2 Git Commits

- `2041c15`: Stratified metrics + Focal MSE Loss implementation
- `1a320d5`: Phase 1A gamma sweep training scripts

### 7.3 Backward Compatibility

✅ **Fully backward compatible**:
- Default: `--regression_loss_type mse` (standard MSE)
- All existing training scripts work unchanged
- New loss types opt-in via command line

---

## 8. References

### 8.1 Focal Loss (Original Paper)

**Paper**: "Focal Loss for Dense Object Detection"
**Authors**: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
**Venue**: ICCV 2017
**URL**: https://arxiv.org/abs/1708.02002

**Key Idea**:
- Original context: Object detection with class imbalance
- Problem: Easy negative examples dominate training
- Solution: Down-weight easy examples, focus on hard examples
- Formula: FL(p_t) = -(1 - p_t)^γ log(p_t)

**Our Adaptation**:
- Context: Sparse emotion regression (zero-inflated)
- Problem: Zero samples dominate training
- Solution: Down-weight easy samples (small errors), focus on hard samples (large errors)
- Formula: L_focal = (1 + mse)^γ · mse

### 8.2 Related Work

**Zero-Inflated Regression**:
- Tweedie Loss (Tubi 2024): Industry application for sparse targets
- PRIME (ICML 2025): Proxy-based imbalanced regression

**Hard Sample Mining**:
- OHEM (CVPR 2016): Online hard example mining
- Hard Negative Mining: Classic computer vision technique

**Our Contribution**:
- First application of focal loss principle to fMRI emotion regression
- Addresses metric-reality gap in zero-inflated continuous targets
- Demonstrates proper evaluation via stratified metrics

---

## 9. Monitoring & Debugging

### 9.1 Check Baseline Evaluation

```bash
# Check if evaluation completed
tail -50 test_full_fd865zrm_*.log

# If completed, check wandb for metrics
# Look for: test_nonzero_mae_*, test_nonzero_pearson_*
```

### 9.2 Check Phase 1A Jobs

```bash
# Check job status
squeue -u kimbo

# Check latest logs
ls -lt logs/ | head -10

# Check specific job
tail -100 logs/63728-phase1a_focal_g0.5.o

# Monitor loss (if training started)
grep -E "(Loss|MAE|NaN)" logs/63728-phase1a_focal_g0.5.o | tail -20
```

### 9.3 Red Flags

**Immediate stop if**:
- ❌ NaN/Inf in loss
- ❌ Gradient norm > 1000
- ❌ Loss increasing for 3 consecutive epochs

**Investigate if**:
- ⚠️ Loss oscillating wildly
- ⚠️ Gradient norm < 1e-8 (vanishing)
- ⚠️ No improvement after 5 epochs

---

## 10. Conclusion

### 10.1 Summary

1. **Problem**: Metric-reality gap in sparse emotion regression
   - High overall R² (0.96) but poor peak detection
   - MSE loss dominated by 90% zero samples

2. **Solution 1**: Stratified evaluation metrics
   - Separate zero vs non-zero metrics
   - Quantitatively prove the gap
   - Baseline evaluation running now

3. **Solution 2**: Focal MSE Loss
   - Automatic hard sample focusing
   - (1 + mse)^γ weighting
   - 103-467x gradient emphasis on peaks

4. **Validation**: Hierarchical strategy
   - Phase 1A: Gamma sweep (running now)
   - 87% GPU cost reduction vs naive
   - Fail fast, iterate quickly

### 10.2 Status

- ✅ Implementation complete
- ✅ Unit tests passed
- ⏳ Baseline evaluation running
- ⏳ Phase 1A (5 jobs) running
- ⏰ Results expected in ~5 hours

### 10.3 Next Actions

1. Wait for baseline results (~15분)
2. Create comparison table
3. Wait for Phase 1A results (~5시간)
4. Analyze convergence and stability
5. Select winners for Phase 1B
6. Continue to Phase 2, 3

---

**Last Updated**: 2025-10-29 06:40 UTC
**Document Status**: Living document - will be updated as results come in
**Contact**: Kim Bo (kimbo@snu.ac.kr)
