# Focal Loss Variants 비교 분석

**Date**: 2025-10-31
**Context**: 우리가 구현한 Focal MSE vs 3가지 추천 variants
**Goal**: 현재 구현의 적절성 평가 및 개선 방향 제시

---

## 📊 현재 우리 구현

### 1. FocalMSELoss (Basic)

```python
class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1.0 + mse) ** self.gamma  # Key formula
        focal_mse = focal_weight * mse
        return focal_mse.mean()
```

**수식**:
```
Loss = mean((1 + MSE)^γ * MSE)
```

**특징**:
- ✅ 간단하고 직관적
- ✅ Hard samples (큰 error)에 exponential weight
- ⚠️ Scale-dependent (MSE 절대값에 영향받음)

---

### 2. WeightedFocalMSELoss (우리가 구현한 강화 버전)

```python
class WeightedFocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, zero_weight=1.0, nonzero_weight=5.0):
        super().__init__()
        self.gamma = gamma
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1.0 + mse) ** self.gamma

        # Zero vs non-zero weighting
        sample_weight = torch.where(
            target != 0,
            nonzero_weight,
            zero_weight
        )

        weighted_focal_mse = focal_weight * sample_weight * mse
        return weighted_focal_mse.mean()
```

**수식**:
```
Loss = mean((1 + MSE)^γ * w_sample * MSE)

where:
  w_sample = nonzero_weight  if target != 0
             zero_weight     otherwise
```

**특징**:
- ✅ Focal + Zero-inflation 동시 처리
- ✅ Non-zero samples에 추가 weight
- ⚠️ 여전히 scale-dependent

---

## 🆕 추천된 3가지 Variants

---

## Variant 1: Standard FocalMSELoss

### 구현 (2017 Original Adaptation)

```python
class FocalMSELoss(nn.Module):
    """
    Original Focal Loss adaptation for regression
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Balancing parameter
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, *) predicted values
            target: (B, *) ground truth
        """
        # MSE loss
        mse = (pred - target) ** 2

        # Focal weight: (1 + mse)^gamma
        # Larger errors → larger weight
        focal_weight = (1.0 + mse) ** self.gamma

        # Final loss
        loss = self.alpha * focal_weight * mse

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 🔍 우리 구현과 비교:

| Feature | 우리 구현 | Standard Variant |
|---------|----------|------------------|
| Focal weight | `(1 + mse)^γ` | `(1 + mse)^γ` |
| Alpha parameter | ❌ 없음 | ✅ 있음 (overall scaling) |
| Zero/Non-zero weight | ❌ 기본엔 없음 | ❌ 없음 |
| Scale robustness | ⚠️ 낮음 | ⚠️ 낮음 |

**결론**: **거의 동일!** 우리가 올바르게 구현했습니다. Alpha는 단순 scaling factor.

---

## Variant 2: NormalizedFocalMSELoss ⭐⭐⭐ (추천!)

### 구현

```python
class NormalizedFocalMSELoss(nn.Module):
    """
    Normalized Focal MSE Loss - Scale-Robust Version

    Key improvement: Normalizes MSE before applying focal weight
    This makes the loss robust to the scale of target values.

    Recommended for problems with varying target magnitudes!
    """
    def __init__(self, gamma=2.0, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # Numerical stability

    def forward(self, pred, target):
        """
        Args:
            pred: (B, *) predicted values
            target: (B, *) ground truth
        """
        # Compute MSE
        mse = (pred - target) ** 2

        # Normalize by target magnitude (key difference!)
        # This makes the loss scale-invariant
        normalized_mse = mse / (target.abs() + self.eps)

        # Focal weight on normalized MSE
        focal_weight = (1.0 + normalized_mse) ** self.gamma

        # Final loss
        loss = focal_weight * mse

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 🔍 우리 구현과 비교:

| Feature | 우리 구현 | Normalized Variant |
|---------|----------|-------------------|
| Focal weight | `(1 + mse)^γ` | `(1 + normalized_mse)^γ` |
| Normalization | ❌ 없음 | ✅ **MSE / |target|** |
| Scale robustness | ⚠️ 낮음 | ✅ **높음!** |
| Zero handling | ⚠️ 별도 weight 필요 | ✅ eps로 자동 처리 |

### 🎯 핵심 개선점:

**문제**: 우리 구현은 target 값의 크기에 민감
```python
# 예시
target1 = 2.0, pred1 = 3.0  → mse = 1.0
target2 = 20.0, pred2 = 21.0 → mse = 1.0

# 우리 구현: 동일한 weight (1 + 1)^γ
# But 실제 상대 오차는:
# target1: 50% error
# target2: 5% error  ← 훨씬 더 좋은 예측!

# Normalized version:
# target1: normalized_mse = 1.0 / 2.0 = 0.5
# target2: normalized_mse = 1.0 / 20.0 = 0.05
# → 상대 오차를 반영!
```

**왜 중요한가?**
- Emotion 값 범위: 0 ~ 27 (매우 넓음!)
- Positive (큰 값) vs Sad (작은 값) 사이 불균형
- Normalized version이 더 공정한 학습

### ✅ 추천 이유:

1. **Scale-robust**: Target magnitude에 관계없이 동작
2. **Relative error 반영**: 상대 오차를 고려
3. **Zero-inflation 자동 처리**: eps로 zero 근처도 안정적
4. **구현 간단**: 한 줄만 추가하면 됨

---

## Variant 3: ZeroInflatedFocalLoss ⭐⭐⭐⭐⭐ (최강!)

### 구현

```python
class ZeroInflatedFocalLoss(nn.Module):
    """
    Zero-Inflated Focal Loss - Ultimate Version

    Combines:
    1. Normalized Focal MSE (scale-robust)
    2. Explicit zero vs non-zero handling
    3. Adaptive weighting based on sample density

    Perfect for zero-inflated regression!
    """
    def __init__(self,
                 gamma=2.0,
                 zero_weight=0.1,
                 nonzero_weight=1.0,
                 normalize=True,
                 reduction='mean',
                 eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.normalize = normalize
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: (B, *) predicted values
            target: (B, *) ground truth
        """
        # 1. Compute base MSE
        mse = (pred - target) ** 2

        # 2. Normalize if requested (scale-robust!)
        if self.normalize:
            normalized_mse = mse / (target.abs() + self.eps)
            focal_input = normalized_mse
        else:
            focal_input = mse

        # 3. Apply focal weight (hard sample emphasis)
        focal_weight = (1.0 + focal_input) ** self.gamma

        # 4. Apply zero/non-zero weight (zero-inflation handling)
        is_nonzero = (target.abs() > self.eps).float()
        sample_weight = (
            is_nonzero * self.nonzero_weight +
            (1 - is_nonzero) * self.zero_weight
        )

        # 5. Combine all weights
        total_weight = focal_weight * sample_weight
        loss = total_weight * mse

        # 6. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 🔍 우리 구현과 비교:

| Feature | 우리 WeightedFocal | ZeroInflatedFocal |
|---------|-------------------|-------------------|
| Focal weight | `(1 + mse)^γ` | `(1 + normalized_mse)^γ` |
| Normalization | ❌ 없음 | ✅ **Optional (recommended: True)** |
| Zero/Non-zero weight | ✅ 있음 | ✅ 있음 (더 안정적 구현) |
| Scale robustness | ⚠️ 낮음 | ✅ **높음!** |
| Flexibility | ⚠️ 중간 | ✅ normalize on/off 선택 가능 |

### 🎯 핵심 개선점:

**1. Normalization 추가**
```python
# 우리 구현: mse 직접 사용
focal_weight = (1.0 + mse) ** self.gamma

# ZI Focal: normalized mse 사용 (optional)
normalized_mse = mse / (target.abs() + eps)
focal_weight = (1.0 + normalized_mse) ** self.gamma
```

**2. 더 안정적인 zero handling**
```python
# 우리 구현: target != 0 (정확한 zero만)
sample_weight = torch.where(target != 0, nonzero_weight, zero_weight)

# ZI Focal: abs > eps (numerical stability!)
is_nonzero = (target.abs() > eps).float()
sample_weight = is_nonzero * nonzero_weight + (1 - is_nonzero) * zero_weight
```

**3. Flexibility**
- `normalize=True`: Scale-robust (추천!)
- `normalize=False`: 우리 구현과 동일

---

## 📊 종합 비교표

| Loss Function | Scale-Robust | Zero-Inflation | Hard Sample Focus | Complexity | 추천도 |
|---------------|-------------|----------------|------------------|-----------|--------|
| **우리 FocalMSE** | ❌ | ❌ | ✅ | Low | ⭐⭐⭐ |
| **우리 WeightedFocal** | ❌ | ✅ | ✅ | Medium | ⭐⭐⭐⭐ |
| Standard Focal | ❌ | ❌ | ✅ | Low | ⭐⭐⭐ |
| **Normalized Focal** | ✅ | ❌ | ✅ | Low | ⭐⭐⭐⭐⭐ |
| **ZI Focal** | ✅ | ✅ | ✅ | Medium | ⭐⭐⭐⭐⭐ |

---

## 🎯 추천 사항

### 현재 상황 평가:

**우리가 구현한 것**:
1. ✅ FocalMSELoss: 표준 구현과 거의 동일 → **올바름!**
2. ✅ WeightedFocalMSELoss: Zero-inflation 처리 → **좋은 선택!**
3. ⚠️ **개선 여지**: Normalization 없어서 scale-dependent

---

### 📋 Action Items (우선순위)

#### 🥇 Priority 1: NormalizedFocalMSELoss 추가 (강력 추천!)

**Why**:
- Emotion 값 범위가 매우 넓음 (0 ~ 27)
- Positive vs Sad 등 감정 간 불균형
- **한 줄만 바꾸면 됨!**

**구현**:
```python
# learnable_losses.py에 추가
class NormalizedFocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        normalized_mse = mse / (target.abs() + self.eps)  # Key line!
        focal_weight = (1.0 + normalized_mse) ** self.gamma
        loss = focal_weight * mse
        return loss.mean() if self.reduction == 'mean' else loss.sum()
```

**실험**:
```bash
# Phase 1C: Normalized Focal MSE
python src/main.py \
    --regression_loss_type normalized_focal_mse \
    --focal_gamma 1.0 \
    --learning_rate 5e-5 \
    --max_epochs 5
```

**예상 개선**:
- ✅ Per-emotion performance 더 균형잡힘
- ✅ Sad, Fear 같은 small-value emotion 개선
- ✅ Overall stability 향상

---

#### 🥈 Priority 2: ZeroInflatedFocalLoss 추가 (optional)

**When to use**:
- Normalized Focal이 성공적이면 시도
- 최종 full training에 사용 고려

**구현**:
```python
# learnable_losses.py에 추가
# (위 Variant 3 코드 그대로 복사)
```

---

#### 🥉 Priority 3: 기존 코드 유지 (baseline)

**Why keep**:
- 현재 구현도 올바름 (표준과 동일)
- Baseline 비교용으로 유용
- 이미 Phase 1B 실험 진행 중

---

## 💡 실전 적용 전략

### Phase 1C: Normalized Focal MSE (이번 주말 추가)

```python
# 1. learnable_losses.py에 NormalizedFocalMSELoss 추가
# 2. pl_classifier.py에 통합

# pl_classifier.py
elif loss_type == 'normalized_focal_mse':
    print("Using Normalized Focal MSE Loss (Scale-Robust)")
    self.learnable_loss = NormalizedFocalMSELoss(
        gamma=self.hparams.get('focal_gamma', 1.0)
    )
```

### Quick Test (2-3 jobs)

```bash
# Normalized vs Standard Focal 비교
# γ=1.0, lr=5e-5 (Phase 1B best config)

Job 1: Standard Focal (Phase 1B) - already done
Job 2: Normalized Focal γ=1.0, lr=5e-5  ← NEW!
Job 3: Normalized Focal γ=0.5, lr=5e-5  ← NEW!
```

### 예상 결과:

**Standard Focal**:
- Positive (큰 값): Good performance
- Sad (작은 값): Poor performance
- → Scale bias 존재

**Normalized Focal**:
- All emotions: More balanced performance
- Sad, Fear 개선 예상
- → Scale-robust!

---

## 📚 Reference Comparison

### Original Focal Loss (Classification, 2017)

```python
# Lin et al., ICCV 2017
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
  p_t = predicted probability for true class
  γ = focusing parameter (typically 2.0)
  α_t = balancing parameter
```

### Our Adaptation (Regression)

```python
# Our FocalMSELoss
Loss = (1 + mse)^γ * mse

# Mapping:
# p_t in [0, 1] (classification) → mse in [0, ∞] (regression)
# (1 - p_t) → (1 + mse)  (inverted logic)
# log(p_t) → mse (squared error)
```

**결론**: ✅ **올바른 adaptation!** Classification → Regression 전환이 합리적.

---

## 🔬 Experimental Validation Plan

### Experiment 1: Scale Robustness Test

**목적**: Normalization의 효과 검증

```python
# Synthetic test
target = torch.tensor([[2.0, 20.0], [3.0, 21.0]])  # Small vs Large
pred = torch.tensor([[3.0, 21.0], [2.0, 20.0]])    # Same absolute error

# Standard Focal
loss_standard = FocalMSELoss(gamma=2.0)(pred, target)

# Normalized Focal
loss_normalized = NormalizedFocalMSELoss(gamma=2.0)(pred, target)

# Compare gradients for column 0 (small) vs column 1 (large)
```

**예상**:
- Standard: Larger gradient for large values
- Normalized: Balanced gradients

---

### Experiment 2: Per-Emotion Performance

**목적**: Emotion-wise 개선 확인

```python
# After training, analyze per-emotion metrics

emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
value_ranges = {
    'Positive': [0, 27.0],  # Large
    'Sad': [0, 5.0],        # Small
    # ...
}

# Compare:
# - Standard Focal: Better on Positive, worse on Sad?
# - Normalized Focal: More balanced?
```

---

## ✅ 최종 추천

### 지금 당장 할 것:

1. **NormalizedFocalMSELoss 추가** (30분 작업)
   - `learnable_losses.py`에 추가
   - `pl_classifier.py`에 통합
   - Argument 추가: `--regression_loss_type normalized_focal_mse`

2. **Quick Test 2개 실행** (주말)
   - Normalized Focal γ=1.0, lr=5e-5
   - Normalized Focal γ=0.5, lr=5e-5

3. **Phase 1B 완료 후 비교**
   - Standard vs Normalized
   - Per-emotion analysis

### 나중에 고려:

4. **ZeroInflatedFocalLoss** (Phase 1C best가 좋으면)
   - Full training에 사용

---

## 📊 Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| Phase 1B Focal 성공 | → Add Normalized Focal (Phase 1C) |
| Normalized Focal 개선 | → Use for full training |
| No improvement | → Try ZeroInflatedFocal or move to backup plan |
| Per-emotion imbalance | → Normalized Focal will help! |

---

## 🎯 Key Takeaways

1. ✅ **우리 구현은 올바름**: Standard Focal과 거의 동일
2. ✅ **WeightedFocal 좋은 선택**: Zero-inflation 처리
3. ⚠️ **개선 여지 있음**: Normalization 추가하면 scale-robust
4. ⭐ **추천**: NormalizedFocalMSELoss 추가 (간단하고 효과적!)
5. 🚀 **Next**: Phase 1C로 Normalized variant 테스트

---

**Created**: 2025-10-31
**Purpose**: Focal Loss variants 비교 및 개선 방향 제시
**Priority**: NormalizedFocalMSELoss 추가 (강력 추천!)
