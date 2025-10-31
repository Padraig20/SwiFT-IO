# Backup Plan: Focal MSE/Tweedie 실패 시 대안

**Date**: 2025-10-31
**Scenario**: Phase 1B에서 Focal MSE와 Tweedie가 Baseline 대비 개선이 없거나 미미한 경우
**Goal**: 다음 실험 방향 제시

---

## 🚨 실패 시나리오 정의

### Scenario 1: 완전 실패
- Non-zero Pearson < 0.30 (Baseline 0.32 미달)
- Non-zero MAE > 2.50 (Baseline 2.45보다 나쁨)
- → Loss function 교체로는 해결 안 됨

### Scenario 2: 미미한 개선
- Non-zero Pearson: 0.32-0.35 범위 (< 10% 개선)
- Non-zero MAE: 2.35-2.45 범위 (< 5% 개선)
- → 통계적으로 유의미하지 않음

### Scenario 3: 부분 성공
- 일부 emotion에서만 개선 (예: Positive만)
- Sparse emotion (Sad, Fear)에서 여전히 실패
- → 접근법 재검토 필요

---

## 🎯 Alternative Approaches (우선순위)

---

## 🥇 Priority 1: Zero-Inflated Two-Stage Model

### 개요
**핵심 아이디어**: Zero classification과 magnitude regression을 **명시적으로 분리**

### Why This Could Work
1. ✅ 통계적으로 가장 이론적으로 탄탄
2. ✅ Zero vs non-zero detection을 classification으로 (이미 AUROC 0.995 증명!)
3. ✅ Magnitude prediction은 non-zero에만 집중
4. ✅ 각 task가 독립적으로 최적화 가능

### Architecture

```python
class ZeroInflatedEmotionDecoder(nn.Module):
    def __init__(self, backbone, hidden_dim=256, num_emotions=7):
        super().__init__()
        self.backbone = backbone  # SwiFT-IO encoder

        # Two separate heads
        self.zero_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_emotions)  # Logits for zero prob
        )

        self.magnitude_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_emotions)  # Magnitude prediction
        )

    def forward(self, x):
        # Shared representation
        features = self.backbone(x)  # (B, S, hidden_dim)

        # Two outputs
        zero_logits = self.zero_classifier(features)  # (B, S, 7)
        magnitude = self.magnitude_regressor(features)  # (B, S, 7)

        # Apply activation for magnitude (ensure non-negative)
        magnitude = F.softplus(magnitude)

        return zero_logits, magnitude
```

### Loss Function

```python
class ZeroInflatedLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha  # Weight between classification and regression
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, zero_logits, magnitude_pred, target):
        """
        Args:
            zero_logits: (B, S, 7) - logits for P(zero)
            magnitude_pred: (B, S, 7) - predicted magnitude
            target: (B, S, 7) - true values
        """
        # Stage 1: Binary classification (zero vs non-zero)
        is_zero = (target == 0).float()
        loss_clf = self.bce(zero_logits, is_zero)

        # Stage 2: Magnitude regression (only on non-zero)
        mask_nonzero = (target != 0)

        if mask_nonzero.sum() > 0:
            loss_reg = self.mse(
                magnitude_pred[mask_nonzero],
                target[mask_nonzero]
            )
        else:
            loss_reg = torch.tensor(0.0).to(target.device)

        # Combined loss
        total_loss = self.alpha * loss_clf + loss_reg

        return total_loss, loss_clf, loss_reg
```

### Inference

```python
def predict(zero_logits, magnitude_pred):
    """
    Combine zero probability and magnitude prediction
    """
    # Probability of non-zero
    prob_nonzero = torch.sigmoid(-zero_logits)  # 1 - P(zero)

    # Final prediction
    # Option 1: Hard threshold
    is_nonzero = (prob_nonzero > 0.5)
    final_pred = is_nonzero.float() * magnitude_pred

    # Option 2: Soft (expectation)
    final_pred = prob_nonzero * magnitude_pred

    return final_pred
```

### Expected Improvement
- **Non-zero Pearson**: 0.50+ (classification guides timing)
- **Non-zero MAE**: 1.80-2.00 (focused regression)
- **Overall MAE**: 0.70-0.80

### Implementation Time: ~1 week

---

## 🥈 Priority 2: PRIME (Proxy-based Imbalanced Regression)

### 개요
**Reference**: ICML 2025 "PRIME: Deep Imbalanced Regression with Proxies"

### Why This Could Work
1. ✅ 최신 SOTA for imbalanced regression
2. ✅ Magnitude-aware feature learning
3. ✅ Balanced representation space
4. ✅ 이론적으로 가장 발전된 접근

### High-Level Idea

```python
class PRIMEEmotionDecoder(nn.Module):
    def __init__(self, backbone, hidden_dim=256, num_emotions=7, num_proxies=100):
        super().__init__()
        self.backbone = backbone

        # Learnable proxies (uniformly distributed in target space)
        # Each proxy represents a "bin" of target values
        self.proxies = nn.Parameter(torch.randn(num_proxies, hidden_dim))
        self.proxy_values = nn.Parameter(
            torch.linspace(0, 30, num_proxies).unsqueeze(1).repeat(1, num_emotions)
        )  # (num_proxies, 7)

        self.regressor = nn.Linear(hidden_dim, num_emotions)

    def forward(self, x):
        features = self.backbone(x)  # (B, S, hidden_dim)

        # Align features to proxies
        # ...PRIME alignment logic...

        # Regression
        magnitude = self.regressor(features)

        return magnitude, features, alignment_info
```

### Loss Function

```python
class PRIMELoss(nn.Module):
    def __init__(self, lambda_align=1.0):
        super().__init__()
        self.lambda_align = lambda_align
        self.mse = nn.MSELoss()

    def forward(self, pred, target, features, proxies, proxy_values):
        # 1. Regression loss
        loss_reg = self.mse(pred, target)

        # 2. Proxy alignment loss
        # Find nearest proxy for each target value
        # Align feature to that proxy
        loss_align = compute_proxy_alignment(
            features, target, proxies, proxy_values
        )

        total_loss = loss_reg + self.lambda_align * loss_align

        return total_loss
```

### Challenges
- ⚠️ 구현 복잡도 높음
- ⚠️ Official code 없으면 시간 오래 걸림
- ⚠️ Hyperparameter tuning 필요 (num_proxies, lambda)

### Action Items
1. Wait for official code release (check ICML 2025)
2. Study paper in detail
3. Implement simplified version
4. Full implementation if promising

### Expected Timeline: 2-3 weeks

---

## 🥉 Priority 3: Multi-Task Learning (Classification + Regression)

### 개요
**핵심**: Classification의 성공 (AUROC 0.995)을 regression에 활용

### Architecture

```python
class MultiTaskEmotionDecoder(nn.Module):
    def __init__(self, backbone, hidden_dim=256, num_emotions=7):
        super().__init__()
        self.backbone = backbone

        # Shared encoder
        self.shared_proj = nn.Linear(backbone_dim, hidden_dim)

        # Task-specific heads
        self.cls_head = nn.Linear(hidden_dim, num_emotions)  # Binary classification
        self.reg_head = nn.Linear(hidden_dim, num_emotions)  # Magnitude

    def forward(self, x):
        # Shared features
        features = self.backbone(x)
        shared = self.shared_proj(features)

        # Two tasks
        cls_logits = self.cls_head(shared)  # Zero vs non-zero
        magnitude = self.reg_head(shared)  # Magnitude

        return cls_logits, magnitude
```

### Loss Function

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Weight for classification
        self.beta = beta   # Weight for regression
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, cls_logits, magnitude_pred, target):
        # Classification loss (zero vs non-zero)
        binary_target = (target > 0).float()
        loss_cls = self.bce(cls_logits, binary_target)

        # Regression loss (all samples)
        loss_reg = self.mse(magnitude_pred, target)

        # Combined
        total_loss = self.alpha * loss_cls + self.beta * loss_reg

        return total_loss
```

### Benefits
- ✅ Classification guides "when" emotions occur
- ✅ Regression focuses on "how much"
- ✅ Shared representation learning
- ✅ 구현 비교적 간단

### Expected Improvement
- Non-zero Pearson: 0.45+
- Non-zero MAE: 2.00-2.20

### Implementation Time: 3-4 days

---

## Priority 4: Density-Aware Weighted Loss

### 개요
**Reference**: NeurIPS 2024 "Density Ratio Estimation for Imbalanced Learning"

### Key Idea
Rare samples (큰 값)에 자동으로 높은 weight 부여

```python
class DensityWeightedLoss(nn.Module):
    def __init__(self, use_kde=True):
        super().__init__()
        self.use_kde = use_kde

    def estimate_density_weights(self, targets):
        """
        Estimate density and compute inverse weights
        """
        from scipy.stats import gaussian_kde

        targets_np = targets.cpu().numpy().flatten()

        # Remove zeros for density estimation
        nonzero_targets = targets_np[targets_np > 0]

        if len(nonzero_targets) > 10:
            # Fit KDE
            kde = gaussian_kde(nonzero_targets)
            density = kde(targets_np + 1e-8)  # Avoid zero

            # Inverse density = higher weight for rare
            weights = 1.0 / (density + 1e-8)

            # Normalize
            weights = weights / weights.mean()
        else:
            weights = np.ones_like(targets_np)

        return torch.FloatTensor(weights).to(targets.device)

    def forward(self, pred, target):
        # Get density weights
        weights = self.estimate_density_weights(target)

        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = (weights * mse).mean()

        return weighted_mse
```

### Pros & Cons
- ✅ Theoretically grounded
- ✅ Automatic weighting (no manual tuning)
- ⚠️ Density estimation overhead
- ⚠️ May be sensitive to outliers

### Implementation Time: 2-3 days

---

## Priority 5: Sequence-to-Sequence with Attention

### 개요
현재 문제가 **temporal smoothness** 부족일 수 있음

### Hypothesis
- Peak detection 실패 원인: 각 timepoint를 독립적으로 예측
- 실제 emotion은 시간적으로 smooth하게 변화
- Attention으로 temporal context 활용

### Architecture

```python
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, backbone, hidden_dim=256):
        super().__init__()
        self.backbone = backbone

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        self.regressor = nn.Linear(hidden_dim, num_emotions)

    def forward(self, x):
        # (B, S, C, H, W, D) → (B, S, hidden_dim)
        features = self.backbone(x)

        # Temporal attention: (S, B, hidden_dim)
        features = features.permute(1, 0, 2)
        attn_out, _ = self.temporal_attn(features, features, features)
        attn_out = attn_out.permute(1, 0, 2)  # (B, S, hidden_dim)

        # Regression
        output = self.regressor(attn_out)

        return output
```

### Additional: Temporal Smoothness Regularization

```python
def temporal_smoothness_loss(predictions, alpha=0.1):
    """
    Encourage smooth temporal predictions

    L_smooth = α * Σ |pred[t] - pred[t-1]|²
    """
    # predictions: (B, S, 7)
    diff = predictions[:, 1:, :] - predictions[:, :-1, :]
    smoothness_loss = (diff ** 2).mean()

    return alpha * smoothness_loss
```

### Implementation Time: 1 week

---

## Priority 6: Data Augmentation & Preprocessing

### 현재 문제 재검토
혹시 **data quality** 문제일 수 있음

### Investigation Points

#### 1. Label Noise
```python
# Check label consistency
# Same movie watched by multiple subjects
# Should have similar emotion patterns

def check_label_consistency():
    # Load all subjects' emotion labels
    # Compute correlation across subjects
    # If very low → label quality issue
```

#### 2. fMRI Preprocessing
```python
# Current: Smoothing + Z-normalization
# Try:
# - Different smoothing kernels
# - Subject-specific normalization
# - Temporal filtering (bandpass)
```

#### 3. Peak Selection
```python
# Current: All timepoints equally important
# Alternative: Focus on high-confidence peaks

def select_reliable_peaks(emotion_labels, threshold=0.8):
    """
    Only use timepoints where emotion > threshold percentile
    """
    for emotion in range(7):
        values = emotion_labels[:, emotion]
        nonzero = values[values > 0]

        if len(nonzero) > 0:
            thresh = np.percentile(nonzero, 80)
            # Only train on peaks above threshold
```

### Implementation Time: 1 week (investigation)

---

## 📊 Decision Tree

```
Phase 1B 결과 확인
    |
    ├─ 성공 (Non-zero Pearson > 0.40)
    │   └─> Full training with best config
    │       └─> 추가 분석 & 논문 작성
    |
    ├─ 부분 성공 (0.35 < Pearson < 0.40)
    │   ├─> Zero-Inflated Model 시도 (Priority 1)
    │   └─> 성공 시 Full training
    |
    └─ 실패 (Pearson < 0.35)
        ├─> 1주차: Zero-Inflated Model (Priority 1)
        │   └─> 여전히 실패 시
        ├─> 2주차: Multi-Task Learning (Priority 3)
        │   └─> 여전히 실패 시
        ├─> 3주차: Data Quality 재검토 (Priority 6)
        │   └─> Label noise? Preprocessing issue?
        └─> 4주차: PRIME 구현 시도 (Priority 2)
```

---

## 🎯 Emergency Action Plan

### 만약 모든 접근이 실패한다면?

#### Option A: Task Reformulation
- Regression 대신 **Ordinal Regression**
- Target bins: [0], (0, 2], (2, 5], (5, 10], (10, inf)
- Classification 문제로 전환

#### Option B: Hybrid Approach
- Peak detection (classification) + Magnitude estimation (regression)
- Two-stage pipeline

#### Option C: Subject-Level Analysis
- 혹시 일부 subject만 예측 가능한가?
- Subject clustering → Group-wise model

#### Option D: Different Emotion Representation
- Current: Per-timepoint continuous values
- Alternative: Segment-level summaries
- Or: Binary peaks + duration

---

## 📅 Timeline Estimate

| Week | Primary Task | Backup If Fails |
|------|-------------|-----------------|
| Week 1 (현재) | Phase 1B 완료 & 분석 | - |
| Week 2 | Full training (best config) | Zero-Inflated Model |
| Week 3 | Analysis & Paper writing | Multi-Task Learning |
| Week 4 | - | PRIME / Data Investigation |

---

## 📚 Reference Materials

### Papers to Read (if needed)

1. **Zero-Inflated Models**
   - Lambert (1992): "Zero-Inflated Poisson Regression"
   - Relevant for understanding two-stage approaches

2. **Multi-Task Learning**
   - Ruder (2017): "An Overview of Multi-Task Learning"
   - Kendall et al. (2018): "Multi-Task Learning Using Uncertainty"

3. **Imbalanced Regression**
   - Branco et al. (2016): "A Survey of Predictive Modeling on Imbalanced Domains"
   - Yang & Xu (2020): "Deep Regression on Manifolds"

4. **Temporal Models**
   - Vaswani et al. (2017): "Attention Is All You Need"
   - For temporal attention mechanisms

---

## 💡 Key Insights

### Why Current Approach Might Fail

1. **Loss function alone may not be enough**
   - 근본 문제가 feature representation일 수 있음
   - Zero-Inflated Model로 명시적 분리 필요

2. **Sparsity 정도가 너무 심함**
   - Sad: 90.1% zero
   - 단순 loss 변경으로는 한계

3. **Temporal structure 무시**
   - 각 timepoint 독립 예측
   - Smoothness constraint 필요

### Success Factors for Alternatives

✅ **Zero-Inflated Model**:
- Classification AUROC 0.995 이미 증명
- Magnitude만 집중하면 개선 가능성 높음

✅ **Multi-Task Learning**:
- 두 task가 서로 도움
- Joint training이 효과적

✅ **PRIME**:
- 최신 SOTA
- But 구현 시간 많이 필요

---

## 🔬 Experimental Protocol for Alternatives

각 alternative 시도 시:

### 1. Baseline Comparison
- 항상 current best (Focal MSE or Tweedie)와 비교
- Stratified metrics 사용

### 2. Ablation Study
- Component별 영향 분석
- 예: Zero-Inflated에서 classification head만, regression head만

### 3. Error Analysis
- 어떤 샘플에서 개선되었나?
- 어떤 emotion에서 개선되었나?

### 4. Computational Cost
- Training time
- Inference time
- Memory usage

---

## ✅ Checklist Before Moving to Alternatives

Phase 1B 실패 판단 전에 확인할 것:

- [ ] 모든 hyperparameter 조합 시도했는가?
- [ ] Training이 충분히 수렴했는가? (40+ epochs)
- [ ] Baseline과 공정한 비교인가? (동일 seed, split, epochs)
- [ ] Evaluation metric이 올바른가? (stratified)
- [ ] WandB 로그 확인: Loss curve가 정상인가?
- [ ] 일부 emotion에서는 성공했는가?

---

**Created**: 2025-10-31
**Purpose**: Focal MSE/Tweedie 실패 시 명확한 다음 단계 제시
**Priority**: Zero-Inflated Model → Multi-Task → PRIME → Data Investigation
