# Sparse & Imbalanced Regression Loss Functions - Literature Survey

**Date**: 2025-10-29
**Focus**: Regression tasks with sparse/imbalanced targets (magnitude prediction)
**Context**: fMRI emotion decoding with zero-inflated continuous targets

---

## 🎯 Research Question

**문제**: Sparse & imbalanced regression에서 magnitude를 정확히 예측하는 방법
- 대부분의 target이 0 (zero-inflated)
- 0이 아닌 값의 magnitude가 중요 (classification이 아닌 regression)
- 기존 MSE는 majority (0) 값에 지배됨

---

## 🔥 최신 연구 (2024-2025)

### 1. PRIME: Deep Imbalanced Regression with Proxies ⭐⭐⭐

**Conference**: ICML 2025  
**Paper**: "PRIME: Deep Imbalanced Regression with Proxies"  
**URL**: https://icml.cc/virtual/2025/poster/44388

**핵심 아이디어**:
- **Learnable proxies**를 feature space에 배치
- Proxies는 target value의 순서를 반영하며 균등하게 분포
- 각 데이터 포인트를 적절한 proxy에 align하도록 학습
- Classification의 class imbalance 해결법을 regression으로 확장

**장점**:
- ✅ 최신 연구 (2025)
- ✅ Imbalanced regression 전용으로 설계
- ✅ Unified framework - 다양한 imbalance 패턴에 적용 가능
- ✅ Magnitude prediction에 효과적

**적용 방법**:
```python
# Proxy-based loss
# 1. Create learnable proxies uniformly distributed in feature space
# 2. Align each sample to its corresponding proxy based on target value
# 3. Minimize alignment loss

# Pseudo-code structure:
proxies = create_uniform_proxies(n_proxies, feature_dim)
proxy_assignments = assign_to_proxy(targets, proxies)
alignment_loss = align_features_to_proxies(features, proxies, proxy_assignments)
regression_loss = mse(predictions, targets)
total_loss = regression_loss + λ * alignment_loss
```

**추천도**: ⭐⭐⭐⭐⭐ (가장 최신이며 imbalanced regression 전용)

---

### 2. Density-Aware Reweighting via Density Ratio Estimation

**Conference**: NeurIPS 2024  
**Paper**: "Revive Re-weighting in Imbalanced Learning by Density Ratio Estimation"  
**URL**: https://proceedings.neurips.cc/paper_files/paper/2024

**핵심 아이디어**:
- **Density ratio**를 이용한 sample reweighting
- Empirical label distribution이 true label density를 정확히 반영하지 못하는 문제 해결
- Conditional probability ratio: `r(x|y) = P_bal(x|y) / P(x|y)`

**수식**:
```
Loss = E_P[ r(x|y) * loss(x, y) ]

where r(x|y) = P_balanced(x|y) / P_empirical(x|y)
```

**장점**:
- ✅ Regression의 continuous label space에 특화
- ✅ Theoretically grounded (density ratio estimation)
- ✅ Balanced risk minimization

**적용 난이도**: Medium (density ratio 추정 필요)

**추천도**: ⭐⭐⭐⭐ (이론적으로 탄탄, 구현 약간 복잡)

---

### 3. Tweedie Loss for Zero-Inflated Regression ⭐⭐⭐

**Source**: Industry (Tubi) + Academic research (2024)  
**Paper**: "Optimizing Video Recommendation Systems: A Deep Dive into Tweedie Regression"  
**URL**: https://www.shaped.ai/blog/optimizing-video-recommendation-systems

**핵심 아이디어**:
- **Tweedie distribution**: Zero-inflated + positively skewed continuous data 모델링
- Compound Poisson-Gamma process
- Power parameter `p` (1 < p < 2)로 zero-inflation 정도 조절

**수식**:
```python
# Tweedie Loss
L_tweedie = -log(p(y | μ, p, φ))

where:
- μ = predicted mean
- p = power parameter (1 < p < 2)
- φ = dispersion parameter
```

**장점**:
- ✅ **Zero-inflated data에 최적화** ← 당신의 문제와 정확히 일치!
- ✅ Industry에서 검증됨 (Tubi: +0.4% revenue, +0.15% watch time)
- ✅ 구현 간단 (scikit-learn, PyTorch에 있음)
- ✅ Magnitude prediction 우수

**실제 적용 예시**:
```python
import torch
import torch.nn as nn

class TweedieLoss(nn.Module):
    def __init__(self, p=1.5):
        super().__init__()
        self.p = p  # 1 < p < 2
    
    def forward(self, pred, target):
        # Tweedie deviance
        if self.p == 1:  # Poisson
            loss = -target * torch.log(pred + 1e-8) + pred
        elif self.p == 2:  # Gamma
            loss = torch.log(pred + 1e-8) + target / (pred + 1e-8)
        else:  # General Tweedie
            loss = (
                -target * torch.pow(pred + 1e-8, 1 - self.p) / (1 - self.p)
                + torch.pow(pred + 1e-8, 2 - self.p) / (2 - self.p)
            )
        return loss.mean()

# Usage
criterion = TweedieLoss(p=1.5)  # p 값은 hyperparameter
```

**추천도**: ⭐⭐⭐⭐⭐ (Zero-inflated에 perfect fit!)

---

### 4. Focal Loss for Regression (Variants)

**Source**: Multiple papers (2024-2025)  
**Key Papers**: 
- "Uncertainty Weighted Gradients for Model Calibration" (CVPR 2025)
- "A Comprehensive Survey of Loss Functions" (2024)

**핵심 아이디어**:
- Classification의 focal loss를 regression에 적용
- Hard examples (큰 error)에 집중

**Variants**:

#### a) Focal MSE Loss
```python
class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Focus on hard samples
        focal_weight = (1 + mse) ** self.gamma
        return (focal_weight * mse).mean()
```

#### b) Adaptive Focal Loss (with gradient magnitude adjustment)
```python
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Adaptively weight based on error magnitude
        weight = torch.exp(self.alpha * mse / (mse.mean() + 1e-8))
        return (weight * mse).mean()
```

**장점**:
- ✅ 구현 매우 간단
- ✅ 기존 MSE에 쉽게 추가 가능
- ✅ Hard examples (0이 아닌 값)에 자동 집중

**추천도**: ⭐⭐⭐⭐ (간단하고 효과적)

---

### 5. Zero-Inflated Model-Based Losses

**Source**: Statistical ML literature (2024)  
**Key Papers**:
- "Zero-Inflated Data: A Comparison of Regression Models"
- "Uncertainty-aware probabilistic graph neural networks" (2024)

**핵심 아이디어**:
- **Two-stage model**:
  1. Classification: Zero vs Non-zero (Binary)
  2. Regression: Magnitude prediction (Continuous, on non-zero only)

**수식**:
```
P(y | x) = π(x) * δ_0 + (1 - π(x)) * f(y | x, y > 0)

where:
- π(x) = probability of zero
- δ_0 = point mass at zero
- f(y | x, y > 0) = continuous distribution for non-zero
```

**구현**:
```python
class ZeroInflatedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_zero_prob, pred_magnitude, target):
        # Stage 1: Zero vs Non-zero classification
        is_zero = (target == 0).float()
        loss_classification = self.bce(pred_zero_prob, is_zero)
        
        # Stage 2: Magnitude prediction (only on non-zero)
        mask_nonzero = (target != 0)
        if mask_nonzero.sum() > 0:
            loss_regression = self.mse(
                pred_magnitude[mask_nonzero],
                target[mask_nonzero]
            )
        else:
            loss_regression = 0
        
        return loss_classification + loss_regression
```

**장점**:
- ✅ Zero와 magnitude를 명시적으로 분리
- ✅ Interpretable (두 단계가 명확)
- ✅ 통계적 근거 확실

**단점**:
- ⚠️ Model architecture 변경 필요 (두 개의 output head)

**추천도**: ⭐⭐⭐⭐ (명확하고 효과적, 구조 변경 필요)

---

## 📊 비교 요약

| Method               | Year    | Ease of Use | Theoretical | Zero-Inflated | Magnitude | Recommended |
| -------------------- | ------- | ----------- | ----------- | ------------- | --------- | ----------- |
| **PRIME**            | 2025    | Medium      | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐       |
| **Tweedie Loss**     | 2024    | Easy        | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐       |
| **Density Reweight** | 2024    | Medium      | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐          | ⭐⭐⭐⭐      | ⭐⭐⭐⭐        |
| **Focal MSE**        | 2024    | Very Easy   | ⭐⭐⭐         | ⭐⭐⭐           | ⭐⭐⭐⭐      | ⭐⭐⭐⭐        |
| **Zero-Inflated**    | 2024    | Hard        | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐      | ⭐⭐⭐⭐        |
| Weighted MSE         | Classic | Very Easy   | ⭐⭐          | ⭐⭐            | ⭐⭐⭐       | ⭐⭐⭐         |

---

## 🎯 당신의 문제에 맞는 추천 순위

### 🥇 1st Priority: **Tweedie Loss** ⭐⭐⭐⭐⭐

**이유**:
- Zero-inflated regression의 gold standard
- 구현 간단, 이론적으로 탄탄
- Industry에서 검증됨
- **당신의 데이터 특성과 정확히 일치** (sparse, continuous, magnitude important)

**시작 코드**:
```python
criterion = TweedieLoss(p=1.5)  # 1 < p < 2
# p를 grid search로 찾기: [1.2, 1.5, 1.8]
```

---

### 🥈 2nd Priority: **Zero-Inflated Model** ⭐⭐⭐⭐

**이유**:
- Zero classification + magnitude regression 명시적 분리
- Excited (binary) 문제도 자연스럽게 해결
- Interpretable

**구조**:
```python
# Model architecture
class EmotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...  # fMRI encoder
        self.zero_head = nn.Linear(hidden, 7)  # Zero probability
        self.magnitude_head = nn.Linear(hidden, 7)  # Magnitude
    
    def forward(self, x):
        h = self.encoder(x)
        zero_logits = self.zero_head(h)
        magnitude = self.magnitude_head(h)
        return zero_logits, magnitude
```

---

### 🥉 3rd Priority: **PRIME (if feasible)** ⭐⭐⭐⭐⭐

**이유**:
- 가장 최신 (ICML 2025)
- Imbalanced regression 전용 SOTA
- Magnitude prediction 우수

**주의**:
- 구현이 약간 복잡 (proxy learning 필요)
- Official code 나오면 시도 권장

---

### 4th Priority: **Focal MSE (baseline)** ⭐⭐⭐⭐

**이유**:
- 가장 구현 쉬움
- Baseline으로 좋음
- Weighted MSE보다 개선

---

## 💡 실전 적용 전략

### Step 1: Baseline (이번 주)
```python
# Tweedie Loss 구현 & 실험
criterion = TweedieLoss(p=1.5)
# Grid search p: [1.2, 1.5, 1.8]
```

### Step 2: Comparison (다음 주)
```python
# Zero-Inflated Model 구현
# vs Tweedie 비교
```

### Step 3: Advanced (향후)
```python
# PRIME 구현 (official code 나오면)
```

---

## 📚 Key References

### Must-Read Papers

1. **PRIME (ICML 2025)**
   - Title: "PRIME: Deep Imbalanced Regression with Proxies"
   - URL: https://icml.cc/virtual/2025/poster/44388
   - Code: TBA

2. **Tweedie Regression (2024)**
   - Title: "Optimizing Video Recommendation Systems"
   - URL: https://www.shaped.ai/blog/optimizing-video-recommendation-systems
   - Code: Available in scikit-learn, PyTorch

3. **Density-Aware Reweighting (NeurIPS 2024)**
   - Title: "Revive Re-weighting in Imbalanced Learning"
   - URL: https://proceedings.neurips.cc/paper_files/paper/2024

4. **Zero-Inflated Models (2024)**
   - Title: "Zero-Inflated Data: A Comparison of Regression Models"
   - URL: https://towardsdatascience.com/zero-inflated-data-comparison

### Survey Papers

5. **Loss Functions Survey (2025)**
   - Title: "A Comprehensive Survey of Loss Functions"
   - URL: https://link.springer.com/article/10.1007/s10462-025-11198-7
   - Comprehensive overview of all loss functions

6. **Focal Loss Review (2024)**
   - Title: "Loss Functions in Deep Learning: A Comprehensive Review"
   - URL: https://arxiv.org/html/2504.04242v1

---

## 🔬 Related Work

### Imbalanced Regression (General)
- "Deep Imbalanced Regression" (Zhang et al. 2023)
- "Regularization for Deep Imbalanced Regression" (IEEE 2024)

### Zero-Inflated Models
- "Zero-Inflated Poisson/Negative Binomial" (Lambert 1992)
- "Uncertainty-aware probabilistic graph neural networks" (2024)

### Focal Loss Variants
- "Uncertainty Weighted Gradients" (CVPR 2025)
- "Dual Focal Loss" (2024)

---

## 🎯 Action Items for Your Project

- [ ] **Week 1**: Implement Tweedie Loss
  - [ ] 구현 (30분)
  - [ ] Grid search p parameter [1.2, 1.5, 1.8]
  - [ ] Compare with baseline MSE
  - [ ] Plot: pred vs true for each emotion

- [ ] **Week 2**: Implement Zero-Inflated Model
  - [ ] Modify architecture (two heads)
  - [ ] Implement ZI loss
  - [ ] Compare with Tweedie

- [ ] **Week 3**: Analysis
  - [ ] Per-emotion performance
  - [ ] Non-zero sample correlation
  - [ ] Error distribution analysis

- [ ] **Future**: Track PRIME official code release
  - [ ] Check ICML 2025 proceedings
  - [ ] Implement when available

---

## 📝 Notes

### Why These Methods Work for Your Problem

1. **Tweedie**: 
   - Explicitly models zero-inflation
   - Continuous distribution for non-zero
   - Parameter `p` controls sparsity

2. **Zero-Inflated**:
   - Separates zero classification from magnitude
   - Each task optimized independently
   - Natural for your 90% zero (Sad) case

3. **PRIME**:
   - Learns balanced feature space
   - Proxies guide minority samples
   - Magnitude-aware alignment

### Your Data Characteristics Recap

- **7 emotions**: Different sparsity levels (38% ~ 90% zero)
- **Zero-inflated**: Most values are 0
- **Continuous non-zero**: Range [0.05, 27.0]
- **Magnitude matters**: Not just presence/absence
- **Subject-wise**: Same timeseries for all subjects

→ **Tweedie Loss** or **Zero-Inflated Model** are perfect fits!

---

**Generated**: 2025-10-29 12:20:31
