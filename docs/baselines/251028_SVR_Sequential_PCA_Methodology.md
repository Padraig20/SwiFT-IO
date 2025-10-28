# SVR with Sequential PCA Baseline Methodology

**작성일**: 2025-10-28
**목적**: SwiFT-IO와 비교를 위한 Sequential PCA + SVR baseline 설명

---

## 📊 Overview

Sequential PCA + SVR baseline은 **시간적 정보를 유지**하면서도 차원을 효과적으로 축소하는 전통적인 머신러닝 접근법입니다. Time-averaging 방식과 달리, 30개 timepoints의 temporal dynamics를 보존하면서 각 timepoint를 독립적으로 PCA로 축소한 후 concatenate합니다.

**핵심 아이디어**:
- 각 timepoint를 동일한 PCA transformation으로 축소
- 30개의 축소된 벡터를 concatenate하여 최종 feature 생성
- SVR로 emotion regression 수행

---

## 🔧 Method Details

### 1. Dimensionality Reduction Strategy

#### 1.1 Sequential PCA (시간 정보 유지)

**Input**: fMRI sequence `(batch, 96, 96, 96, 30)`
- 96×96×96 = 884,736 voxels per timepoint
- 30 timepoints per sequence

**Process**:

```
Step 1: Reshape all training timepoints
    Training data: (11,344 sequences, 30 timepoints)
    → Total: 340,320 timepoints

    For PCA fitting, treat each timepoint as independent sample:
    (340,320 timepoints, 884,736 voxels)

Step 2: Fit IncrementalPCA
    PCA.partial_fit() on batches of timepoints
    → Learn 100 principal components
    → Variance explained: ~40-50%

Step 3: Transform each sequence
    For sequence with 30 timepoints:
        For t = 0, 1, ..., 29:
            timepoint[t]: (96, 96, 96) = 884,736 voxels
                ↓ [Flatten]
            (884,736,)
                ↓ [PCA.transform()]
            (100,) ← reduced representation

        Concatenate all 30 timepoints:
            [t0: 100-d, t1: 100-d, ..., t29: 100-d]
                ↓
            Final feature: (3,000,)

Step 4: Target preparation
    Target sequence: (30, 7) emotions
        ↓ [Mean over time]
    Target: (7,) average emotions
```

**최종 Feature Dimension**:
- **3,000 features** (30 timepoints × 100 PCA components)

---

### 2. PCA Fitting vs Inference

#### 2.1 Training Phase (PCA Fitting)

```python
# IncrementalPCA 초기화
pca_model = IncrementalPCA(n_components=100, batch_size=1000)

# 모든 training timepoints로 batch-wise fitting
for batch in train_dataloader:
    fmri_data = batch['fmri_sequence']  # (B, 1, 96, 96, 96, 30)

    # Reshape: (B, 30, 96, 96, 96) → (B*30, 884736)
    # 모든 timepoints를 독립적인 samples로 취급!
    X_batch = fmri_data.reshape(B * 30, 884736)

    # Incremental fitting
    pca_model.partial_fit(X_batch)

# 결과: 340,320 timepoints로 학습된 PCA model
```

**핵심**:
- **하나의 PCA 모델**을 모든 timepoints에 공유
- 시간에 걸쳐 **일관된 spatial patterns** 추출
- `IncrementalPCA`로 메모리 효율적 학습

#### 2.2 Inference Phase (PCA Transform)

```python
def reduce_features(fmri_seq):
    """
    Apply FITTED PCA to each timepoint

    Args:
        fmri_seq: (30, 96, 96, 96)
    Returns:
        features: (3000,)
    """
    reduced_timepoints = []

    for t in range(30):
        frame_flat = fmri_seq[t].reshape(-1)  # (884736,)

        # FITTED PCA 사용 (no re-fitting!)
        reduced = pca_model.transform(frame_flat.reshape(1, -1))  # (1, 100)

        reduced_timepoints.append(reduced.flatten())

    # Concatenate across time
    features = np.concatenate(reduced_timepoints)  # (3000,)

    return features
```

**중요**:
- Validation/Test 시 **절대 PCA를 다시 fit하지 않음**
- Training에서 학습된 PCA만 사용 (data leakage 방지)

---

### 3. SVR Training

#### 3.1 Feature Standardization

```python
# PCA 후 추가 standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)  # (11344, 3000)
```

**이유**: PCA는 variance를 기준으로 축소하지만 scale은 조정하지 않음

#### 3.2 Multi-output Regression

```python
# 각 emotion마다 독립적인 SVR 학습
for emotion_idx in range(7):
    y_train = Y_train[:, emotion_idx]  # (11344,)

    # Emotion-specific scaler
    scaler_e = StandardScaler()
    X_scaled = scaler_e.fit_transform(X_train_pca)

    # Emotion-specific SVR
    model_e = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model_e.fit(X_scaled, y_train)
```

**결과**:
- 7개 emotions → 7개 SVR models
- 7개 StandardScalers (각 emotion마다)

---

### 4. Key Implementation Details

#### 4.1 메모리 최적화

1. **IncrementalPCA**:
   - Batch-wise fitting (batch_size=1000)
   - 전체 데이터를 메모리에 올리지 않음

2. **Float32 사용**:
   ```python
   X_train = X_train.astype(np.float32)  # ~50% memory saving
   ```

3. **Checkpoint 저장**:
   - PCA model checkpoint
   - Training data checkpoint
   - Per-emotion model checkpoints

#### 4.2 병렬 처리

```python
# Parallel SVR training (7 emotions)
n_jobs = min(n_cpus, 7, 3)  # Max 3 workers for memory efficiency
results = Parallel(n_jobs=n_jobs)(
    delayed(train_single_emotion)(e, X_train, Y_train)
    for e in range(7)
)
```

---

## 📈 Feature Dimension Comparison

| Method | Temporal Info | Feature Dim | Memory |
|--------|--------------|-------------|---------|
| **Raw fMRI** | ✅ Full (30 TRs) | 26,542,080 | ~100 GB |
| **Time-Averaged PCA** | ❌ Removed (averaged) | 100 | ~4.5 MB |
| **Sequential PCA** | ✅ Preserved (30 vectors) | **3,000** | ~135 MB |

---

## 🎯 Expected Benefits & Comparison Purpose

### 장점

1. **Temporal Dynamics 유지**:
   - Time-averaging과 달리 시간에 걸친 변화 보존
   - Emotion의 dynamic pattern 포착 가능

2. **Consistent Feature Space**:
   - 모든 timepoints에 동일한 PCA 적용
   - 시간에 걸쳐 비교 가능한 representation

3. **해석 가능성**:
   - Principal components = 주요 brain activation patterns
   - Linear SVR with RBF kernel = 잘 이해된 방법론

### SwiFT-IO와의 비교 목적

1. **Baseline Performance**:
   - 전통적 ML 방법의 상한선 제시
   - SwiFT-IO의 성능 향상 정량화

2. **Feature Learning vs Feature Engineering**:
   - Sequential PCA: Hand-crafted dimensionality reduction
   - SwiFT-IO: End-to-end learned representations

3. **Temporal Modeling**:
   - Sequential PCA: Independent PCA per timepoint + concatenation
   - SwiFT-IO: Transformer attention across timepoints

4. **Scalability**:
   - Sequential PCA: 제한된 feature dimension (3000)
   - SwiFT-IO: Flexible hidden dimensions with self-attention

---

## 📂 Code Location

**Main implementation**:
- `src/baselines/svr_with_reduction.py`
  - Line 109-114: PCA initialization
  - Line 307-398: `fit_pca_on_train_data()` - PCA fitting
  - Line 239-305: `reduce_features()` - PCA transform per timepoint
  - Line 400-440: `_train_single_emotion()` - SVR training

**Training script**:
- `src/train_svr_with_reduction.py`
  - Usage: `--reduction_method pca --pca_components 100`

**Key functions**:
1. `fit_pca_on_train_data()`: Batch-wise IncrementalPCA fitting
2. `reduce_features()`: Transform each timepoint with fitted PCA
3. `prepare_data_from_dataloader()`: Convert sequences to (N, 3000) features

---

## 🔬 Academic Method Section (Draft)

### SVR with Sequential PCA Baseline

To establish a traditional machine learning baseline for comparison with SwiFT-IO, we implemented Support Vector Regression (SVR) with sequential PCA-based dimensionality reduction. Unlike time-averaging approaches that collapse temporal information, our method preserves temporal dynamics while achieving computational tractability. Specifically, we first fitted a single IncrementalPCA model (n_components=100) on all training timepoints (340,320 timepoints from 11,344 sequences), treating each timepoint as an independent sample in the 884,736-dimensional voxel space. During inference, each of the 30 timepoints in a sequence was independently transformed using this fitted PCA, and the resulting 100-dimensional representations were concatenated to form a 3,000-dimensional feature vector. We then trained separate SVR models (RBF kernel, C=1.0, ε=0.1) for each of the 7 emotions, with StandardScaler applied to the PCA-reduced features for each emotion independently. This approach provides a baseline that (1) maintains temporal information through sequential concatenation, (2) uses a consistent spatial feature space across time via shared PCA transformation, and (3) employs well-established linear methods for interpretability. By comparing SwiFT-IO's performance against this baseline, we can quantify the benefits of end-to-end learned representations and temporal attention mechanisms over traditional feature engineering with hand-crafted dimensionality reduction.

---

## 📊 Expected Results Pattern

Based on similar studies and our preliminary analysis:

- **Train R²**: ~0.40-0.50 (PCA explains limited variance)
- **Test R²**: ~0.20-0.30 (generalization gap expected)
- **Emotion-specific variance**: High variance across emotions (some emotions easier than others)

**Hypothesis**: SwiFT-IO should outperform this baseline by 20-40% in test R², demonstrating the value of learned temporal representations over hand-crafted features.

---

## 🔍 Implementation Notes

### Checkpoint Files

When running training, the following checkpoints are saved:

1. `pca_model_checkpoint.pkl`:
   - Fitted IncrementalPCA model
   - Variance explained ratio
   - Total frames used for fitting

2. `train_data_checkpoint.pkl`:
   - X_train: (11344, 3000) PCA-reduced features
   - Y_train: (11344, 7) emotion labels
   - Feature dimension

3. `svr_emotion_{e}_checkpoint.pkl` (×7):
   - SVR model for emotion e
   - StandardScaler for emotion e
   - Training metrics (MSE, MAE, R²)

### Memory Requirements

- **PCA fitting**: ~10 GB RAM (incremental, peak usage)
- **Training data**: ~135 MB (float32)
- **SVR training**: ~5-10 GB per emotion (RBF kernel cache)

**Recommendation**: Use SLURM with 32 GB RAM allocation

---

## 📚 References

This implementation follows standard practices in neuroimaging analysis:

- **PCA for fMRI**: Dimensionality reduction is a common preprocessing step
- **SVR for regression**: Well-established baseline for continuous predictions
- **IncrementalPCA**: Enables fitting on large datasets that don't fit in memory

**Key difference from typical approaches**:
- Most studies use time-averaging or ROI-averaging
- Our sequential PCA preserves temporal information while reducing spatial dimensions
- This provides a stronger baseline than simple averaging methods
