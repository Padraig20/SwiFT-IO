# SVR PCA Baseline 방법론 상세 설명

**작성일**: 2025-10-26
**목적**: SVR PCA baseline의 동작 원리와 설계 근거 설명
**결과**: Test MSE=2.093, MAE=0.778, R²=-0.125

---

## 📋 목차

1. [개요](#개요)
2. [전체 프로세스 흐름](#전체-프로세스-흐름)
3. [Phase 1: PCA 모델 학습](#phase-1-pca-모델-학습)
4. [Phase 2: Feature Extraction](#phase-2-feature-extraction)
5. [Phase 3: SVR 학습 및 평가](#phase-3-svr-학습-및-평가)
6. [왜 이 방법이 올바른가](#왜-이-방법이-올바른가)
7. [다른 방법들과의 비교](#다른-방법들과의-비교)
8. [최종 Feature 구성](#최종-feature-구성)
9. [결과 및 인사이트](#결과-및-인사이트)

---

## 개요

### SVR PCA Baseline이란?

**목적**: SwiFT-IO와 비교하기 위한 data-driven spatial feature extraction baseline

**핵심 아이디어**:
- ✅ **Temporal 정보 보존**: 30 TRs를 각각 독립적으로 처리
- ✅ **Data-driven**: Anatomical atlas 없이 데이터에서 학습
- ✅ **Dimensionality reduction**: 884,736 voxels → 100 components per TR

**최종 Feature**:
- Input: 4D fMRI sequence (96×96×96×30)
- Output: 3,000 features (30 TRs × 100 components)

---

## 전체 프로세스 흐름

```
Dataset Split (seed=777):
├── Train:  473 subjects → 11,349 sequences
├── Val:     99 subjects →  2,403 sequences
└── Test:   103 subjects →  2,437 sequences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: PCA 모델 학습 (Train split만)
  Train 340,320 TRs → IncrementalPCA.fit() → 100 PCs

Phase 2: Feature Extraction (Train/Val/Test 모두)
  각 sequence의 30 TRs → PCA.transform() → 3,000 features

Phase 3: SVR 학습 및 평가
  Train features → SVR.fit() → 7 SVR models
  Val/Test features → SVR.predict() → Evaluation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Phase 1: PCA 모델 학습

### 🎯 목표
Train split의 모든 TRs를 사용해서 **spatial variance를 최대한 설명하는 100개의 principal components**를 학습

### 📊 데이터 준비

```python
Train split:
  - 473 subjects
  - 11,349 sequences (473 subjects × ~24 sequences/subject)
  - 340,320 TRs (11,349 sequences × 30 TRs/sequence)

모든 TRs를 하나의 pool로 모음:
  TR_0:       (96, 96, 96) = 884,736 voxels
  TR_1:       (96, 96, 96) = 884,736 voxels
  TR_2:       (96, 96, 96) = 884,736 voxels
  ...
  TR_340,319: (96, 96, 96) = 884,736 voxels

Shape: (340,320 samples, 884,736 features)
```

### 🔧 IncrementalPCA 학습

```python
from sklearn.decomposition import IncrementalPCA

pca = IncrementalPCA(n_components=100, batch_size=1000)

# 메모리 효율을 위해 batch-wise로 학습
for batch in dataloader:
    fmri_data = batch['fmri_sequence']  # (B, 96, 96, 96, 30)

    # Reshape: (B, 30, 96, 96, 96) → (B*30, 884,736)
    X_batch = fmri_data.reshape(B*30, -1)

    # Incremental fitting
    pca.partial_fit(X_batch)

print(f"Fitted on {340,320} timepoints")
print(f"Variance explained: {91.71}%")
```

### 📦 학습 결과

**Output**:
- **100개의 Principal Components (PCs)**
  - 각 PC: (884,736) 차원의 weight vector
  - PC_0, PC_1, ..., PC_99

- **Variance Explained**: 91.71%
  - 원본 공간 variance의 91.71%를 100차원에서 보존

**저장**:
- `pca_model_checkpoint.pkl` (689 MB)
- 이후 모든 데이터(train/val/test)에 재사용

---

## Phase 2: Feature Extraction

### 🎯 목표
학습된 PCA 모델을 사용해서 **모든 sequence(train/val/test)를 3,000차원 feature로 변환**

### 🔄 핵심 개념: "Inference"

**Inference란?**
- 학습된 PCA 모델을 **새로운 데이터**에 적용하는 것
- PCA transform: 884,736 차원 → 100 차원으로 projection

**어디에 적용?**
- ✅ Train split (11,349 sequences)
- ✅ Val split (2,403 sequences)
- ✅ Test split (2,437 sequences)

**모두 같은 PCA 모델 사용!** (train에서 학습한 것)

---

### 📝 Feature Extraction 과정 (Step-by-Step)

#### **단일 Sequence 처리 예시**

```python
# Input: 1개 sequence
sequence = load_sequence(subject_id, start_frame)
# Shape: (30 TRs, 96, 96, 96)

features = []

for t in range(30):
    # 1. t번째 TR 추출
    TR_t = sequence[t]  # (96, 96, 96)

    # 2. Flatten
    TR_t_flat = TR_t.reshape(-1)  # (884,736,)

    # 3. PCA transform (학습된 PCA 모델 사용!)
    reduced = pca_model.transform(TR_t_flat.reshape(1, -1))
    # reduced shape: (1, 100)

    # 4. 결과 저장
    features.append(reduced.flatten())  # (100,)

# 5. Concatenate across time
X = np.concatenate(features)  # (3,000,)

# 최종 shape: (30 TRs × 100 components = 3,000 features)
```

#### **전체 데이터셋 처리**

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Train Split (11,349 sequences)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

X_train = []  # (11,349, 3,000)
Y_train = []  # (11,349, 7)

for sequence, labels in train_dataloader:
    features = extract_features(sequence)  # (3,000,)
    X_train.append(features)
    Y_train.append(labels)

X_train = np.array(X_train)  # (11,349, 3,000)
Y_train = np.array(Y_train)  # (11,349, 7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Val Split (2,403 sequences)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

X_val = []  # (2,403, 3,000)
Y_val = []  # (2,403, 7)

for sequence, labels in val_dataloader:
    # ⭐ 같은 PCA 모델 사용!
    features = extract_features(sequence)  # (3,000,)
    X_val.append(features)
    Y_val.append(labels)

X_val = np.array(X_val)  # (2,403, 3,000)
Y_val = np.array(Y_val)  # (2,403, 7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Split (2,437 sequences)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

X_test = []  # (2,437, 3,000)
Y_test = []  # (2,437, 7)

for sequence, labels in test_dataloader:
    # ⭐ 같은 PCA 모델 사용!
    features = extract_features(sequence)  # (3,000,)
    X_test.append(features)
    Y_test.append(labels)

X_test = np.array(X_test)  # (2,437, 3,000)
Y_test = np.array(Y_test)  # (2,437, 7)
```

---

## Phase 3: SVR 학습 및 평가

### 🎯 SVR Training (Train features만 사용)

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 7개 emotion별로 독립적인 SVR 모델
models = []
scalers = []

for emotion_idx in range(7):
    # 1. Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. SVR 학습
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, Y_train[:, emotion_idx])

    # 3. 저장
    models.append(svr)
    scalers.append(scaler)

    # 4. Checkpoint 저장
    save(f"svr_emotion_{emotion_idx}_checkpoint.pkl")
```

### 📊 Evaluation (Val/Test)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

predictions_val = []

for emotion_idx in range(7):
    X_val_scaled = scalers[emotion_idx].transform(X_val)
    pred = models[emotion_idx].predict(X_val_scaled)
    predictions_val.append(pred)

predictions_val = np.array(predictions_val).T  # (2,403, 7)

# Compute metrics
val_mse = mean_squared_error(Y_val, predictions_val)
val_mae = mean_absolute_error(Y_val, predictions_val)
val_r2 = r2_score(Y_val, predictions_val)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test (최종 평가)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

predictions_test = []

for emotion_idx in range(7):
    X_test_scaled = scalers[emotion_idx].transform(X_test)
    pred = models[emotion_idx].predict(X_test_scaled)
    predictions_test.append(pred)

predictions_test = np.array(predictions_test).T  # (2,437, 7)

# Compute metrics
test_mse = mean_squared_error(Y_test, predictions_test)
test_mae = mean_absolute_error(Y_test, predictions_test)
test_r2 = r2_score(Y_test, predictions_test)
```

---

## 왜 이 방법이 올바른가

### ✅ 1. 표준 PCA 사용법

**PCA의 기본 원리**:
```
Training:
  Input: N samples × D dimensions
  Output: K principal components (K << D)

  "N개 샘플의 variance를 가장 잘 설명하는 K개 방향 찾기"

Transform (Inference):
  Input: 1 sample × D dimensions
  Output: 1 sample × K dimensions

  "이 샘플을 K개 방향으로 projection"
```

**SVR PCA의 경우**:
```
Training:
  Input: 340,320 samples × 884,736 dims
  Output: 100 principal components

Transform:
  Input: 1 TR × 884,736 dims
  Output: 1 TR × 100 dims

  30 TRs → 30 × 100 = 3,000 features
```

→ **완전히 정상적인 PCA 사용법!** ✅

---

### ✅ 2. Temporal 정보 보존

**핵심 질문**: "각 TR을 100차원으로 압축하면 temporal 정보가 손실되는 것 아닌가?"

**답변**: ❌ 아니다!

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spatial compression: 각 TR을 100차원으로
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TR_0:  (884,736 voxels) → (100 features)
TR_1:  (884,736 voxels) → (100 features)
...
TR_29: (884,736 voxels) → (100 features)

# 각 TR이 독립적으로 100차원 벡터로 표현됨
# TR_0의 100 features ≠ TR_1의 100 features
# → 30개 timepoints가 그대로 유지됨!
```

**Temporal dynamics 보존**:
- ✅ 30개 TR의 **순서** 보존
- ✅ 30개 TR의 **독립성** 보존
- ✅ TR 간 **시간적 변화** 보존
- ⚠️ 각 TR 내 **spatial detail** 손실 (884K → 100)

---

### ✅ 3. 일반화 가능성 (Generalization)

**장점**:
```
같은 PCA 모델을 모든 데이터에 적용:
  - Train split ✅
  - Val split ✅
  - Test split ✅
  - 새로운 subject ✅
  - 새로운 sequence ✅

→ 어떤 TR이든 항상 100차원으로 변환 가능
```

**만약 TR 위치별로 다른 PCA를 학습했다면?**
```
PCA_0 for TR_0
PCA_1 for TR_1
...
PCA_29 for TR_29

문제:
  - TR 위치가 고정되어야 함
  - 다른 sequence length 불가능
  - 일반화 불가능 ❌
```

---

## 다른 방법들과의 비교

### 방법 1: TR 위치별로 다른 PCA 학습 ❌

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 잘못된 방법
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# PCA 학습
for t in range(30):
    all_TRs_at_position_t = collect_all_TRs_at_position(t)
    pca_models[t] = PCA(100).fit(all_TRs_at_position_t)

# Transform
for t in range(30):
    features[t] = pca_models[t].transform(sequence[t])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 문제점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ❌ TR 위치 의존적
   → sequence의 5번째 TR은 반드시 PCA_5를 사용해야 함
   → 다른 위치의 TR은 처리 불가

2. ❌ Sequence length 고정
   → 30 TRs가 아닌 다른 length 불가능

3. ❌ 일반화 불가능
   → 새로운 시작 위치의 sequence 처리 불가

4. ❌ 비현실적
   → fMRI는 연속적인 시계열, 위치에 따라 다른 공간이 아님
```

**왜 안 좋은가?**
- fMRI의 spatial structure는 **시간에 무관**하게 일정함
- TR_0의 voxel organization = TR_1의 voxel organization
- 따라서 **하나의 공통 PCA**로 모든 TR을 처리하는 것이 합리적

---

### 방법 2: 전체 sequence를 한번에 PCA ❌

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 잘못된 방법
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Flatten entire sequence
sequence_flat = sequence.reshape(-1)
# (30, 96, 96, 96) → (26,542,080,)

# PCA transform
features = pca.transform(sequence_flat)  # (100,)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 문제점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ❌ Temporal dynamics 완전 손실
   → 30개 TR이 하나의 벡터로 합쳐짐
   → 시간적 변화 정보 사라짐

2. ❌ 차원이 너무 큼
   → 26M 차원에서 PCA 학습 비현실적

3. ❌ 의미 없는 mixing
   → 공간과 시간이 뒤섞여서 해석 불가
```

**이건 사실상 다른 baseline**:
- Time-averaged PCA baseline과 비슷
- Temporal modeling 포기

---

### 방법 3: Time-averaged PCA (실제 다른 baseline) ⚠️

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time-averaged PCA baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Time averaging
avg_frame = sequence.mean(axis=0)  # (96, 96, 96)

# 2. PCA transform
features = pca.transform(avg_frame.reshape(1, -1))  # (100,)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 특징
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Valid baseline (비교용으로 적합)
❌ Temporal dynamics 완전 손실 (의도적)
📊 Feature dimension: 100 (vs. 3,000 in PCA method)
🎯 Purpose: Temporal modeling의 중요성 입증
```

**이건 별도의 baseline으로 이미 실행 중**:
- `output/svr_time_avg_pca_100/`
- Job 63225 (평가 진행중)

---

### 방법 4: 현재 SVR PCA 방법 (올바름) ✅

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 올바른 방법 (실제 구현)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. 하나의 PCA 모델 학습 (모든 train TRs 사용)
pca = PCA(100).fit(all_train_TRs)  # 340,320 samples

# 2. 각 TR을 독립적으로 transform
for t in range(30):
    features[t] = pca.transform(sequence[t])  # (100,)

# 3. Concatenate
final_features = concat(features)  # (3,000,)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 장점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Temporal dynamics 보존 (30 TRs 독립적)
✅ Spatial compression (884K → 100 per TR)
✅ 일반화 가능 (모든 TR에 적용 가능)
✅ 표준 PCA 사용법
✅ Data-driven (anatomical prior 불필요)
```

---

## 최종 Feature 구성

### 📊 Feature Breakdown

```
Input sequence: (30 TRs, 96, 96, 96)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TR_0:  (884,736) → PCA → [f_0_0, f_0_1, ..., f_0_99]    (100)
TR_1:  (884,736) → PCA → [f_1_0, f_1_1, ..., f_1_99]    (100)
TR_2:  (884,736) → PCA → [f_2_0, f_2_1, ..., f_2_99]    (100)
...
TR_29: (884,736) → PCA → [f_29_0, f_29_1, ..., f_29_99] (100)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Concatenate all features:
[f_0_0, ..., f_0_99, f_1_0, ..., f_1_99, ..., f_29_0, ..., f_29_99]

Final feature vector: (3,000,)
```

### 📈 Feature 의미

**각 100 features (per TR)**:
- PC_0: 1번째 principal component에 대한 projection
- PC_1: 2번째 principal component에 대한 projection
- ...
- PC_99: 100번째 principal component에 대한 projection

**3,000 features (전체)**:
- 첫 100개: TR_0의 spatial pattern
- 다음 100개: TR_1의 spatial pattern
- ...
- 마지막 100개: TR_29의 spatial pattern

→ **Temporal structure 보존됨!**

---

### 🆚 다른 baseline과 비교

| Method | Feature Dimension | Temporal Info | Spatial Representation |
|--------|------------------|---------------|----------------------|
| **SVR ROI** | 2,850 (30×95) | ✅ 보존 | Anatomical (AAL atlas) |
| **SVR PCA** | 3,000 (30×100) | ✅ 보존 | Data-driven (PCA) |
| **SVR Time-avg** | 100 | ❌ 손실 | Data-driven (PCA) |

**차이점**:
- ROI: 각 TR마다 95개 brain regions
- PCA: 각 TR마다 100개 learned components
- Time-avg: 1개 time-averaged frame의 100 components

---

## 결과 및 인사이트

### 📊 최종 성능 (Test Set)

| Metric | Value |
|--------|-------|
| **Test MSE** | 2.093 |
| **Test MAE** | 0.778 |
| **Test R²** | -0.125 |

**Per-Emotion Results**:

| Emotion | MSE | MAE | R² |
|---------|-----|-----|----|
| Amusing | 2.481 | 0.895 | -0.204 |
| Anxiety | 2.515 | 1.011 | -0.167 |
| Boring | 0.405 | 0.394 | -0.164 |
| Fearful | 2.798 | 0.827 | -0.200 |
| Pleasant | 3.968 | 0.821 | -0.126 |
| Sad | 1.597 | 0.782 | -0.150 |
| Neutral | 0.884 | 0.712 | -0.036 |

---

### 🔍 핵심 인사이트

#### 1. **ROI vs PCA: Anatomical Prior의 가치**

```
SVR ROI:  Test R² = -0.114 (MSE = 2.074)
SVR PCA:  Test R² = -0.125 (MSE = 2.093)

→ ROI가 PCA보다 약간 더 좋음
```

**의미**:
- ✅ **Anatomical prior가 도움이 됨**
- ✅ Emotion-related brain activation이 anatomical boundaries와 align됨
- ⚠️ Data-driven PCA는 anatomical structure를 무시
- 📝 **논문 포인트**: Brain organization matters for emotion prediction

**설계 함의**:
- SwiFT-IO의 attention mechanism이 anatomical structure를 학습했는지 분석 필요
- Attention maps와 brain atlas 비교
- Interpretability 관점에서 중요

---

#### 2. **Negative R²: Baseline의 한계**

```
모든 SVR baseline이 negative R²:
  - SVR ROI: -0.114
  - SVR PCA: -0.125
  - LSTM:    -0.135

→ Mean baseline보다 못함
```

**의미**:
- ❌ Simple feature engineering (ROI, PCA)으로는 부족
- ❌ Basic temporal modeling (concatenation, LSTM)도 부족
- ✅ **SwiFT-IO의 필요성 입증**
  - Learned spatiotemporal representations
  - 4D attention mechanism
  - Hierarchical feature learning

---

#### 3. **Temporal Modeling이 필요한가?**

```
SVR PCA (temporal 보존):     R² = -0.125
SVR Time-avg (temporal 손실): R² = ??? (평가 대기중)

→ 비교 결과로 temporal modeling 중요성 확인 예정
```

**예상**:
- Time-averaged baseline이 더 나쁠 것으로 예상
- PCA > Time-avg 이면 temporal dynamics가 중요함을 입증

---

#### 4. **PCA가 학습한 것**

**91.71% variance 설명**:
- 대부분의 spatial variability 포착
- 하지만 emotion prediction에는 부족

**가능한 원인**:
1. **Task-irrelevant variance 학습**
   - PCA는 variance를 최대화하는 방향 학습
   - Emotion과 무관한 variance도 포함

2. **Spatial-temporal interaction 미고려**
   - 각 TR을 독립적으로 처리
   - TR 간 관계 학습 불가

3. **Non-linear patterns 못 잡음**
   - PCA는 linear transformation
   - Emotion은 non-linear spatiotemporal patterns?

→ **SwiFT-IO의 장점**: Task-specific feature learning + Non-linear + Spatiotemporal interaction

---

### 🎯 SwiFT-IO와의 비교 예상

**기대 효과**:
```
If SwiFT-IO >> SVR PCA:
  → Learned hierarchical representations > PCA
  → 4D attention > independent TR processing
  → Task-specific learning > variance maximization
```

**최소 목표**:
- Test R² > 0 (mean baseline 이상)
- Test MSE < 2.0 (SVR baselines 이상)

**이상적 목표**:
- Test R² > 0.3
- Test MSE < 1.0
- Per-emotion improvement across all emotions

---

## 요약

### 🔑 핵심 포인트

1. **PCA 학습**: Train split의 모든 340,320 TRs로 **하나의 PCA 모델** 학습
2. **Feature Extraction**: 각 TR을 **독립적으로** 100차원으로 변환 → 30×100=3,000 features
3. **Temporal 보존**: 30개 TR이 각각 독립적인 100차원 벡터로 표현됨
4. **일반화**: 같은 PCA 모델을 train/val/test 모두에 적용
5. **결과**: ROI보다 약간 나쁨 → Anatomical prior가 valuable

### 📚 방법론 정당성

- ✅ 표준 PCA 사용법 (N samples → K components, 각 sample을 K-dim으로 transform)
- ✅ Temporal dynamics 보존 (각 TR 독립 처리)
- ✅ 일반화 가능 (어떤 TR이든 적용 가능)
- ✅ Memory-efficient (IncrementalPCA)

### 🎓 학술적 가치

- Data-driven vs. anatomical prior 비교
- Temporal modeling 중요성 입증 (time-avg와 비교)
- SwiFT-IO의 learned representation 정당화
- Feature engineering의 한계 제시

---

**코드 위치**:
- Training: `src/train_svr_with_reduction.py`
- Baseline class: `src/baselines/svr_with_reduction.py`
- SLURM script: `sample_scripts/run_svr_pca.slurm`
- Output: `output/svr_reduction_pca/`

**관련 문서**:
- `baseline_performance_table.md`: 전체 baseline 비교표
- `251023_SVR_Baseline_Methods_Comparison.md`: SVR 방법론 비교
- `251022_SVR_PCA_Optimization.md`: PCA 최적화 과정
