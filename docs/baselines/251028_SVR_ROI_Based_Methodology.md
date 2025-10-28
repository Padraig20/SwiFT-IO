# SVR with ROI-based Features Baseline Methodology

**작성일**: 2025-10-28
**목적**: SwiFT-IO와 비교를 위한 ROI-based SVR baseline 설명

---

## 📊 Overview

ROI-based SVR baseline은 **해부학적으로 정의된 뇌 영역(Region of Interest)을 feature로 사용**하는 전통적인 neuroimaging 분석 접근법입니다. FreeSurfer의 자동 parcellation을 통해 추출된 95개의 공통 ROI를 사용하며, 각 ROI 내 voxel들의 평균 BOLD signal을 feature로 활용합니다.

**핵심 아이디어**:
- FreeSurfer 자동 parcellation으로 뇌를 95개 영역으로 분할
- 각 ROI의 시계열 데이터를 사용하여 temporal dynamics 유지
- 해부학적으로 해석 가능한 feature로 emotion prediction

---

## 🔧 Method Details

### 1. ROI Extraction (Preprocessing)

#### 1.1 FreeSurfer Parcellation

**Process**:
```
Raw fMRI: (96, 96, 96, 750 TRs)
    ↓ [FreeSurfer recon-all]
Anatomical parcellation
    ↓ [Apply to functional data]
ROI timeseries extraction
    ↓
CSV file: (750 TRs, 109 ROIs + 1 TR column)
```

**ROI Definition**:
- FreeSurfer의 Desikan-Killiany atlas 기반
- Subcortical structures (aseg.mgz)
- Cortical parcels (aparc.annot)

#### 1.2 Common ROIs Across Subjects

**문제**: 모든 subjects가 동일한 ROI set을 가지지 않음
- 총 CSV columns: 110 (TR 제외 시 109 ROIs)
- 일부 subjects에 누락된 ROIs 존재

**해결**: Common ROI 추출
- 677명의 모든 subjects에 공통적으로 존재하는 ROIs만 선택
- **95 common ROIs** 확정
- 14개 ROIs 제외 (일부 subjects에 누락)

```python
# Find common ROIs across all subjects
all_column_sets = []
for csv_file in all_subject_csvs:
    df = pd.read_csv(csv_file, nrows=1)
    cols = set(df.columns) - {'TR'}
    all_column_sets.append(cols)

# Intersection of all column sets
common_roi_columns = set.intersection(*all_column_sets)
# Result: 95 common ROIs
```

---

### 2. ROI Categories and Composition

#### 2.1 95 Common ROIs Breakdown

| 카테고리 | 개수 | 설명 |
|---------|------|------|
| **Ventricles & CSF** | 6 | 뇌실 및 뇌척수액 |
| **Corpus Callosum** | 4 | 뇌량 (좌우 반구 연결) |
| **Left Subcortical** | 11 | 좌측 피질하 구조 |
| **Right Subcortical** | 11 | 우측 피질하 구조 |
| **Left Cortex** | 30 | 좌측 대뇌피질 (Desikan-Killiany) |
| **Right Cortex** | 33 | 우측 대뇌피질 (Desikan-Killiany) |
| **합계** | **95** | |

#### 2.2 Detailed ROI List

**Ventricles & CSF (6 ROIs)**:
- 3rd-Ventricle
- 4th-Ventricle
- Left-Lateral-Ventricle
- Right-Lateral-Ventricle
- Brain-Stem
- CSF

**Corpus Callosum (4 ROIs)**:
- CC_Central
- CC_Mid_Anterior
- CC_Mid_Posterior
- CC_Posterior

**Subcortical Structures (11 pairs, Left/Right)**:
- **Accumbens-area** (측좌핵, nucleus accumbens): 보상 처리
- **Amygdala** (편도체): 정서 처리, 특히 공포/불안
- **Caudate** (미상핵): 운동 제어, 학습
- **Cerebellum-Cortex** (소뇌 피질)
- **Cerebellum-White-Matter** (소뇌 백질)
- **Cerebral-White-Matter** (대뇌 백질)
- **Hippocampus** (해마): 기억 형성
- **Pallidum** (담창구): 운동 제어
- **Putamen** (피각): 운동 제어, 학습
- **Thalamus-Proper*** (시상): 감각 정보 중계
- **VentralDC** (복측 diencephalon)

**Cortical Parcels - Left Hemisphere (30 ROIs)**:
- ctx-lh-bankssts (상측두구 뒤쪽)
- ctx-lh-caudalmiddlefrontal (중간 전두엽 뒤쪽)
- ctx-lh-cuneus (쐐기엽)
- ctx-lh-frontalpole (전두극)
- ctx-lh-fusiform (방추형회, 얼굴/물체 인식)
- ctx-lh-inferiorparietal (하두정엽)
- ctx-lh-inferiortemporal (하측두엽)
- ctx-lh-insula (섬엽, 내수용감각)
- ctx-lh-isthmuscingulate (협부대상피질)
- ctx-lh-lateraloccipital (외측후두엽)
- ctx-lh-lateralorbitofrontal (외측안와전두피질)
- ctx-lh-lingual (설상회)
- ctx-lh-medialorbitofrontal (내측안와전두피질)
- ctx-lh-middletemporal (중측두엽)
- ctx-lh-paracentral (중심방소엽)
- ctx-lh-parsopercularis (뚜껑부, 브로카 영역)
- ctx-lh-parsorbitalis (안와부)
- ctx-lh-parstriangularis (삼각부)
- ctx-lh-pericalcarine (거리구주변피질)
- ctx-lh-postcentral (중심뒤이랑, 감각피질)
- ctx-lh-posteriorcingulate (후대상피질)
- ctx-lh-precentral (중심앞이랑, 운동피질)
- ctx-lh-precuneus (쐐기앞소엽)
- ctx-lh-rostralmiddlefrontal (중간전두엽 앞쪽)
- ctx-lh-superiorfrontal (상전두엽)
- ctx-lh-superiorparietal (상두정엽)
- ctx-lh-superiortemporal (상측두엽, 청각처리)
- ctx-lh-supramarginal (연상회)
- ctx-lh-temporalpole (측두극)
- ctx-lh-transversetemporal (횡측두이랑, 1차 청각피질)

**Cortical Parcels - Right Hemisphere (33 ROIs)**:
- Left와 대부분 대칭 + 3개 추가:
  - ctx-rh-caudalanteriorcingulate (앞대상피질 뒤쪽)
  - ctx-rh-entorhinal (내후각피질, 기억)
  - ctx-rh-rostralanteriorcingulate (앞대상피질 앞쪽)

---

### 3. Feature Construction for SVR

#### 3.1 ROI Timeseries Loading

```python
def load_roi_timeseries(subject_id):
    """
    Load precomputed ROI timeseries from CSV

    Returns:
        roi_timeseries: (750 TRs, 95 common ROIs)
    """
    # CSV file path
    filepath = f"/scratch/HBN/9.2.movieDM_ROI_timeseries/
                 sub-{subject_id}_movieDM_roi_temporal_activity.csv"

    # Load CSV
    roi_data = pd.read_csv(filepath)

    # Select only 95 common ROI columns (consistent across all subjects)
    roi_timeseries = roi_data[common_roi_columns].values  # (750, 95)

    return roi_timeseries
```

#### 3.2 Sequence Extraction and Feature Generation

```python
def reduce_features(subject_id, start_frame, sequence_length=30):
    """
    Extract ROI features for a sequence

    Args:
        subject_id: Subject identifier
        start_frame: Starting TR index (0-719 for seq_len=30)
        sequence_length: Number of TRs (default: 30)

    Returns:
        features: (2850,) flattened ROI sequence
    """
    # Load full timeseries (750 TRs, 95 ROIs)
    roi_timeseries = load_roi_timeseries(subject_id)

    # Extract sequence slice
    end_frame = start_frame + sequence_length
    roi_sequence = roi_timeseries[start_frame:end_frame, :]  # (30, 95)

    # Flatten to single feature vector
    features = roi_sequence.flatten()  # (2850,)

    # Feature structure:
    # [TR0_ROI0, TR0_ROI1, ..., TR0_ROI94,
    #  TR1_ROI0, TR1_ROI1, ..., TR1_ROI94,
    #  ...
    #  TR29_ROI0, TR29_ROI1, ..., TR29_ROI94]

    return features
```

**Feature Dimension**:
- **30 timepoints × 95 ROIs = 2,850 features**

---

### 4. SVR Training

#### 4.1 Data Preparation

```python
# For each training sequence
X_train_list = []
Y_train_list = []

for batch in train_dataloader:
    subject_names = batch['subject_name']  # List of subject IDs
    start_frames = batch['TR']             # Starting TR indices
    targets = batch['target']              # (B, 30, 7) emotions

    for b in range(batch_size):
        # Extract ROI features
        features = reduce_features(
            subject_id=subject_names[b],
            start_frame=start_frames[b],
            sequence_length=30
        )  # (2850,)

        # Average targets over time
        targets_mean = targets[b].mean(axis=0)  # (7,) emotions

        X_train_list.append(features)
        Y_train_list.append(targets_mean)

X_train = np.stack(X_train_list)  # (11344, 2850)
Y_train = np.stack(Y_train_list)  # (11344, 7)
```

#### 4.2 Feature Standardization

```python
# Per-emotion standardization
for emotion_idx in range(7):
    y_train = Y_train[:, emotion_idx]

    # Standardize ROI features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # (11344, 2850)

    # Train SVR
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
```

---

### 5. Key Implementation Details

#### 5.1 Common ROI Cache

**목적**: 모든 subjects에 공통인 ROI 찾기 (한 번만 수행)

```python
# Cache file location
cache_file = '/scratch/HBN/9.2.movieDM_ROI_timeseries/_common_roi_cache.json'

# Cache structure
{
  "common_roi_columns": [...95 ROI names...],
  "num_subjects_checked": 677,
  "num_common_rois": 95
}
```

**이점**:
- 첫 실행 시 677개 CSV 스캔 (시간 소요)
- 이후 실행은 cache 파일 사용 (즉시)
- Data consistency 보장

#### 5.2 ROI Timeseries Cache

```python
# In-memory cache for loaded ROI timeseries
self.roi_cache = {}  # {subject_id: roi_timeseries}

# Cache on first load per subject
if subject_id not in self.roi_cache:
    roi_timeseries = load_from_csv(subject_id)
    self.roi_cache[subject_id] = roi_timeseries
```

**메모리 사용**:
- Per subject: 750 TRs × 95 ROIs × 8 bytes = ~570 KB
- 전체 training set (~350 subjects): ~200 MB

---

## 📈 Feature Dimension Comparison

| Method | Temporal Info | Feature Dim | Compression Ratio |
|--------|--------------|-------------|-------------------|
| **Raw fMRI** | ✅ Full (30 TRs) | 26,542,080 | 1× (baseline) |
| **Time-Avg PCA** | ❌ Averaged | 100 | 265,421× |
| **Sequential PCA** | ✅ Preserved (30 vectors) | 3,000 | 8,847× |
| **ROI-based** | ✅ Preserved (30 TRs) | **2,850** | **9,313×** |

**ROI vs Sequential PCA**:
- ROI: 2,850 features (30 × 95)
- PCA: 3,000 features (30 × 100)
- 비슷한 압축률이지만 **의미론적 차이** 존재

---

## 🎯 Expected Benefits & Comparison Purpose

### ROI-based 접근법의 장점

#### 1. **해부학적 해석 가능성**
- 각 feature가 **명확한 뇌 영역**에 대응
- "어떤 뇌 영역이 emotion prediction에 중요한가?" 직접 분석 가능
- Neuroscience literature와 직접 비교 가능

**예시**:
```python
# SVR weight 분석
emotion_weights = model.coef_  # (2850,)
roi_weights = emotion_weights.reshape(30, 95)  # (30 TRs, 95 ROIs)

# 가장 중요한 ROI 찾기
roi_importance = np.abs(roi_weights).mean(axis=0)  # Average over time
top_rois = np.argsort(roi_importance)[-10:]  # Top 10 ROIs

# Result: ["Left-Amygdala", "Right-Insula", ...]
# → Emotion processing과 관련된 뇌 영역!
```

#### 2. **Prior Knowledge 활용**
- FreeSurfer: 수천 개의 뇌 영상으로 검증된 parcellation
- Desikan-Killiany atlas: 신경과학 표준
- **Domain knowledge를 feature engineering에 통합**

#### 3. **노이즈 감소**
- ROI 내 voxel 평균 → signal-to-noise ratio 향상
- 개별 voxel의 noise가 averaging으로 상쇄

#### 4. **계산 효율성**
- 884,736 voxels → 95 ROIs (9,313배 압축)
- PCA fitting 불필요 (사전 계산된 ROI timeseries 사용)
- 빠른 학습 및 inference

### ROI-based 접근법의 단점

#### 1. **정보 손실**
- ROI 내부의 spatial heterogeneity 무시
- 예: Amygdala 내에서도 subregion별로 다른 기능
- 884,736 voxels → 95 ROIs: 큰 정보 손실

#### 2. **Parcellation 의존성**
- FreeSurfer 결과의 정확도에 의존
- Atlas 선택에 따라 결과 변동 가능
- Individual variability 고려 부족

#### 3. **Rigid Feature Space**
- 사전 정의된 ROI만 사용
- Data-driven optimization 불가능
- 새로운 brain pattern 발견 어려움

### SwiFT-IO와의 비교 목적

#### 1. **Feature Engineering vs Feature Learning**

| Aspect | ROI-based SVR | SwiFT-IO |
|--------|---------------|----------|
| **Feature type** | Hand-crafted (anatomical ROIs) | Learned (self-attention) |
| **Prior knowledge** | Strong (FreeSurfer parcellation) | Weak (only architecture) |
| **Flexibility** | Fixed ROI structure | Adaptive representations |
| **Interpretability** | High (anatomical regions) | Low (latent features) |

#### 2. **Temporal Modeling**

```
ROI-based SVR:
    ROI timeseries (30, 95) → Flatten (2850,) → SVR
    ↑ Simple concatenation, no temporal interaction modeling

SwiFT-IO:
    Voxel timeseries → Transformer → Self-attention across time
    ↑ Explicit temporal dependency modeling
```

#### 3. **Spatial Representation**

```
ROI-based:
    95 anatomical regions (fixed, coarse-grained)

SwiFT-IO:
    Patch-based encoding → learned spatial features (fine-grained)
```

#### 4. **Expected Performance Pattern**

**가설**:
- ROI-based SVR이 **baseline으로 합리적인 성능** 제공
  - Prior knowledge 활용으로 안정적인 성능
  - 특히 emotion-related ROIs (Amygdala, Insula 등)에서 강점

- SwiFT-IO가 **ROI-based를 능가**할 것으로 예상
  - End-to-end learning으로 ROI 경계 너머의 정보 활용
  - Temporal attention으로 dynamic emotion 변화 포착
  - Fine-grained spatial features로 subtle pattern 감지

**정량적 예상**:
- ROI-based SVR test R²: 0.25-0.35
- SwiFT-IO test R²: 0.40-0.55 (15-50% 향상)

---

## 🔬 Academic Method Section (Draft)

### SVR with ROI-based Features Baseline

To establish an anatomically-grounded baseline for comparison with SwiFT-IO, we implemented Support Vector Regression (SVR) using region-of-interest (ROI) features derived from FreeSurfer's automated parcellation. We first applied FreeSurfer's recon-all pipeline to extract 95 common ROIs across all 677 subjects, comprising bilateral subcortical structures (e.g., amygdala, hippocampus, nucleus accumbens), Desikan-Killiany cortical parcels, and white matter regions. For each 30-TR fMRI sequence, we extracted the corresponding ROI timeseries (30 timepoints × 95 ROIs) and flattened them into 2,850-dimensional feature vectors, preserving temporal dynamics while leveraging anatomical prior knowledge. We trained separate SVR models (RBF kernel, C=1.0, ε=0.1) for each of the 7 emotions, with StandardScaler applied to ROI features. This approach provides a neuroscientifically interpretable baseline that (1) uses well-established anatomical parcellation for feature engineering, (2) maintains temporal information through sequential ROI measurements, (3) reduces dimensionality by 9,313-fold compared to raw voxel data while retaining biologically meaningful spatial structure, and (4) allows direct interpretation of which brain regions contribute to emotion prediction. By comparing SwiFT-IO's performance against this ROI-based baseline, we can quantify the benefits of end-to-end learned representations over traditional hand-crafted anatomical features, and assess whether learned fine-grained voxel-level patterns provide advantages over coarse-grained ROI averaging.

---

## 📊 ROI-based Features: Neuroscientific Relevance

### Emotion Processing 관련 주요 ROIs

#### 1. **Limbic System (변연계)**
- **Amygdala (편도체)**: 공포, 불안, 정서 학습
- **Hippocampus (해마)**: 감정적 기억 형성
- **Anterior Cingulate Cortex**: 정서 조절, 충돌 감지

#### 2. **Reward System (보상계)**
- **Nucleus Accumbens (측좌핵)**: 보상, 동기부여
- **Ventral Tegmental Area** (VTA, 부분적으로 Brain-Stem에 포함)

#### 3. **Social & Interoceptive Processing**
- **Insula (섬엽)**: 내수용감각, 정서적 인식
- **Superior Temporal Sulcus**: 사회적 인지

#### 4. **Cognitive Control**
- **Prefrontal Cortex** (rostralmiddlefrontal, superiorfrontal): 정서 조절
- **Posterior Cingulate**: Default mode network

### Expected ROI Contributions

**가설**:
1. **Subcortical structures** (특히 Amygdala, Nucleus Accumbens)가 높은 가중치
2. **Temporal cortex** (얼굴 표정 처리)
3. **Prefrontal regions** (정서 조절)
4. **Insula** (내적 감정 상태 인식)

---

## 📂 Code Location

**Main implementation**:
- `src/baselines/svr_with_reduction.py`
  - Line 115-122: ROI method initialization
  - Line 131-191: `find_common_roi_columns()` - Find 95 common ROIs
  - Line 193-237: `load_roi_timeseries()` - Load ROI data from CSV
  - Line 254-269: `reduce_features()` - Extract ROI sequence features

**Data location**:
- ROI timeseries CSVs: `/scratch/HBN/9.2.movieDM_ROI_timeseries/`
  - Format: `sub-{ID}_movieDM_roi_temporal_activity.csv`
  - Shape per file: (750 TRs, 110 columns)
- Common ROI cache: `/scratch/HBN/9.2.movieDM_ROI_timeseries/_common_roi_cache.json`

**Training script**:
- `src/train_svr_with_reduction.py`
  - Usage: `--reduction_method roi --roi_atlas freesurfer`

---

## 🔍 Implementation Notes

### File Structure

```
/scratch/HBN/9.2.movieDM_ROI_timeseries/
├── _common_roi_cache.json              # 95 common ROI list
├── sub-NDARAA947ZG5_movieDM_roi_temporal_activity.csv
├── sub-NDARAB458VK9_movieDM_roi_temporal_activity.csv
└── ... (677 subjects total)
```

### CSV Format

```csv
TR,Left-Cerebral-White-Matter,Left-Lateral-Ventricle,...,ctx-rh-transversetemporal
0,1312.11,1472.86,...,1234.56
1,1310.92,1469.30,...,1235.67
...
749,1315.23,1471.45,...,1236.78
```

- **Rows**: 750 TRs (full movie duration)
- **Columns**: 1 TR + 109 ROI values
- **Values**: Mean BOLD signal within each ROI

### Memory Requirements

- **ROI timeseries loading**: ~200 MB (cached in memory)
- **Training data**: (11344, 2850) × 4 bytes = ~130 MB (float32)
- **SVR training**: ~5-10 GB per emotion (RBF kernel cache)

**총 메모리**: ~16 GB 권장

---

## 📚 References

### FreeSurfer & Desikan-Killiany Atlas

- **Desikan et al. (2006)**: "An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest." NeuroImage.
- **FreeSurfer**: https://surfer.nmr.mgh.harvard.edu/

### ROI-based fMRI Analysis

- ROI averaging은 neuroimaging에서 표준 전처리 방법
- Reduces noise, increases statistical power
- Allows anatomically-specific hypothesis testing

### Comparison with Data-Driven Methods

- **Knowledge-driven (ROI)**: Uses neuroscientific prior knowledge
- **Data-driven (PCA, ICA)**: Discovers patterns from data
- **Hybrid (SwiFT-IO)**: Learns representations with architectural priors

---

## 🎯 Key Takeaways

1. **95 common ROIs** extracted from FreeSurfer parcellation
2. **2,850-dimensional features** (30 TRs × 95 ROIs)
3. **Anatomically interpretable** - each feature = specific brain region
4. **Neuroscientific validity** - uses established parcellation atlas
5. **Efficient baseline** - prior knowledge reduces need for large data
6. **Comparison value** - quantifies benefit of learned representations

이 baseline을 통해 **hand-crafted anatomical features vs learned voxel-level representations**의 성능 차이를 정량화할 수 있습니다.
