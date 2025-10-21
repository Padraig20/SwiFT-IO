# SVM vs SVR Baseline 비교 분석

**Date**: 2025-10-14
**Project**: SwiFT-IO Baseline Comparisons

---

## 1. train_svm_baseline.py vs train_svr_baseline.py 차이점

### 핵심 차이: Dimensionality Reduction (차원 축소)

#### SVR baseline (기존)
- 4D fMRI를 **그대로 flatten**만 함
- 입력: `(96, 96, 96, 30)` → flatten → `(96×96×96×30)` = **약 2,654,208 차원**
- 30개 timepoint를 모두 하나의 긴 벡터로 이어붙임
- 너무 많은 feature 때문에 학습이 매우 느리고 메모리도 많이 씀

#### SVM baseline (새로 만든 것)
- 4D fMRI를 **먼저 차원 축소**한 후 flatten
- 3가지 reduction 방법 제공:
  1. **time_avg**: `(96×96×96)` = 884,736 차원
  2. **pca**: `(30×100)` = 3,000 차원
  3. **roi**: `(30×512)` = 15,360 차원

### 코드 구조 비교

| 구분 | SVR (svr_baseline.py) | SVM (svm_baseline.py) |
|------|----------------------|----------------------|
| **클래스** | `SVRBaseline` | `SVMBaseline` |
| **feature 처리** | `flatten_sequence()` - 그냥 flatten | `reduce_features()` - 차원 축소 후 flatten |
| **입력 차원** | 2.6M (매우 큼) | 884K ~ 15K (훨씬 작음) |
| **추가 파라미터** | 없음 | `reduction_method`, `pca_components`, `roi_atlas` |
| **학습 속도** | 느림 | 빠름 |

---

## 2. time_avg가 "average"인 이유

### 코드 설명 (svm_baseline.py:238-240)

```python
elif self.reduction_method == 'time_avg':
    # Time-averaged: simply average across time
    avg_frame = fmri_seq.mean(axis=0)  # (96, 96, 96)
    features = avg_frame.flatten()  # (96*96*96,)
```

### 무슨 일이 일어나는가?

**원본 데이터:**
```
fmri_seq shape: (30, 96, 96, 96)
                 ↑
              30개 timepoint
```

예를 들어, 특정 voxel `(x=10, y=20, z=30)`의 값이:
```
timepoint 0:  0.5
timepoint 1:  0.7
timepoint 2:  0.3
timepoint 3:  0.6
...
timepoint 29: 0.4
```

**time_avg를 적용하면:**
```python
avg_frame[10, 20, 30] = mean([0.5, 0.7, 0.3, 0.6, ..., 0.4])
                      = 0.52  (예시)
```

즉, **각 voxel별로 30개 timepoint의 값을 평균냄**

**결과:**
- 30개 timepoint → 1개의 "평균된" brain image
- `(30, 96, 96, 96)` → `(96, 96, 96)`
- **시간 정보가 완전히 사라짐** - 이게 핵심!

---

## 3. 왜 이게 유용한가?

### SwiFT와의 비교

| 모델 | 시간 정보 | 예상 성능 |
|------|----------|----------|
| **SwiFT-IO** | 30 timepoints 전부 활용 (4D Swin Transformer) | **High** ⭐ |
| **SVR baseline** | 30 timepoints flatten (하지만 너무 많은 차원) | Medium |
| **SVM time_avg** | 30 timepoints 평균 → 시간 정보 없음 | **Low** |

### 논문에서 argument

- "SVM baseline (time_avg)은 시간 정보를 완전히 무시함 → 성능 낮음"
- "SwiFT-IO는 시간 정보를 활용함 → 성능 높음"
- **→ 따라서 temporal dynamics가 emotion prediction에 중요하다!**

---

## 4. 세 가지 Reduction Method 상세 비교

### Option 1: time_avg (RECOMMENDED)

**장점:**
- 가장 간단한 구현
- 명확한 해석: "시간 정보를 완전히 무시하면 어떻게 되는가?"
- SwiFT의 temporal modeling 능력을 부각시키기 좋음
- 학습 속도 빠름 (884K 차원)

**단점:**
- 시간 정보 손실

**Feature 차원:**
```
96 × 96 × 96 = 884,736 features
```

### Option 2: PCA

**장점:**
- 시간 정보 보존하면서 차원 축소
- 각 timepoint의 주요 spatial variance 보존
- 매우 작은 feature 차원 (3,000)

**단점:**
- PCA fitting에 시간 걸림
- 해석이 덜 명확함

**Feature 차원:**
```
30 timepoints × 100 PCA components = 3,000 features
```

### Option 3: ROI

**장점:**
- Neuroscience에서 흔히 쓰는 방법
- Anatomically meaningful
- 시간 정보 보존

**단점:**
- Atlas 의존적
- 구현이 복잡함

**Feature 차원:**
```
30 timepoints × 512 ROIs = 15,360 features
```

---

## 5. 실행 방법

### SLURM으로 실행
```bash
sbatch sample_scripts/run_svm_baseline.slurm
```

### 직접 실행 (time_avg)
```bash
python src/train_svm_baseline.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --reduction_method time_avg \
    --output_dir output/svm_baseline_time_avg
```

### 직접 실행 (PCA)
```bash
python src/train_svm_baseline.py \
    --reduction_method pca \
    --pca_components 100 \
    --output_dir output/svm_baseline_pca
```

### 직접 실행 (ROI)
```bash
python src/train_svm_baseline.py \
    --reduction_method roi \
    --output_dir output/svm_baseline_roi
```

---

## 6. 코드 파일 위치

- **SVM Baseline Class**: `src/baselines/svm_baseline.py`
- **Training Script**: `src/train_svm_baseline.py`
- **SLURM Script**: `sample_scripts/run_svm_baseline.slurm`
- **SVR Baseline Class**: `src/baselines/svr_baseline.py`
- **SVR Training Script**: `src/train_svr_baseline.py`

---

## 7. 요약

1. **SVR vs SVM 차이**: SVM은 차원 축소를 먼저 함 (학습 빠르고 메모리 적음)

2. **time_avg의 의미**:
   - 30개 timepoint를 **시간축으로 평균냄**
   - 각 voxel에서 30개 값 → 1개 평균값
   - 결과: 시간 정보 완전히 사라짐

3. **왜 유용한가**: SwiFT의 temporal modeling 능력을 부각시키는 baseline (일부러 시간 정보 제거)

---

## Tags
#baseline #SVM #SVR #dimensionality-reduction #temporal-modeling #SwiFT-IO
