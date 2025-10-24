# LSTM Baseline 오류 분석 및 해결 전략

**날짜**: 2025-10-24
**분석자**: Claude Code
**목적**: LSTM baseline 반복 실패 원인 파악 및 해결 방안 제시

---

## 📋 Executive Summary

### 문제 요약
LSTM baseline 학습이 **metadata 로딩 단계에서 반복적으로 실패**하고 있습니다.
- **총 시도 횟수**: 33회 이상 (logs 파일 기준)
- **주요 에러**: `ValueError: No matching SUBJECT_IDs found in metadata`
- **성공 케이스**: 1회 (Job 62789) - 학습 완료했으나 test 단계에서 다른 에러 발생

---

## 🔍 문제 분석

### 1. 주요 에러 패턴

#### Error Type 1: Metadata Matching 실패 (가장 빈번)
```python
ValueError: No matching SUBJECT_IDs found in metadata.
```

**발생 위치**: `src/module/utils/data_module.py:48`
```python
def determine_stratified_split(self, subject_dict, seed, stratified_params, metadata_csv_path, ...):
    df = pd.read_csv(metadata_csv_path)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)
    subject_ids = set(str(sid) for sid in subject_dict)

    # Debug output shows:
    # [DEBUG] Sample metadata SUBJECT_IDs: ['sub-NDARYK164AEJ', 'sub-NDARFX083RCV', ...]
    # [DEBUG] Sample data subject_ids: ['sub-NDARNL587LVL', 'sub-NDARFB506UJ3', ...]

    df = df[df["SUBJECT_ID"].isin(subject_ids)].copy()

    if df.empty:  # ← 여기서 실패
        raise ValueError("No matching SUBJECT_IDs found in metadata.")
```

**문제점**:
- `subject_dict`의 subject ID와 `metadata CSV`의 SUBJECT_ID가 매칭되지 않음
- Debug 출력 보면 둘 다 존재하는데 매칭 실패

**가능한 원인**:
1. **Format 불일치**: `subject_dict` keys와 CSV SUBJECT_ID의 형식이 다름
2. **Data path 문제**: LSTM이 다른 dataset을 로드하고 있을 수 있음
3. **Metadata path 문제**: 잘못된 metadata CSV 참조

---

#### Error Type 2: Test Step 실패 (1회 발생)
```python
RuntimeError: a Tensor with 210 elements cannot be converted to Scalar
```

**발생 위치**: `src/module/pl_classifier.py:554`
```python
def test_step(self, batch, batch_idx):
    ...
    output = [(logit.cpu().detach(), targets.cpu().item())
              for logit, targets in zip(logits, target)]
    # ↑ targets.cpu().item() 에서 실패
```

**문제점**:
- `targets`가 scalar가 아니라 210개 원소를 가진 tensor
- `.item()`은 single element tensor에만 사용 가능
- LSTM의 출력 shape과 expected shape 불일치

---

### 2. 실패한 Job들 상세

| Job ID | 날짜 | 에러 타입 | 위치 |
|--------|------|----------|------|
| 62560-62565 | Oct 14 | Metadata matching | data_module.py:48 |
| 62587-62595 | Oct 15 | Metadata matching | data_module.py:48 |
| 62601 | Oct 15 | Metadata matching | data_module.py:48 |
| 62737-62780 | Oct 20 | Metadata matching | data_module.py:48 |
| 62781-62788 | Oct 20 | Metadata matching | data_module.py:48 |
| **62789** | **Oct 20** | **Test step error** | **pl_classifier.py:554** |

**패턴**:
- 32/33 jobs가 **동일한 metadata 에러**로 실패
- 1/33 job만 학습 완료 후 test에서 실패

---

### 3. 성공 케이스 분석 (Job 62789)

**성공 요인**:
- Training은 **정상 완료** (Epoch 2까지)
- Validation도 통과
- Valid MSE: 3.199 (개선됨)
- Train metrics 정상 기록:
  ```
  train_mse_emotion_0: 1.010
  train_mse_emotion_1: 0.348
  train_mse_emotion_2: 1.100
  train_mse_emotion_3: 2.940
  train_mse_emotion_4: 0.0473
  train_mse_emotion_5: 0.471
  train_mse_emotion_6: 0.875
  ```

**실패 원인** (Test 단계):
- Output shape 문제
- pl_classifier의 test_step이 LSTM output을 제대로 처리하지 못함

---

## 🎯 근본 원인 분석

### 1. Metadata 매칭 실패의 핵심 원인

기존 SwiFT-IO 시스템의 `data_module.py`는:
- **4D fMRI 파일** 기반으로 subject ID 추출
- Metadata CSV와 **직접 매칭** 시도
- LSTM 학습 시 **다른 형식의 데이터**를 로드할 가능성

**추정**:
```python
# SwiFT-IO의 데이터 로딩
subject_dict = {
    'sub-NDARYK164AEJ': <fmri_data>,
    'sub-NDARFX083RCV': <fmri_data>,
    ...
}

# LSTM이 로드하는 데이터 (추정)
subject_dict = {
    'NDARYK164AEJ': <fmri_data>,  # 'sub-' prefix 없음?
    'NDARFX083RCV': <fmri_data>,
    ...
}
```

### 2. LSTM과 SwiFT-IO의 불일치

| 항목 | SwiFT-IO | LSTM Baseline | 호환성 |
|------|----------|---------------|--------|
| **Data loading** | 4D patches | Full sequence | ⚠️ 다름 |
| **Spatial processing** | Swin blocks | Pooling | ⚠️ 다름 |
| **Output shape** | (B, T, num_emotions) | (B, T, num_emotions) | ✅ 같아야 함 |
| **Metadata** | CSV matching | CSV matching | ✅ 같아야 함 |
| **Test step** | Classification/Regression | Regression | ⚠️ 구현 다름 |

---

## 💡 해결 전략 제안

### 전략 A: Data Module 수정 (권장 ⭐)

**목표**: LSTM이 SwiFT-IO와 동일한 dataloader 사용

**장점**:
- ✅ 공정한 비교 (동일한 데이터)
- ✅ Metadata 문제 근본 해결
- ✅ 향후 baseline 추가 시 재사용 가능

**단점**:
- ⚠️ data_module 수정 필요
- ⚠️ LSTM encoder가 4D input 처리해야 함

**구체적 방법**:

#### Option A1: LSTM-specific data loading 추가
```python
# data_module.py에 추가
def setup(self, stage=None):
    if self.hparams.model == 'lstm_encoder':
        # LSTM용 전처리
        self.train_dataset = LSTMDataset(...)
    else:
        # 기존 SwiFT-IO 로직
        self.train_dataset = fMRIDataset(...)
```

#### Option A2: Unified preprocessing
```python
# 공통 전처리 함수
def preprocess_fmri(data, model_type):
    if model_type == 'lstm':
        # Spatial pooling (96³ → 16³)
        pooled = F.adaptive_avg_pool3d(data, (16, 16, 16))
        return pooled
    else:
        # SwiFT-IO patches
        return data
```

---

### 전략 B: LSTM 독립 실행 (빠른 해결)

**목표**: LSTM을 별도 스크립트로 분리

**장점**:
- ✅ 빠른 구현
- ✅ 기존 코드 건드리지 않음
- ✅ 디버깅 쉬움

**단점**:
- ❌ 코드 중복
- ❌ 공정한 비교 어려울 수 있음
- ❌ Split seed 불일치 가능성

**구체적 방법**:

#### Step 1: 독립적인 LSTM 학습 스크립트
```python
# train_lstm_standalone.py
from torch.utils.data import DataLoader
import nibabel as nib
import pandas as pd

# 1. 직접 fMRI 로딩
def load_fmri_data(image_path):
    subjects = os.listdir(image_path)
    data = {}
    for subj in subjects:
        nii_path = os.path.join(image_path, subj, 'fmri.nii.gz')
        img = nib.load(nii_path)
        data[subj] = img.get_fdata()
    return data

# 2. 직접 metadata 매칭
metadata = pd.read_csv(metadata_path)
matched_data = {k: v for k, v in data.items()
                if k in metadata['SUBJECT_ID'].values}

# 3. LSTM 모델 학습
...
```

#### Step 2: Metadata 매칭 디버깅
```python
# 매칭 전 ID 형식 확인
print("Data keys sample:", list(data.keys())[:5])
print("Metadata IDs sample:", metadata['SUBJECT_ID'].head())

# Format 통일
data_clean = {}
for k, v in data.items():
    # 'sub-' prefix 제거/추가
    clean_k = k.replace('sub-', '') if 'sub-' in k else f'sub-{k}'
    data_clean[clean_k] = v
```

---

### 전략 C: Hybrid 접근 (균형)

**목표**: data_module 최소 수정 + LSTM 전용 처리

**장점**:
- ✅ 기존 시스템 활용
- ✅ LSTM 맞춤 최적화
- ✅ 비교 공정성 유지

**단점**:
- ⚠️ 중간 복잡도

**구체적 방법**:

#### Step 1: Metadata 로딩 수정
```python
# data_module.py
def determine_stratified_split(self, subject_dict, ...):
    df = pd.read_csv(metadata_csv_path)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)

    # ✨ Format 정규화 추가
    subject_ids = set(str(sid) for sid in subject_dict)

    # Try different formats
    def normalize_id(sid):
        # 'sub-' prefix 통일
        if not sid.startswith('sub-'):
            return f'sub-{sid}'
        return sid

    subject_ids_norm = {normalize_id(sid) for sid in subject_ids}
    df['SUBJECT_ID_NORM'] = df['SUBJECT_ID'].apply(normalize_id)

    df = df[df["SUBJECT_ID_NORM"].isin(subject_ids_norm)].copy()

    if df.empty:
        # 디버깅 정보 출력
        print(f"[ERROR] No match found!")
        print(f"Data IDs: {list(subject_ids)[:3]}")
        print(f"Metadata IDs: {list(df_orig['SUBJECT_ID'].head(3))}")
        raise ValueError("No matching SUBJECT_IDs found in metadata.")
```

#### Step 2: Test Step 수정
```python
# pl_classifier.py
def test_step(self, batch, batch_idx):
    ...
    # LSTM 체크
    if self.model_name == 'lstm_encoder':
        # Regression output handling
        output = [(logit.cpu().detach(), targets.cpu().detach())
                  for logit, targets in zip(logits, target)]
        # .item() 제거 (여러 값 가능)
    else:
        # 기존 로직
        output = [(logit.cpu().detach(), targets.cpu().item())
                  for logit, targets in zip(logits, target)]
```

---

## 🚀 권장 실행 계획

### Phase 1: 즉시 조치 (디버깅)
**목표**: 정확한 원인 파악

1. **Metadata 형식 확인**
   ```bash
   # 실제 데이터 ID 형식 확인
   python -c "
   import os
   image_path = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
   subjects = os.listdir(image_path)
   print('Sample data IDs:', subjects[:5])
   "

   # Metadata ID 형식 확인
   python -c "
   import pandas as pd
   df = pd.read_csv('<metadata_path>')
   print('Sample metadata IDs:', df['SUBJECT_ID'].head())
   "
   ```

2. **Debug 스크립트 작성**
   ```python
   # debug_lstm_metadata.py
   # Metadata 매칭 테스트만 수행
   ```

### Phase 2: 단기 해결 (1-2일)
**목표**: LSTM 학습 성공

**Option 1**: 전략 C (Hybrid) 선택
- data_module.py의 `determine_stratified_split` 수정
- ID 정규화 로직 추가
- Test step 수정

**Option 2**: 전략 B (독립) 선택
- 별도 LSTM 스크립트 작성
- Metadata 직접 매칭
- 빠른 프로토타입

### Phase 3: 장기 개선 (1주)
**목표**: 재사용 가능한 baseline 시스템

- 전략 A 구현
- 모든 baseline이 공통 dataloader 사용
- 문서화 및 테스트

---

## 📊 각 전략 비교

| 항목 | 전략 A (Data Module) | 전략 B (독립) | 전략 C (Hybrid) |
|------|---------------------|--------------|----------------|
| **구현 시간** | 3-5일 | 1-2일 | 2-3일 |
| **공정성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **재사용성** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **리스크** | 중간 | 낮음 | 낮음 |
| **유지보수** | 쉬움 | 어려움 | 보통 |

---

## 🎯 최종 권장사항

### 1순위: **전략 C (Hybrid)** ⭐
**이유**:
- ✅ 빠른 구현 (2-3일)
- ✅ 공정한 비교 유지
- ✅ 낮은 리스크
- ✅ 기존 시스템 최대한 활용

**실행 단계**:
1. Phase 1 디버깅 (즉시)
2. `data_module.py` ID 정규화 추가 (1일)
3. `pl_classifier.py` test_step 수정 (1일)
4. LSTM 재학습 및 검증 (1일)

### 2순위: **전략 B (독립)**
**이유**:
- ✅ 가장 빠름 (1-2일)
- ✅ 안전 (기존 코드 영향 없음)

**단점**:
- ⚠️ 코드 중복
- ⚠️ 공정성 검증 필요

---

## 🔧 구체적 코드 수정 제안

### 1. data_module.py 수정

```python
def determine_stratified_split(self, subject_dict, seed, stratified_params, metadata_csv_path,
                              train_split_size=0.7, val_split_size=0.15):
    """
    Stratified split with ID normalization for LSTM compatibility
    """
    df = pd.read_csv(metadata_csv_path)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)

    # ✨ NEW: ID normalization function
    def normalize_subject_id(sid):
        """Normalize subject ID format (ensure 'sub-' prefix)"""
        sid = str(sid).strip()
        if not sid.startswith('sub-'):
            return f'sub-{sid}'
        return sid

    # Normalize both data IDs and metadata IDs
    subject_ids_norm = {normalize_subject_id(sid) for sid in subject_dict}
    df['SUBJECT_ID_NORM'] = df['SUBJECT_ID'].apply(normalize_subject_id)

    # Debug output
    print(f"\n[DEBUG] Subject ID Normalization:")
    print(f"  Data IDs (normalized): {list(subject_ids_norm)[:3]}")
    print(f"  Metadata IDs (normalized): {list(df['SUBJECT_ID_NORM'].head(3))}")
    print(f"  Total data subjects: {len(subject_ids_norm)}")
    print(f"  Total metadata subjects: {len(df)}")

    # Match using normalized IDs
    df = df[df["SUBJECT_ID_NORM"].isin(subject_ids_norm)].copy()

    print(f"  Matched subjects: {len(df)}")

    if df.empty:
        print(f"\n[ERROR] Matching failed!")
        print(f"Sample data IDs: {list(list(subject_dict.keys())[:5])}")
        print(f"Sample metadata IDs: {list(df_orig['SUBJECT_ID'].head())}")
        raise ValueError("No matching SUBJECT_IDs found in metadata.")

    # Use original SUBJECT_ID for split
    X = df["SUBJECT_ID"].values

    # ... rest of the function unchanged
```

### 2. pl_classifier.py 수정

```python
def test_step(self, batch, batch_idx):
    """
    Test step with LSTM-compatible output handling
    """
    images, target = batch
    logits = self(images)

    # ✨ NEW: Model-specific output handling
    if hasattr(self.hparams, 'model') and 'lstm' in self.hparams.model.lower():
        # LSTM regression output: (B, T, num_emotions)
        # target: (B, T, num_emotions)
        output = [(logit.cpu().detach(), targets.cpu().detach())
                  for logit, targets in zip(logits, target)]
    else:
        # SwiFT-IO classification: scalar targets
        output = [(logit.cpu().detach(), targets.cpu().item())
                  for logit, targets in zip(logits, target)]

    return output
```

### 3. 디버깅 스크립트

```python
# debug_lstm_data.py
"""
Debug script to identify metadata matching issue
"""
import os
import pandas as pd
from src.module.utils.data_module import fMRIDataModule
from argparse import Namespace

# Minimal args for data loading
args = Namespace(
    image_path='/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120',
    dataset_name='HBN',
    downstream_task='emotions',
    input_type='movieDM',
    dataset_split_seed=777,
    sequence_length=30,
    # ... other required args
)

# Try to load data
print("="*80)
print("LSTM Data Loading Debug")
print("="*80)

try:
    data_module = fMRIDataModule(**vars(args))
    data_module.setup(stage='fit')
    print("\n✓ Data loading SUCCESSFUL!")
    print(f"  Train samples: {len(data_module.train_dataset)}")
    print(f"  Val samples: {len(data_module.val_dataset)}")

except Exception as e:
    print(f"\n✗ Data loading FAILED!")
    print(f"  Error: {e}")

    # Additional debugging
    print("\nAttempting manual data inspection...")
    image_path = args.image_path
    subjects = sorted([d for d in os.listdir(image_path)
                      if os.path.isdir(os.path.join(image_path, d))])

    print(f"\nFound {len(subjects)} subjects in data directory")
    print(f"Sample IDs: {subjects[:5]}")

    # Check metadata
    metadata_path = '/scratch/HBN/metadata/HBN_metadata.csv'  # adjust path
    df = pd.read_csv(metadata_path)
    print(f"\nMetadata has {len(df)} entries")
    print(f"Sample metadata IDs: {list(df['SUBJECT_ID'].head())}")

    # Check overlap
    overlap = set(subjects) & set(df['SUBJECT_ID'].values)
    print(f"\nDirect overlap: {len(overlap)} subjects")

    # Try with 'sub-' prefix
    subjects_with_prefix = {f'sub-{s}' if not s.startswith('sub-') else s
                           for s in subjects}
    overlap_norm = subjects_with_prefix & set(df['SUBJECT_ID'].values)
    print(f"With normalization: {len(overlap_norm)} subjects")
```

---

## ✅ 성공 기준

LSTM baseline이 성공적으로 작동하려면:

1. ✅ **Data loading 성공**: metadata 매칭 에러 없음
2. ✅ **Training 완료**: 최소 3 epoch 학습
3. ✅ **Validation 정상**: valid_mse 기록
4. ✅ **Test 완료**: test_step 에러 없이 완료
5. ✅ **결과 저장**: checkpoint 및 metrics 저장

**Target metrics** (참고용):
- Train MSE: ~1.0 (Job 62789 기준)
- Valid MSE: ~3.2 (Job 62789 기준)
- Test MSE: SVR-ROI (2.074)보다 나을 것으로 예상

---

## 📚 참고 자료

### 관련 파일
- LSTM 구현: `src/module/models/encoder/lstm_encoder.py`
- LSTM decoder: `src/module/models/decoder/lstm_decoder.py`
- Training script: `src/train_lstm_baseline.py`
- Data module: `src/module/utils/data_module.py`
- Classifier: `src/module/pl_classifier.py`

### 로그 위치
- 최신 성공: `logs/lstm_cpu-62789.out`
- 실패 로그: `logs/lstm_baseline-*.out`
- Test 로그: `logs/lstm_test-*.out`

### 문서
- LSTM 설계: `251014_LSTM_BASELINE.md`
- Baseline 비교: `251023_SVR_Baseline_Methods_Comparison.md`

---

**마지막 업데이트**: 2025-10-24
**상태**: 분석 완료, 해결 전략 제시
**다음 단계**: Phase 1 디버깅 시작
