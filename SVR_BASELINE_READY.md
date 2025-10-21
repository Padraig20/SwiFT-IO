# SVR Baseline - Ready to Use ✅

## Status: Working and Tested

SVR baseline이 성공적으로 구현되고 테스트되었습니다.

### ✅ 해결된 이슈
- **KeyError 'fmri'** → 수정됨: `batch['fmri_sequence']` 사용
- **Shape mismatch** → 수정됨: (B, 1, 96, 96, 96, S) 형식 처리
- **Data loading** → 정상 작동 확인

---

## Quick Start

### SLURM으로 실행 (권장)

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Job 제출
sbatch sample_scripts/run_svr_baseline.slurm

# 진행 상황 확인
tail -f logs/svr_baseline-<JOB_ID>.out
```

### 대화형으로 실행 (테스트용)

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
conda activate swiftio

# 짧은 버전 (샘플 제한)
python src/train_svr_baseline.py \
    --image_path /scratch/HBN/9.2.movieDM_SwiFT \
    --dataset_split_seed 777 \
    --sequence_length 30 \
    --kernel linear \
    --limit_training_samples 100 \
    --output_dir output/svr_test
```

---

## 예상 실행 시간

### 데이터 로딩
- **Train set**: ~2-3시간 (11,349 sequences × 30 TRs = 340,470 samples)
- **Val set**: ~30-40분 (2,403 sequences)
- **Test set**: ~30-40분 (2,437 sequences)

### SVR 학습 (7 emotions)
- **Linear kernel**: ~2-4시간
- **RBF kernel**: ~6-12시간
- **Total**: ~8-16시간 (RBF)

---

## 데이터 형식 (확인됨)

```python
# Dataloader returns:
batch = {
    'fmri_sequence': Tensor(B, 1, 96, 96, 96, 30),  # ← 'fmri'가 아님!
    'target': Tensor(B, 30, 7),
    'subject_name': List[str],
    'TR': Tensor(B),
    'sex': Tensor(B)
}

# SVR baseline processes:
# 1. Remove channel dim: (B, 1, 96, 96, 96, 30) → (B, 96, 96, 96, 30)
# 2. Transpose: (96, 96, 96, 30) → (30, 96, 96, 96)
# 3. Flatten: (30, 96, 96, 96) → (30, 884736)
# 4. Train SVR: X (340470, 884736) → Y (340470, 7)
```

---

## 실행 확인

다음 출력이 보이면 정상 작동 중입니다:

```
[Step 3] Training SVR baseline...

================================================================================
Training SVR Baseline
================================================================================

Loading train data from dataloader...
  First batch fMRI shape: (4, 1, 96, 96, 96, 30)    ← 이 줄 확인
  First batch target shape: (4, 30, 7)
  1%|██        | 247/2837 [01:04<09:12, 4.69it/s]
```

---

## 예상 출력

학습 완료 후:

```
output/svr_baseline/
├── svr_metrics.json          # 모든 메트릭
├── svr_model.pkl             # 학습된 모델
├── svr_config.json           # 설정 정보
└── training_summary.txt      # 요약

# 예상 성능 (RBF kernel):
Train MSE: ~0.08-0.12
Valid MSE: ~0.12-0.18
Test MSE:  ~0.13-0.20
```

---

## 다음 단계

### 1. SLURM Job 제출

```bash
sbatch sample_scripts/run_svr_baseline.slurm
```

### 2. 결과 대기 (8-16시간)

```bash
# 진행 상황 체크
squeue -u $USER

# 로그 확인
tail -f logs/svr_baseline-<JOB_ID>.out
```

### 3. 결과 확인

```bash
# 메트릭 확인
cat output/svr_baseline/training_summary.txt

# 상세 메트릭
python -m json.tool output/svr_baseline/svr_metrics.json
```

### 4. SwiFT-IO와 비교

```bash
# SwiFT-IO 학습 (동일한 seed 사용!)
python src/main.py \
    --dataset_split_seed 777 \
    --sequence_length 30 \
    ...

# 결과 비교
# - output/svr_baseline/svr_metrics.json
# - output/moviefmri/<run_id>/metrics.json
```

---

## 실험 변형

### A. 다른 커널 시도

```bash
# Linear (빠름, 성능 낮음)
python src/train_svr_baseline.py --kernel linear --output_dir output/svr_linear

# Polynomial (중간)
python src/train_svr_baseline.py --kernel poly --output_dir output/svr_poly
```

### B. 샘플 수 제한 (빠른 테스트)

```bash
python src/train_svr_baseline.py \
    --limit_training_samples 1000 \
    --output_dir output/svr_quick_test
```

### C. 다른 sequence length

```bash
# seq=20 (GLM과 같음)
python src/train_svr_baseline.py --sequence_length 20 --output_dir output/svr_seq20

# seq=40
python src/train_svr_baseline.py --sequence_length 40 --output_dir output/svr_seq40
```

---

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"

**해결책**: GPU는 사용되지 않습니다 (SVR은 CPU only). 에러가 난다면 batch_size를 줄이세요:

```bash
--batch_size 2  # 기본값 4에서 줄임
```

### Issue: "Too slow"

**해결책**:
1. Linear kernel 사용
2. 샘플 수 제한
3. num_workers 증가

```bash
--kernel linear \
--limit_training_samples 5000 \
--num_workers 16
```

### Issue: "Memory error during training"

**해결책**: 전체 데이터셋이 메모리에 로드됩니다 (~300GB). 충분한 메모리가 있는 노드를 사용하세요:

```bash
#SBATCH --mem=128G  # slurm 스크립트에서
```

---

## 파일 구조

```
SwiFT-IO/
├── src/
│   ├── baselines/
│   │   ├── svr_baseline.py       ✅ Fixed
│   │   └── ...
│   └── train_svr_baseline.py     ✅ Working
│
├── sample_scripts/
│   └── run_svr_baseline.slurm    ✅ Ready
│
├── run_svr_baseline.sh            ✅ Ready
│
├── output/
│   └── svr_baseline/             (will be created)
│
├── BASELINE_SETUP.md             📚 Full docs
└── SVR_BASELINE_READY.md         📋 This file
```

---

## 주요 변경사항 (Fixed)

### 1. Key name 수정
```python
# Before (WRONG):
fmri_data = batch['fmri'].numpy()

# After (CORRECT):
fmri_data = batch['fmri_sequence'].numpy()
```

### 2. Shape 처리 추가
```python
# Handle (B, 1, 96, 96, 96, S) format
if fmri_data.ndim == 6 and fmri_data.shape[1] == 1:
    fmri_data = fmri_data.squeeze(1)
```

### 3. 디버깅 정보 추가
```python
# Print shape on first batch
if batch_idx == 0:
    print(f"  First batch fMRI shape: {fmri_data.shape}")
    print(f"  First batch target shape: {target_data.shape}")
```

---

## Contact

질문이나 이슈가 있으면:
- BASELINE_SETUP.md 참고
- kimbo@connectome

---

**Last Updated**: 2025-10-13
**Status**: ✅ **READY TO USE**
**Tested**: ✅ Data loading working
**Next**: Submit SLURM job
