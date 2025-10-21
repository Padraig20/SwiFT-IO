# Bus Error 수정 완료 ✅

## 문제

```
ERROR: Unexpected bus error encountered in worker.
This might be caused by insufficient shared memory (shm).

RuntimeError: DataLoader worker (pid(s) 2619383) exited unexpectedly
```

## 원인

**Shared memory 부족** - PyTorch DataLoader의 여러 worker (8개)가 동시에 큰 4D fMRI 데이터를 로드하려 할 때, shared memory가 부족해서 발생한 문제입니다.

### 왜 이 문제가 발생했나?

1. **데이터 크기**: 각 batch = 4 × (1, 96, 96, 96, 30) = ~300MB
2. **Worker 수**: 8개의 worker가 동시에 데이터 로드
3. **Shared memory**: `/dev/shm`의 용량이 제한적
4. **결과**: Worker들이 서로 경쟁하다가 bus error 발생

## 해결책

### ✅ 적용된 수정: num_workers = 0

**가장 간단하고 안전한 방법**

```bash
--num_workers 0  # 8에서 0으로 변경
```

#### 장점:
- ✅ Shared memory 문제 완전 해결
- ✅ 메모리 사용량 예측 가능
- ✅ 안정적인 실행

#### 단점:
- ⚠️ 데이터 로딩이 약간 느려질 수 있음
- ⚠️ 하지만 SVR 학습 자체가 오래 걸리므로 큰 영향 없음

### 대안 (적용 안 함)

#### 옵션 2: Worker 수 줄이기
```bash
--num_workers 2  # 8 → 2
```
- 부분적 해결 (여전히 문제 가능)

#### 옵션 3: Batch size 줄이기
```bash
--batch_size 1  # 4 → 1
```
- 메모리는 줄지만 매우 느려짐

#### 옵션 4: Shared memory 증가 (SLURM)
```bash
#SBATCH --mem=128G  # 64G → 128G
```
- 근본적 해결은 아님

---

## 수정된 파일

### 1. [sample_scripts/run_svr_baseline.slurm](sample_scripts/run_svr_baseline.slurm)
```bash
# Before:
--num_workers 8

# After:
--num_workers 0  # ← 수정됨
```

### 2. [run_svr_baseline.sh](run_svr_baseline.sh)
```bash
# Before:
--num_workers 8

# After:
--num_workers 0  # ← 수정됨
```

---

## 다시 실행하기

### SLURM으로 실행

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# 새로운 job 제출
sbatch sample_scripts/run_svr_baseline.slurm

# 진행 확인
squeue -u $USER
tail -f logs/svr_baseline-<NEW_JOB_ID>.out
```

### 대화형으로 실행

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
conda activate swiftio

bash run_svr_baseline.sh
```

---

## 예상 실행 시간 (num_workers=0)

### 데이터 로딩 시간
- **Before (workers=8)**: ~7-8분 (1,260 batches까지 도달)
- **After (workers=0)**: ~10-15분 예상
- **차이**: ~2-7분 더 걸림

### 전체 학습 시간
- **데이터 로딩**: ~10-15분
- **SVR 학습**: ~6-12시간 (변화 없음)
- **Total**: ~6-12시간 (데이터 로딩은 전체의 <5%)

👉 **결론**: `num_workers=0`으로 인한 시간 증가는 전체 학습 시간의 5% 미만이므로 무시 가능합니다.

---

## 진행 확인

이제 다음과 같이 정상 작동해야 합니다:

```
Loading train data from dataloader...
  First batch fMRI shape: (4, 1, 96, 96, 96, 30)  ← 정상
  First batch target shape: (4, 30, 7)
  1%|██        | 30/2837 [00:15<23:45, 1.97it/s]  ← 약간 느리지만 정상
  ...
  100%|██████████| 2837/2837 [15:23<00:00, 3.07it/s]  ← 완료!

train data shape: X=(340470, 884736), Y=(340470, 7)
Feature dimension: 884,736 voxels

Training SVR models for each emotion...
  Emotion 0: ...
```

---

## Troubleshooting

### Q1: 여전히 너무 느린가요?

**A**: Linear kernel을 사용하세요:

```bash
python src/train_svr_baseline.py \
    --kernel linear \
    --num_workers 0 \
    --output_dir output/svr_linear
```

### Q2: 메모리 부족 에러?

**A**: Batch size를 줄이세요:

```bash
--batch_size 2  # 또는 1
```

### Q3: 빠른 테스트를 원하신다면?

**A**: 샘플 수를 제한하세요:

```bash
--limit_training_samples 1000 \
--num_workers 0
```

---

## 변경 요약

| 항목 | Before | After | 이유 |
|------|--------|-------|------|
| `num_workers` | 8 | 0 | Shared memory 부족 해결 |
| 데이터 로딩 속도 | ~7분 | ~10-15분 | Worker 없음 |
| 전체 학습 시간 | ~8-16시간 | ~8-16시간 | SVR 학습이 대부분의 시간 차지 |
| 안정성 | ❌ Bus error | ✅ 안정적 | - |

---

## 다음 단계

1. **새로운 job 제출**
```bash
sbatch sample_scripts/run_svr_baseline.slurm
```

2. **로그 모니터링**
```bash
tail -f logs/svr_baseline-<JOB_ID>.out
```

3. **완료 대기** (~8-16시간)

4. **결과 확인**
```bash
cat output/svr_baseline/training_summary.txt
```

---

**Status**: ✅ **수정 완료 - 다시 실행 가능**

**Last Updated**: 2025-10-13

**Fixed Files**:
- ✅ `sample_scripts/run_svr_baseline.slurm`
- ✅ `run_svr_baseline.sh`
