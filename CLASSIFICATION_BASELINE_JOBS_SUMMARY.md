# Classification Baseline Training Jobs - Summary

모든 regression baseline을 classification 버전으로 확장했습니다. 각 job script는 성공했던 regression 실험을 기반으로 작성되었습니다.

## 실행 시간 추정 및 Job Scripts

| Baseline | Script | Expected Runtime | Resources | Priority |
|----------|--------|-----------------|-----------|----------|
| **SVC (Full Voxels)** | `run_train_svc_cls_baseline.slurm` | **12-16h** | CPU, 128GB | ⭐⭐⭐ High |
| **LSTM Classification** | `run_train_lstm_cls_baseline.slurm` | **18-22h** | 6×GPU, 192GB | ⭐⭐⭐ High |
| **SVC (ROI)** | `run_train_svc_roi_cls_baseline.slurm` | **5-7h** | CPU, 64GB | ⭐⭐ Medium |
| **SVC (PCA)** | `run_train_svc_pca_cls_baseline.slurm` | **7-9h** | CPU, 64GB | ⭐⭐ Medium |

## 우선순위 순서

### 1단계: 핵심 Baselines (필수) ⭐⭐⭐

이 두 개는 논문의 main comparison을 위해 **반드시** 필요합니다:

```bash
# 1. SVC Baseline (Full voxels) - 가장 기본적인 baseline
sbatch run_train_svc_cls_baseline.slurm

# 2. LSTM Classification - Deep learning baseline
sbatch run_train_lstm_cls_baseline.slurm
```

### 2단계: Feature Engineering Baselines (선택) ⭐⭐

Ablation study와 feature importance 분석을 위해 유용:

```bash
# 3. SVC with ROI - Anatomical prior의 효과 확인
sbatch run_train_svc_roi_cls_baseline.slurm

# 4. SVC with PCA - Data-driven reduction의 효과 확인
sbatch run_train_svc_pca_cls_baseline.slurm
```

## 상세 정보

### 1. SVC Full Voxels Baseline

**Script**: `run_train_svc_cls_baseline.slurm`

```bash
#SBATCH --time=18:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=node3
```

**예상 실행 시간**: 12-16시간
- Sequence length: 20 (regression은 30)
- Feature dimension: 96³ × 20 = 1,769,472 features
- 7개 SVC 모델 독립 학습 (각 emotion마다)
- Classification은 epsilon-SVR보다 빠를 것으로 예상

**Regression 대비**:
- Seq length 33% 감소 → ~30% 시간 단축
- Binary SVC vs epsilon-SVR → 유사한 속도
- Estimated: 12-16h (regression 18-24h 기준)

---

### 2. LSTM Classification Baseline

**Script**: `run_train_lstm_cls_baseline.slurm`

```bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx:6
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node1
```

**예상 실행 시간**: 18-22시간
- 6 RTX GPUs with DDP
- LSTM encoder (256 hidden, 2 layers)
- Classification head (binary per emotion)
- Max epochs: 30, early stopping 예상 epoch 15-20

**Regression 대비**:
- Seq length 33% 감소 → ~30% 시간 단축
- CrossEntropy loss (simpler than MSE regression)
- Early stopping likely faster convergence
- Estimated: 18-22h (regression 6-8h for 30 epochs, but with early stopping)

---

### 3. SVC ROI Baseline

**Script**: `run_train_svc_roi_cls_baseline.slurm`

```bash
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=node4
```

**예상 실행 시간**: 5-7시간
- ROI reduction: ~109 ROIs × 20 TRs = 2,180 features
- Much smaller feature space than full voxels
- Uses precomputed ROI timeseries (FreeSurfer parcellation)

**Regression 대비**:
- Seq length 33% 감소
- ROI regression took ~2-3h for seq=30
- Estimated: 5-7h (including ROI extraction overhead)

---

### 4. SVC PCA Baseline

**Script**: `run_train_svc_pca_cls_baseline.slurm`

```bash
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=node1
```

**예상 실행 시간**: 7-9시간
- PCA: 100 components per TR × 20 TRs = 2,000 features
- IncrementalPCA fitting: ~1-2h on train set
- SVC training: ~5-7h on reduced features

**Regression 대비**:
- PCA regression took ~3-4h for seq=30
- Estimated: 7-9h (33% less data but PCA overhead)

---

## 공통 설정 (모든 baselines)

### Dataset Configuration
```python
sequence_length = 20  # vs 30 for regression
dataset_split_seed = 2  # vs 777 for regression (match mc3r4vhf)
train_split = 0.7
val_split = 0.15
test_split = 0.15
```

### Task Configuration
```python
task_type = "classification"
downstream_task_type = "classification"
num_emotions = 7  # Binary classification per emotion
emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
```

### Data Path
```bash
IMAGE_PATH="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
```

### Reference Experiment
- SwiFT-IO Classification: `mc3r4vhf`
- Checkpoint: `checkpt-epoch=08-valid_acc=1.00.ckpt`

---

## 실행 방법

### 병렬 실행 (추천)

모든 baseline을 동시에 실행하려면:

```bash
# Create logs directory
mkdir -p logs

# Submit all jobs
sbatch run_train_svc_cls_baseline.slurm        # node3, CPU, 12-16h
sbatch run_train_lstm_cls_baseline.slurm       # node1, 6 GPU, 18-22h
sbatch run_train_svc_roi_cls_baseline.slurm    # node4, CPU, 5-7h
sbatch run_train_svc_pca_cls_baseline.slurm    # node1, CPU, 7-9h

# Check job status
squeue -u $USER
```

### 순차 실행

리소스 제약이 있다면 우선순위 순서로:

```bash
# Priority 1: Full voxel SVC
sbatch run_train_svc_cls_baseline.slurm

# Priority 2: LSTM (after SVC finishes or in parallel if GPU available)
sbatch run_train_lstm_cls_baseline.slurm

# Priority 3: ROI-based (after checking Priority 1 results)
sbatch run_train_svc_roi_cls_baseline.slurm

# Priority 4: PCA-based (for completeness)
sbatch run_train_svc_pca_cls_baseline.slurm
```

---

## Output 구조

각 baseline은 다음과 같은 결과를 생성합니다:

### SVC Baselines
```
output/svc_cls_baseline_seq20/
├── svr_model.pkl              # Trained SVC models (7 classifiers)
├── svr_config.json            # Configuration
├── svr_metrics.json           # Train/val/test metrics
└── training_summary.txt       # Summary

output/svc_reduction_roi_cls_seq20/  # ROI version
output/svc_reduction_pca_cls_seq20/  # PCA version
```

### LSTM Baseline
```
output/moviefmri/<run_id>/
├── checkpt-epoch=XX-valid_acc=X.XX.ckpt  # Best checkpoint
├── last.ckpt                              # Last checkpoint
└── wandb logs
```

---

## 평가 방법

학습 완료 후:

```bash
# SVC evaluation
python evaluate_svc_baseline.py \
    --model_path output/svc_cls_baseline_seq20/svr_model.pkl

# LSTM evaluation
python evaluate_lstm_clf_baseline.py --run_id <wandb_run_id>

# SwiFT-IO reference (for comparison)
python evaluate_clf_test_subjects.py --run_id mc3r4vhf
```

---

## 예상 전체 소요 시간

**병렬 실행 (모든 노드 사용 가능시)**:
- 최대: ~22시간 (LSTM이 가장 오래 걸림)
- 최소: ~18시간 (early stopping으로 단축 가능)

**순차 실행 (하나씩)**:
- SVC (16h) + LSTM (20h) + ROI (6h) + PCA (8h) = **~50시간**

**추천 실행 방식**:
1. SVC + LSTM 동시 실행 (다른 노드 사용) → **~20시간**
2. ROI + PCA 선택적 실행 → **추가 +8시간**

---

## 주의사항

### ⚠️ ROI/PCA Baselines
`train_svr_with_reduction.py`가 아직 classification을 완전히 지원하지 않을 수 있습니다.
실행 중 오류 발생 시:
1. Full voxel SVC/LSTM 결과를 먼저 확인
2. ROI/PCA는 optional ablation으로 처리

### 🔧 필요시 코드 수정
`src/train_svr_with_reduction.py`에 task_type 파라미터 추가 필요:
```python
parser.add_argument("--task_type", type=str, default="regression",
                   choices=['regression', 'classification'])
```

---

## 성공 체크리스트

- [ ] SVC Full Voxel Baseline 학습 완료
- [ ] LSTM Classification Baseline 학습 완료
- [ ] SVC ROI Baseline 학습 완료 (optional)
- [ ] SVC PCA Baseline 학습 완료 (optional)
- [ ] 모든 baseline evaluation 완료
- [ ] SwiFT-IO (mc3r4vhf)와 성능 비교 완료
- [ ] 결과를 baseline_performance_table.md에 추가

---

**생성일**: 2025-10-28
**기준**: Regression baselines (seq=30, seed=777) 성공 사례 기반
**목표**: Classification experiment (mc3r4vhf, seq=20, seed=2)와 비교
