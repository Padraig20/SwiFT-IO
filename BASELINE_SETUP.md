# SVR Baseline for SwiFT-IO

## Overview

SVR (Support Vector Regression) baseline 모델이 성공적으로 구현되었습니다. 이 baseline은 SwiFT-IO와 **완전히 동일한 input을 사용**하여 공정한 모델 비교를 가능하게 합니다.

## Key Features

### ✅ 공정한 비교
- **Input**: SwiFT-IO와 동일한 4D fMRI sequences (96×96×96×30)
- **Data Split**: 동일한 train/val/test splits 사용
- **Output**: 동일한 emotion timeseries 예측
- **차이점**: 모델만 다름 (SVR vs Swin4D Transformer)

### 🎯 비교 가능성
| Aspect | SVR Baseline | SwiFT-IO |
|--------|--------------|----------|
| Input | 4D fMRI (96×96×96×30) | 4D fMRI (96×96×96×30) ✅ |
| Train/Val/Test Split | seed=777 | seed=777 ✅ |
| Output | 7 emotions × 30 TRs | 7 emotions × 30 TRs ✅ |
| Model | SVR (sklearn) | Swin4D + Perceiver |
| Features | Flattened voxels (~884k) | Hierarchical learned |

---

## File Structure

```
SwiFT-IO/
├── src/
│   ├── baselines/
│   │   ├── svr_baseline.py          # SVR 모델 구현
│   │   ├── glm_baseline.py          # GLM 모델 (기존)
│   │   ├── roi_selector.py          # ROI 선택 (GLM용)
│   │   ├── __init__.py              # 모듈 imports
│   │   └── README.md                # Baseline 문서
│   │
│   ├── train_svr_baseline.py        # SVR 학습 스크립트
│   ├── train_glm_baseline.py        # GLM 학습 스크립트
│   └── main.py                      # SwiFT-IO 메인
│
├── sample_scripts/
│   └── run_svr_baseline.slurm       # SLURM 실행 스크립트
│
├── run_svr_baseline.sh              # Bash 실행 스크립트
├── run_glm_baseline.sh              # GLM 실행 스크립트
│
└── output/
    ├── svr_baseline/                # SVR 결과 저장
    └── glm_baseline/                # GLM 결과 저장
```

---

## Quick Start

### 1. Interactive Shell에서 실행

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Conda 환경 활성화
conda activate swiftio

# SVR baseline 학습
bash run_svr_baseline.sh
```

### 2. SLURM으로 실행 (권장)

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# SLURM job 제출
sbatch sample_scripts/run_svr_baseline.slurm

# Log 확인
tail -f logs/svr_baseline-<JOB_ID>.out
```

---

## Configuration

### 기본 설정 (run_svr_baseline.sh)

```bash
--image_path /scratch/HBN/9.2.movieDM_SwiFT     # 4D fMRI data
--dataset_split_seed 777                        # Same as SwiFT-IO
--sequence_length 30                            # Same as SwiFT-IO
--kernel rbf                                    # RBF kernel
--C 1.0                                         # Regularization
--epsilon 0.1                                   # SVR epsilon
--batch_size 4                                  # Data loading
--num_workers 8                                 # Parallel loading
```

### 커스터마이징

다른 hyperparameter를 시도하려면:

```bash
python src/train_svr_baseline.py \
    --image_path /scratch/HBN/9.2.movieDM_SwiFT \
    --dataset_split_seed 777 \
    --sequence_length 30 \
    --kernel linear \              # 'linear', 'rbf', 'poly'
    --C 10.0 \                     # Higher C = less regularization
    --epsilon 0.01 \               # Smaller epsilon = stricter fit
    --output_dir output/svr_linear
```

---

## Output

학습이 완료되면 `output/svr_baseline/`에 다음 파일들이 저장됩니다:

### 1. `svr_metrics.json`
```json
{
  "train_mse": 0.1234,
  "train_mae": 0.0567,
  "valid_mse": 0.1456,
  "valid_mae": 0.0678,
  "test_mse": 0.1523,
  "test_mae": 0.0702,
  "train_mse_0": 0.1234,  # Emotion 0 (Anger)
  "train_mse_1": 0.1345,  # Emotion 1 (Happy)
  ...
}
```

### 2. `svr_model.pkl`
- 학습된 SVR 모델 (7개 emotion × 1개 SVR)
- Scalers (standardization)

### 3. `svr_config.json`
```json
{
  "task": "emotions",
  "sequence_length": 30,
  "kernel": "rbf",
  "C": 1.0,
  "epsilon": 0.1,
  "feature_dim": 884736,
  "num_train_samples": 12345,
  "num_val_samples": 2345,
  "num_test_samples": 2345
}
```

### 4. `training_summary.txt`
- 간단한 학습 요약

---

## Expected Performance

### 학습 시간
- **SVR**: ~2-6 시간 (full voxels, RBF kernel)
- **GLM**: ~10-30 분 (34 ROIs only)
- **SwiFT-IO**: ~12-24 시간 (full 4D, Transformer)

### 예상 성능
| Model | Input Dim | Test MSE | Test MAE | Notes |
|-------|-----------|----------|----------|-------|
| SVR (rbf) | ~884k voxels | ~0.15-0.25 | ~0.08-0.12 | Same input as SwiFT-IO ✅ |
| GLM | 34 ROIs | ~0.30-0.50 | ~0.15-0.20 | Reduced input ⚠️ |
| SwiFT-IO | 96³×30 | TBD | TBD | Target performance |

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**원인**: SVR은 ~884k voxels를 메모리에 한번에 로드합니다.

**해결책**:
1. `--batch_size` 줄이기 (4 → 2 or 1)
2. Linear kernel 사용 (`--kernel linear`)
3. Voxel subsampling 추가 (코드 수정 필요)

### Issue 2: Training too slow

**원인**: RBF kernel은 계산량이 많습니다.

**해결책**:
1. Linear kernel 사용 (`--kernel linear`)
2. 샘플 수 제한 (`--limit_training_samples 1000`)

### Issue 3: Import errors

**원인**: Conda 환경이 활성화되지 않음

**해결책**:
```bash
conda activate swiftio
python src/train_svr_baseline.py ...
```

---

## Comparison with SwiFT-IO

### 공정한 비교를 위한 체크리스트

✅ **Same input**: 4D fMRI (96×96×96×30)
✅ **Same data split**: seed=777
✅ **Same output**: 7 emotions × 30 TRs
✅ **Same preprocessing**: Identical data pipeline
✅ **Same evaluation**: MSE, MAE, R², Correlation

### 모델 차이점

| Aspect | SVR | SwiFT-IO |
|--------|-----|----------|
| Architecture | Support Vector Machine | Swin4D Transformer |
| Feature Engineering | None (raw voxels) | Hierarchical patches |
| Temporal Modeling | Independent timepoints | 4D attention |
| Training | Convex optimization | Gradient descent |
| Interpretability | Feature weights | Attention maps |

---

## Next Steps

### 1. Run SVR Baseline
```bash
sbatch sample_scripts/run_svr_baseline.slurm
```

### 2. Compare with SwiFT-IO
- Train SwiFT-IO with same `--dataset_split_seed 777`
- Compare test MSE/MAE/R² between models
- Analyze per-emotion performance

### 3. Additional Experiments

#### Experiment A: Different kernels
```bash
# Linear kernel (faster)
python src/train_svr_baseline.py --kernel linear --output_dir output/svr_linear

# Polynomial kernel
python src/train_svr_baseline.py --kernel poly --output_dir output/svr_poly
```

#### Experiment B: Different sequence lengths
```bash
# Shorter sequence (seq=20, like GLM)
python src/train_svr_baseline.py --sequence_length 20 --output_dir output/svr_seq20

# Longer sequence (seq=40)
python src/train_svr_baseline.py --sequence_length 40 --output_dir output/svr_seq40
```

---

## Citation

If you use this SVR baseline in your research, please cite:

```bibtex
@article{your_paper,
  title={SwiFT-IO: Swin 4D fMRI Transformer for Individualized Output prediction},
  author={Your Name},
  journal={Journal},
  year={2025}
}
```

---

## Contact

For questions or issues:
- Check [src/baselines/README.md](src/baselines/README.md) for detailed documentation
- Open an issue on GitHub
- Contact: kimbo@connectome

---

**Last Updated**: 2025-10-13
**Status**: ✅ Ready to use
