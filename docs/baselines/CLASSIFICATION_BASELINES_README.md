# Classification Baselines for SwiFT-IO

This document describes the classification baseline models for comparing with SwiFT-IO classification performance (run ID: `mc3r4vhf`).

## Overview

We extended all regression baselines to support classification tasks for fair comparison:

1. **SVC Baseline**: Support Vector Classification with RBF kernel
2. **LSTM Classification Baseline**: LSTM encoder + classification head

## Experiment Setup

To match the classification experiment (`mc3r4vhf`):

- **Sequence length**: 20 TRs (vs 30 for regression)
- **Dataset split seed**: 2 (vs 777 for regression)
- **Task**: 7-class emotion classification per timepoint
- **Emotions**: Anger, Happy, Fear, Sad, Excited, Positive, Negative
- **Data**: HBN movieDM fMRI (same as regression baselines)
- **Train/Val/Test split**: 70%/15%/15%

## Models

### 1. SVC (Support Vector Classification) Baseline

**Architecture**:
- One SVC model per emotion (7 total)
- Input: Flattened 4D fMRI sequence (96×96×96×20 → 1,769,472 voxels × 20 TRs)
- Output: Binary classification (0 or 1) per emotion per timepoint
- Kernel: RBF (default)
- Regularization: C=1.0

**Training**:
```bash
sbatch run_train_svc_baseline.slurm
```

**Evaluation**:
```bash
python evaluate_svc_baseline.py \
    --model_path output/svc_baseline_seq20/svr_model.pkl \
    --dataset_split_seed 2 \
    --sequence_length 20
```

### 2. LSTM Classification Baseline

**Architecture**:
- LSTM encoder (256 hidden dim, 2 layers, bidirectional)
- Spatial pooling: Adaptive pooling to 16³
- Classification head: Linear layer → 7-class output per timepoint
- Dropout: 0.3

**Training**:
```bash
sbatch run_train_lstm_clf_baseline.slurm
```

**Evaluation**:
```bash
python evaluate_lstm_clf_baseline.py --run_id <wandb_run_id>
```

## Files Created/Modified

### Core Baseline Code
- `src/baselines/svr_baseline.py`: Added `task_type` parameter ('regression'/'classification')
  - Classification: Uses `SVC` with probability=True
  - Regression: Uses `SVR` with epsilon parameter
  - Metrics: Accuracy, F1, Precision, Recall (classification) or MSE, MAE, R² (regression)

### Training Scripts
- `src/train_svr_baseline.py`: Added `--task_type` argument
- `src/train_lstm_baseline.py`: Added decoder selection based on task type
  - Classification → `lstm_classification_head`
  - Regression → `lstm_regression_head`

### Wrapper Scripts
- `train_svc_baseline.py`: Convenience wrapper for SVC training (sets classification defaults)
- `run_train_svc_baseline.slurm`: SLURM script for SVC training (CPU, 128GB RAM, 24h)
- `run_train_lstm_clf_baseline.slurm`: SLURM script for LSTM classification (GPU, 64GB RAM, 48h)

### Evaluation Scripts
- `evaluate_svc_baseline.py`: Evaluate SVC baseline from checkpoint
- `evaluate_lstm_clf_baseline.py`: Evaluate LSTM classification baseline

## Usage Examples

### Train SVC Baseline (Classification)

```bash
# Using SLURM (recommended)
sbatch run_train_svc_baseline.slurm

# Or directly
python src/train_svr_baseline.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --task_type classification \
    --downstream_task_type classification \
    --sequence_length 20 \
    --dataset_split_seed 2 \
    --output_dir output/svc_baseline_seq20
```

### Train LSTM Classification Baseline

```bash
# Using SLURM (recommended)
sbatch run_train_lstm_clf_baseline.slurm

# Or directly
python src/train_lstm_baseline.py \
    --downstream_task_type classification \
    --sequence_length 20 \
    --dataset_split_seed 2 \
    --lstm_hidden_dim 256 \
    --lstm_num_layers 2 \
    --decoder lstm_classification_head
```

### Evaluate Baselines

```bash
# SVC evaluation
python evaluate_svc_baseline.py \
    --model_path output/svc_baseline_seq20/svr_model.pkl

# LSTM evaluation
python evaluate_lstm_clf_baseline.py \
    --run_id <wandb_run_id>
```

## Expected Outputs

### SVC Baseline
```
output/svc_baseline_seq20/
├── svr_model.pkl                 # Trained model
├── svr_config.json              # Configuration
├── svr_metrics.json             # Train/val/test metrics
├── training_summary.txt         # Summary
└── evaluation/
    ├── test_results.json        # Test set results
    └── evaluation_summary.txt   # Evaluation summary
```

### LSTM Classification Baseline
```
output/moviefmri/<run_id>/
├── checkpt-epoch=XX-valid_acc=X.XX.ckpt  # Best checkpoint
├── last.ckpt                              # Last checkpoint
└── evaluation/
    ├── lstm_clf_test_results.json         # Test results
    └── lstm_clf_evaluation_summary.txt    # Summary
```

## Metrics

For classification baselines, we report:

**Overall**:
- Accuracy: Overall classification accuracy
- F1 Score: Weighted F1 score
- Precision: Weighted precision
- Recall: Weighted recall

**Per-Emotion**:
- Accuracy, F1, Precision, Recall for each of 7 emotions

## Comparison with SwiFT-IO Classification

Reference experiment: `mc3r4vhf`
- Path: `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/mc3r4vhf`
- Checkpoint: `checkpt-epoch=08-valid_acc=1.00.ckpt`

### Evaluation Commands

```bash
# SwiFT-IO classification
python evaluate_clf_test_subjects.py --run_id mc3r4vhf

# SVC baseline
python evaluate_svc_baseline.py \
    --model_path output/svc_baseline_seq20/svr_model.pkl

# LSTM baseline
python evaluate_lstm_clf_baseline.py --run_id <lstm_run_id>
```

## Notes

1. **Computational Requirements**:
   - SVC: CPU-intensive, requires ~128GB RAM, ~24h training time
   - LSTM: GPU-required, ~64GB RAM, ~24-48h training time
   - SwiFT-IO: Multi-GPU, ~256GB RAM, ~100h training time

2. **Data Consistency**:
   - All baselines use the same data splits (seed=2)
   - Same preprocessing pipeline as SwiFT-IO
   - Same evaluation protocol

3. **Fair Comparison**:
   - All models use seq_length=20 for classification
   - Same train/val/test split (70/15/15)
   - Same metrics (Accuracy, F1, Precision, Recall)

4. **Binary vs Multi-class**:
   - We train 7 separate binary classifiers (one per emotion)
   - Each predicts presence/absence of an emotion at each timepoint
   - Overall metrics are computed by flattening all predictions

## References

- Regression baselines: `docs/baselines/baseline_performance_table.md`
- SwiFT-IO architecture: `src/module/models/`
- Data module: `src/module/utils/data_module.py`

## Author

Created: 2025-10-28
Based on regression baseline methodology from baseline_performance_table.md
