# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Rules

- **Respond in Korean** (keep code and technical terms in English)
- **All code must be 100% English** (variables, functions, comments, docstrings)

## Project Overview

SwiFT-IO is a 4D fMRI analysis framework using Swin Transformer with Perceiver IO-inspired decoders for predicting emotions during movie viewing (HBN movieDM dataset). Supports 7 emotions (Anger, Happy, Fear, Sad, Excited, Positive, Negative) with regression and classification tasks.

## Build & Run Commands

```bash
# Environment setup
source ~/.bashrc && conda activate swiftio

# Quick import test
python -c "from src.main import cli_main; print('Import OK')"

# Training via SLURM
sbatch sample_scripts/250602_seq20/regression_seq20_off0_sch_stratified.sh

# Check job status
squeue -u $USER

# View recent logs
tail -50 logs/<JOB_ID>-<NAME>.o
```

## Key Training Parameters

```bash
python src/main.py \
  --model swin4d_ver9 \                # or swin4d_ver7, swin4d_ver11
  --decoder series_decoder \           # or single_target_decoder, averaged_series_decoder
  --downstream_task emotions \         # or sex, age
  --downstream_task_type regression \  # or classification
  --sequence_length 20 \
  --num_targets 7 \
  --stratified_params Age Sex          # stratified train/val/test split
```

## Architecture

### Model Flow
```
4D fMRI Input (B, C, H, W, D, T)
    ↓
Encoder (swin4d_ver7/ver9/ver11)
    ↓
Decoder (series_decoder/single_target_decoder/averaged_series_decoder)
    ↓
Predictions (per-timepoint or aggregated)
```

### Core Components

- **`src/main.py`**: Entry point, handles training/testing with PyTorch Lightning
- **`src/module/pl_classifier.py`**: LitClassifier - main Lightning module with train/val/test logic
- **`src/module/models/load_model.py`**: Factory function loading encoder/decoder by name
- **`src/module/utils/data_module.py`**: fMRIDataModule - data loading, stratified splits
- **`src/module/utils/datasets.py`**: HBN dataset class for movieDM fMRI data

### Encoder Versions
- **Ver7**: Original Swin4D Transformer
- **Ver9**: Enhanced Swin4D with improvements
- **Ver11**: RoPE (Rotary Position Embedding) + optional FlashAttention

### Decoder Types
- **SeriesDecoder**: Per-timepoint predictions (T outputs per sample)
- **SingleTargetDecoder**: Single aggregated output
- **AveragedSeriesDecoder**: Time-averaged predictions (for subject-level tasks like sex/age)

### Loss Functions (`src/module/utils/learnable_losses.py`)
- Standard MSE, Focal MSE, Normalized Focal MSE, Tweedie Loss
- Per-emotion weighted losses, Uncertainty-weighted MSE

## SLURM Job Guidelines

```bash
#SBATCH --nodelist=node3          # Prefer node3 (node1 has NFS issues)
#SBATCH -t 72:00:00               # Minimum 24h, typically 48-72h
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB                # 40GB for regression, 20GB for classification
#SBATCH -o logs/%A-%x.o
```

- **num_workers**: Use 0-4 (≥8 causes shared memory errors)
- Always set WANDB_API_KEY before training with `--loggername wandb`

## Data Paths

- fMRI data: `/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/`
- Train/val/test splits: `data/splits/HBN/`
- Model outputs: `output/moviefmri/`
- Logs: `logs/`

## SVR Baselines

### Baseline Types
```
src/baselines/
├── svr_with_reduction.py   # Whole brain: pca, roi, time_avg modes
├── svr_with_glm_mask.py    # GLM sig voxels: glm_pca, glm_direct modes
└── nonzero_metrics.py      # Non-zero metrics calculator (matches pl_classifier.py)
```

### GLM Mask SVR (Task-Relevant Voxels)
```bash
# GLM sig voxels (~44k) -> PCA -> SVR
python src/train_svr_with_glm_mask.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --reduction_mode glm_pca \
    --pca_components 100 \
    --dataset_split_seed 2 \
    --stratified_params Age Sex \
    --output_dir output/svr_glm_pca_seq20

# GLM sig voxels -> time-average -> SVR (simpler)
python src/train_svr_with_glm_mask.py \
    --reduction_mode glm_direct \
    --output_dir output/svr_glm_direct_seq20
```

### GLM Mask Path
- Union mask: `/scratch/connectome/kimbo/GLM-Baseline-Test/results/full_analysis/smooth_motion/threshold_nonparam/sig_masks_for_ridge/union_mask.nii.gz`
- Shape: (81, 95, 81) - matches fMRI data exactly
- 44,303 significant voxels (~7.5% of brain)

### Non-Zero Metrics
Calculated exactly as in `pl_classifier.py`:
- `nonzero_mae`, `nonzero_mse`, `nonzero_rmse`, `nonzero_pearson`
- `detection_tpr`, `detection_f1`, `detection_auroc`
- Uses original scale with epsilon=1e-6

### Documentation
- Implementation guide: `docs/baselines/svr_glm_mask_implementation.md`
- SLURM scripts: `sample_scripts/svr_baselines/`

## Slash Commands

- `/check-jobs`: Check running SLURM jobs and recent logs
- `/cancel-job <ID>`: Cancel a SLURM job
- `/quick-test`: Quick import test before job submission
- `/baseline-status`: Check baseline experiment progress
