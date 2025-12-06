#!/bin/bash
#SBATCH --job-name=zero_mask_fix
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j-zero_mask_fix.o
#SBATCH --error=logs/%j-zero_mask_fix.e

# Ver11 emotions regression with zero mask bug fix
# Based on 9x23kr7g settings
# Purpose: Check if nonzero metrics change after fixing the zero mask calculation

source ~/.bashrc
conda activate swift

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

TIMESTAMP=$(TZ='Asia/Seoul' date +%Y-%m-%d_%H-%M-%S)

python src/main.py \
    --downstream_task emotions \
    --downstream_task_type regression \
    --model swin4d_ver11 \
    --decoder series_decoder \
    --sequence_length 20 \
    --img_size 96 96 96 20 \
    --embed_dim 60 \
    --patch_size 6 6 6 1 \
    --window_size 4 4 4 20 \
    --first_window_size 4 4 4 20 \
    --depths 2 2 6 2 \
    --num_heads 3 6 12 24 \
    --c_multiplier 2 \
    --last_layer_full_MSA True \
    --num_targets 7 \
    --regression_loss_type normalized_focal_mse \
    --focal_gamma 1.5 \
    --batch_size 2 \
    --accumulate_grad_batches 4 \
    --learning_rate 5e-5 \
    --use_scheduler True \
    --max_epochs 60 \
    --seed 2 \
    --dataset_split_seed 2 \
    --stratified_params Age Sex \
    --label_scaling_method standardization \
    --use_flashattn False \
    --loggername wandb \
    --project_name moviefmri \
    --experiment_name "zero_mask_fix_${TIMESTAMP}"
