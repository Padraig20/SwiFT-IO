#!/bin/bash
#SBATCH --job-name=svr_glm_direct
#SBATCH --nodelist=node3
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -o logs/%A-%x.o

# =============================================================================
# SVR with GLM Significant Voxel Mask + Direct (Time-Averaged)
#
# This script trains SVR using GLM-identified significant voxels
# with time-averaging (no PCA).
#
# Features:
#   - GLM sig voxels (~44k) as features
#   - Time-averaged (single static pattern per sample)
#   - Non-zero metrics matching pl_classifier.py
#   - Faster training than PCA version
# =============================================================================

echo "=========================================="
echo "SVR with GLM Mask + Direct Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Environment setup
source ~/.bashrc
conda activate swiftio

# Change to project directory
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run training
python src/train_svr_with_glm_mask.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type regression \
    --input_type movieDM \
    --dataset_split_seed 2 \
    --stratified_params Age Sex \
    --sequence_length 20 \
    --reduction_mode glm_direct \
    --mask_type union \
    --kernel rbf \
    --C 1.0 \
    --epsilon 0.1 \
    --batch_size 4 \
    --eval_batch_size 16 \
    --num_workers 4 \
    --output_dir output/svr_glm_direct_seq20_stratified

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
