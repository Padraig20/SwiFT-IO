#!/bin/bash
#SBATCH --job-name=svr_glm_pca
#SBATCH --nodelist=node3
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH -o logs/%A-%x.o

# =============================================================================
# SVR with GLM Significant Voxel Mask + PCA
#
# This script trains SVR using GLM-identified significant voxels with PCA.
# Uses split_fixed_2_stratified_Age_Sex.txt for fair comparison.
#
# Features:
#   - GLM sig voxels (~44k) instead of whole brain (~590k)
#   - PCA for temporal information preservation
#   - Non-zero metrics matching pl_classifier.py
# =============================================================================

echo "=========================================="
echo "SVR with GLM Mask + PCA Training"
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
    --reduction_mode glm_pca \
    --pca_components 100 \
    --mask_type union \
    --kernel rbf \
    --C 1.0 \
    --epsilon 0.1 \
    --batch_size 4 \
    --eval_batch_size 16 \
    --num_workers 4 \
    --output_dir output/svr_glm_pca_seq20_stratified

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
