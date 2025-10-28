#!/bin/bash
#SBATCH --job-name=svr_pca_eval
#SBATCH --output=output/svr_reduction_pca/slurm_eval_%j.out
#SBATCH --error=output/svr_reduction_pca/slurm_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=debug

echo "========================================="
echo "SVR PCA Evaluation (Sliding Window)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate swiftio

# Run evaluation
/home/connectome/kimbo/.conda/envs/swiftio/bin/python evaluate_svr_from_checkpoints.py \
    --checkpoint_dir output/svr_reduction_pca \
    --num_workers 8 \
    2>&1 | tee output/svr_reduction_pca/evaluation_slurm.log

echo ""
echo "Finished at: $(date)"
