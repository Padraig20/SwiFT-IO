#!/bin/bash
#SBATCH --job-name=time_avg_pca
#SBATCH --output=output/svr_time_avg_pca_100/slurm_train_%j.out
#SBATCH --error=output/svr_time_avg_pca_100/slurm_train_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=debug

echo "========================================="
echo "Time-Avg + PCA SVR Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate swiftio

# Run training
/home/connectome/kimbo/.conda/envs/swiftio/bin/python train_svr_time_avg_with_pca.py \
    --checkpoint_path output/svr_reduction_time_avg_linear/train_data_checkpoint.pkl \
    --pca_components 100 \
    --kernel rbf \
    --output_dir output/svr_time_avg_pca_100 \
    --num_workers 8 \
    2>&1 | tee output/svr_time_avg_pca_100/training_slurm.log

echo ""
echo "Finished at: $(date)"
