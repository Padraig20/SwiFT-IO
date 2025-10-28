#!/bin/bash
#SBATCH --job-name=time_avg_pca_eval
#SBATCH --output=output/svr_time_avg_pca_100/slurm_eval_%j.out
#SBATCH --error=output/svr_time_avg_pca_100/slurm_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=debug

echo "========================================="
echo "Time-Avg + PCA SVR Evaluation"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Run evaluation (using full Python path, no conda activation needed)
/home/connectome/kimbo/.conda/envs/swiftio/bin/python evaluate_svr_time_avg_pca.py \
    --checkpoint_dir output/svr_time_avg_pca_100 \
    --num_workers 0 \
    2>&1 | tee output/svr_time_avg_pca_100/evaluation_log.txt

echo ""
echo "Finished at: $(date)"
