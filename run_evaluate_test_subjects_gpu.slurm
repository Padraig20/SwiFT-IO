#!/bin/bash
#SBATCH --job-name=eval_test_gpu
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/eval_test_gpu-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/eval_test_gpu-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

echo "=========================================="
echo "MSE Evaluation (GPU-accelerated)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo ""
echo "Configuration:"
echo "  GPU: 1 (RTX)"
echo "  Batch size: 4"
echo "  Num workers: 2"
echo "  Memory: 64G"
echo "  Expected time: ~30-40 minutes"
echo ""

/home/connectome/kimbo/.conda/envs/swiftio/bin/python evaluate_test_subjects_gpu.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation SUCCEEDED!"
else
    echo "❌ Evaluation FAILED (exit code: $EXIT_CODE)"
fi
echo "End Time: $(date)"
echo "=========================================="
