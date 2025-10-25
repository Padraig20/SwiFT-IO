#!/bin/bash
#SBATCH --job-name=ig_bl_10sec
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_baseline_first10sec-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_baseline_first10sec-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Baseline Comparison - FIRST 10 SECONDS"
echo "Subject: sub-NDARMZ366UY8 (Positive top 1)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process Positive top 1 with baseline=first_10sec
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_baseline_comparison.py \
    --run_id opr6oq97 \
    --subjects sub-NDARMZ366UY8 \
    --baseline first_10sec \
    --emotions Positive \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Baseline FIRST 10 SEC complete"
else
    echo "❌ Baseline FIRST 10 SEC FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
