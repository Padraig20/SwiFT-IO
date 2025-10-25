#!/bin/bash
#SBATCH --job-name=ig_fea_r1
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_fear_rank1-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_fear_rank1-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Fear Rank 1: sub-NDARDU853XZ6"
echo "Baseline: first_10sec"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process single subject for Fear with first_10sec baseline
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_baseline_comparison.py \
    --run_id opr6oq97 \
    --subjects sub-NDARDU853XZ6 \
    --baseline first_10sec \
    --emotions Fear \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Fear Rank 1 (sub-NDARDU853XZ6) complete"
else
    echo "❌ Fear Rank 1 (sub-NDARDU853XZ6) FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
