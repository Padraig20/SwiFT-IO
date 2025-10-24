#!/bin/bash
#SBATCH --job-name=ig_10sec
#SBATCH --output=slurm_ig_10sec-%j.out
#SBATCH --error=slurm_ig_10sec-%j.err
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map with Baseline = First 10 Seconds"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run IG map with baseline=first 10 seconds using conda python directly
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_baseline_first10sec.py \
    --run_id opr6oq97 \
    --subject sub-NDARRL379BET \
    --n_jobs 1 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
