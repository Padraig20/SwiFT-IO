#!/bin/bash
#SBATCH --job-name=ig_sad_r4
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_sad_rank4-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_sad_rank4-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Sad Rank 4: sub-NDARTW889JN9"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process single subject for Sad
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \
    --run_id opr6oq97 \
    --subjects sub-NDARTW889JN9 \
    --emotions Sad \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Sad Rank 4 (sub-NDARTW889JN9) complete"
else
    echo "❌ Sad Rank 4 (sub-NDARTW889JN9) FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
