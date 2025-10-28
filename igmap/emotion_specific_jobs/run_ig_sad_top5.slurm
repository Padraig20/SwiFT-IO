#!/bin/bash
#SBATCH --job-name=ig_sad
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_sad_top5-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_sad_top5-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Sad Top 5 Performers"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process top 5 subjects for Sad
# Only process peak sequences for Sad (--emotions Sad)
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \
    --run_id opr6oq97 \
    --subjects sub-NDARTK357VHL sub-NDARUC771VM5 sub-NDARHT829KKB sub-NDARTW889JN9 sub-NDARYR994ELB \
    --emotions Sad \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Sad processing complete"
else
    echo "❌ Sad processing FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
