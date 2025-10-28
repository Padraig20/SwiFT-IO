#!/bin/bash
#SBATCH --job-name=ig_happy
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_happy_top5-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_happy_top5-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Happy Top 5 Performers"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process top 5 subjects for Happy
# Only process peak sequences for Happy (--emotions Happy)
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \
    --run_id opr6oq97 \
    --subjects sub-NDARYR994ELB sub-NDARCJ246FJB sub-NDARZD415ZZ1 sub-NDARKZ479FC8 sub-NDARVF012BD2 \
    --emotions Happy \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Happy processing complete"
else
    echo "❌ Happy processing FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
