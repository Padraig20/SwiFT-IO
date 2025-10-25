#!/bin/bash
#SBATCH --job-name=ig_selective_test
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_selective_test-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_selective_test-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Selective Sequences (TEST)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Test with one subject, top 5 sequences per emotion, all 7 emotions
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \
    --run_id opr6oq97 \
    --subjects sub-NDARAH793FBF \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
