#!/bin/bash
#SBATCH --job-name=select_peaks
#SBATCH --output=select_emotion_peaks-%j.out
#SBATCH --error=select_emotion_peaks-%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "Selecting Emotion Peak Sequences"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run selection (for all test subjects, top 10 sequences per emotion)
/home/connectome/kimbo/.conda/envs/swiftio/bin/python select_emotion_peak_sequences.py \
    --run_id opr6oq97 \
    --top_k 10

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
