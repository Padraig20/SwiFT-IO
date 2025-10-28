#!/bin/bash
#SBATCH --job-name=lstm_test
#SBATCH --output=lstm_test_eval-%j.out
#SBATCH --error=lstm_test_eval-%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "LSTM Baseline - Test Set Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run evaluation (using full Python path, no conda activation needed)
/home/connectome/kimbo/.conda/envs/swiftio/bin/python evaluate_lstm_test.py \
    --checkpoint_path output/moviefmri/8tq0p4p2/lstm-epoch=10-valid_mse=3.1798.ckpt \
    --num_workers 0

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
