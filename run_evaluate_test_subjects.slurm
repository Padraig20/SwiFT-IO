#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --output=eval_test_subjects-%j.out
#SBATCH --error=eval_test_subjects-%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "Evaluating Test Set Subjects"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run evaluation
/home/connectome/kimbo/.conda/envs/swiftio/bin/python evaluate_test_subjects.py \
    --run_id opr6oq97

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
