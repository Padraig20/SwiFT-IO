#!/bin/bash
#SBATCH --job-name=ig_top5_%a
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_top5_%a-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_top5_%a-%j.o
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --array=1-5

echo "=========================================="
echo "IG Map - Top 5 Subject (Parallel)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Read top 5 subjects file
TOP5_FILE="analysis/4_IGmap/subject_performance/opr6oq97_top5_subjects.txt"

if [ ! -f "$TOP5_FILE" ]; then
    echo "❌ Error: Top 5 subjects file not found: $TOP5_FILE"
    echo "Please run MSE evaluation first to generate this file."
    exit 1
fi

# Get the subject for this array task
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TOP5_FILE")

if [ -z "$SUBJECT" ]; then
    echo "❌ Error: Could not read subject for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo ""
echo "Processing Subject: $SUBJECT"
echo ""

# Run IG map for this subject
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \
    --run_id opr6oq97 \
    --subjects "$SUBJECT" \
    --top_k_seqs 5 \
    --n_steps 20 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Subject $SUBJECT completed successfully!"
else
    echo "❌ Subject $SUBJECT failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
