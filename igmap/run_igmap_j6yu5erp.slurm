#!/bin/bash
#SBATCH --job-name=igmap_j6yu5erp
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/igmap_j6yu5erp-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/igmap_j6yu5erp-%j.o

echo "=========================================="
echo "IG Map Generation for j6yu5erp"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Unbuffer output for real-time logging
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Configuration:"
echo "  Run ID: j6yu5erp"
echo "  Model: Swin4D v9"
echo "  Subject file: test_subjects.txt"
echo "  Instance level: No (averaged positive/negative)"
echo "  n_jobs: 1 (sequential processing for stability)"
echo "=========================================="
echo ""

echo "[$(date '+%H:%M:%S')] Starting IG map generation..."

python igmap/run_igmap_manual_by_subject_group.py \
    --run_id j6yu5erp \
    --subject_file test_subjects.txt \
    --n_jobs 1 \
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ IG Map generation SUCCEEDED!"
    echo "  Output: analysis/4_IGmap/results_each/j6yu5erp/"
else
    echo "✗ IG Map generation FAILED (exit code: $EXIT_CODE)"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Print resource usage
echo ""
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize
echo ""
