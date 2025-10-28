#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=node2
#SBATCH --job-name=svc_pca_cls
#SBATCH --output=logs/svc_pca_cls-%j.out
#SBATCH --error=logs/svc_pca_cls-%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# SVC PCA-based Classification Baseline
# Based on successful SVR PCA regression baseline
#
# Expected runtime: ~7-9 hours
# Reasoning:
#   - Seq length: 20 (vs 30 for regression) → 33% less data
#   - PCA reduction: 100 components × 20 TRs = 2,000 features
#   - PCA fitting: ~1-2h for IncrementalPCA on train set
#   - SVC training: ~5-7h on reduced features
#   - Original PCA regression took ~3-4h for seq=30
#   - Estimated: 7-9h for seq=20 classification

echo "=========================================="
echo "SVC PCA-based Classification Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Properly initialize conda
source /home/connectome/kimbo/.bashrc
eval "$(conda shell.bash hook)"
conda activate swiftio

# Verify conda environment
echo "Conda environment: $CONDA_DEFAULT_ENV"
which python

# Install required package if missing
pip install iterative-stratification -q

# Create necessary directories
mkdir -p logs
mkdir -p output/svc_reduction_pca_cls_seq20

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo ""

# Unbuffer output for real-time logging
export PYTHONUNBUFFERED=1

# Set number of threads for CPU parallelization
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo "=========================================="
echo "Training Configuration:"
echo "  Task: Classification (7 emotions)"
echo "  Reduction method: PCA"
echo "  PCA components per timepoint: 100"
echo "  Feature dimension: 2,000 (20 × 100)"
echo "  Temporal info: PRESERVED"
echo "  Sequence length: 20 TRs"
echo "  Dataset split seed: 2"
echo "=========================================="
echo ""

echo "[$(date '+%H:%M:%S')] Starting training..."

# Note: train_svr_with_reduction.py needs to be updated to support classification
python -u src/train_svr_with_reduction.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type classification \
    --task_type classification \
    --input_type movieDM \
    --dataset_split_seed 2 \
    --sequence_length 20 \
    --stride_between_seq 1 \
    --stride_within_seq 1 \
    --reduction_method pca \
    --pca_components 100 \
    --kernel rbf \
    --C 1.0 \
    --standardize \
    --batch_size 8 \
    --eval_batch_size 8 \
    --num_workers 0 \
    --output_dir output/svc_reduction_pca_cls_seq20 \
    --experiment_name svc_pca_cls \
    --train_split 0.7 \
    --val_split 0.15

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training SUCCEEDED!"
    echo "  Output: output/svc_reduction_pca_cls_seq20/"
else
    echo "✗ Training FAILED (exit code: $EXIT_CODE)"
    echo "  Note: train_svr_with_reduction.py may need classification support"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Print resource usage
echo ""
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize
echo ""
