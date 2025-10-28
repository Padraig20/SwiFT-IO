#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=node4
#SBATCH --job-name=svc_cls_baseline
#SBATCH --output=logs/svc_cls_baseline-%j.out
#SBATCH --error=logs/svc_cls_baseline-%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# SVC Baseline Training for CLASSIFICATION
# Based on successful SVR regression baseline (run_svr_baseline.slurm)
#
# Expected runtime: ~12-16 hours
# Reasoning:
#   - Seq length: 20 (vs 30 for regression) → 33% less data per sample
#   - Classification: Binary SVC likely faster than epsilon-SVR
#   - Similar voxel count: 96^3 × 20 = ~1.77M features
#   - Estimated: 12-16h (vs 18-24h for regression with seq=30)

echo "=========================================="
echo "SVC Classification Baseline Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p output/svc_cls_baseline_seq20

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Node: $SLURM_NODELIST"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: 128GB"
echo ""

echo "=========================================="
echo "Configuration:"
echo "  Task: Classification (7 emotions, binary per emotion)"
echo "  Sequence length: 20 TRs"
echo "  Dataset split seed: 2 (match mc3r4vhf)"
echo "  Kernel: RBF"
echo "  C: 1.0"
echo "  Feature dim: 96×96×96×20 = 1,769,472 features"
echo "  Model: 7 separate SVC classifiers (one per emotion)"
echo "=========================================="
echo ""

echo "[$(date '+%H:%M:%S')] Starting SVC training..."

python src/train_svr_baseline.py \
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
    --kernel rbf \
    --C 1.0 \
    --standardize \
    --batch_size 8 \
    --eval_batch_size 16 \
    --num_workers 0 \
    --output_dir output/svc_cls_baseline_seq20 \
    --experiment_name svc_cls_seq20_seed2 \
    --train_split 0.7 \
    --val_split 0.15

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SVC Training SUCCEEDED!"
    echo "  Output: output/svc_cls_baseline_seq20/"
    echo "  Files:"
    echo "    - svr_model.pkl (trained SVC models)"
    echo "    - svr_metrics.json (train/val/test metrics)"
    echo "    - svr_config.json (configuration)"
else
    echo "✗ SVC Training FAILED (exit code: $EXIT_CODE)"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Print resource usage
echo ""
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,CPUTime
echo ""
