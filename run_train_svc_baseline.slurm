#!/bin/bash
#SBATCH --job-name=svc_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/svc_baseline_%j.log
#SBATCH --error=logs/svc_baseline_%j.err

# SVC Baseline Training for Classification Task
# Match classification experiment: seq_length=20, seed=2

echo "=========================================="
echo "SVC Baseline Training - Classification"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftv9

# Navigate to project directory
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Create output directories
mkdir -p output/svc_baseline_seq20
mkdir -p logs

# Training parameters
IMAGE_PATH="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
SEQUENCE_LENGTH=20
DATASET_SPLIT_SEED=2
OUTPUT_DIR="output/svc_baseline_seq20"

echo ""
echo "Configuration:"
echo "  Image path: $IMAGE_PATH"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Split seed: $DATASET_SPLIT_SEED"
echo "  Task type: classification"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Run training
python src/train_svr_baseline.py \
    --image_path $IMAGE_PATH \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type classification \
    --task_type classification \
    --input_type movieDM \
    --dataset_split_seed $DATASET_SPLIT_SEED \
    --sequence_length $SEQUENCE_LENGTH \
    --kernel rbf \
    --C 1.0 \
    --standardize \
    --batch_size 4 \
    --eval_batch_size 16 \
    --num_workers 16 \
    --stride_between_seq 1 \
    --stride_within_seq 1 \
    --output_dir $OUTPUT_DIR \
    --train_split 0.7 \
    --val_split 0.15

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
