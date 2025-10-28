#!/bin/bash
#SBATCH --job-name=lstm_clf_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/lstm_clf_baseline_%j.log
#SBATCH --error=logs/lstm_clf_baseline_%j.err

# LSTM Baseline Training for Classification Task
# Match classification experiment: seq_length=20, seed=2

echo "=========================================="
echo "LSTM Baseline Training - Classification"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftv9

# Navigate to project directory
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Create output and log directories
mkdir -p output/moviefmri
mkdir -p logs

# Training parameters
IMAGE_PATH="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
SEQUENCE_LENGTH=20
DATASET_SPLIT_SEED=2
PROJECT_NAME="moviefmri"
EXPERIMENT_NAME="lstm_clf_baseline_seq20"

echo ""
echo "Configuration:"
echo "  Image path: $IMAGE_PATH"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Split seed: $DATASET_SPLIT_SEED"
echo "  Task type: classification"
echo "  Project: $PROJECT_NAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# Run training
python src/train_lstm_baseline.py \
    --seed 2 \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type classification \
    --loggername wandb \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --image_path $IMAGE_PATH \
    --input_type movieDM \
    --dataset_split_seed $DATASET_SPLIT_SEED \
    --sequence_length $SEQUENCE_LENGTH \
    --lstm_hidden_dim 256 \
    --lstm_num_layers 2 \
    --lstm_dropout 0.3 \
    --lstm_pooling adaptive \
    --lstm_pooled_dim 16 \
    --lstm_decoder_hidden 128 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --batch_size 16 \
    --eval_batch_size 32 \
    --num_workers 8 \
    --eval_num_workers 8 \
    --max_epochs 30 \
    --devices 1 \
    --num_nodes 1 \
    --accelerator gpu \
    --strategy auto \
    --precision 16-mixed \
    --train_split 0.7 \
    --val_split 0.15 \
    --stride_between_seq 1 \
    --stride_within_seq 1 \
    --num_targets 7 \
    --decoder lstm_classification_head \
    --model lstm_encoder

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
