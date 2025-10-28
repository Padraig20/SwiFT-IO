#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --job-name=lstm_cls_baseline
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/lstm_cls_baseline-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/lstm_cls_baseline-%j.o
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx:6
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32

# LSTM Classification Baseline - Full Training
# Based on successful LSTM regression baseline (run_lstm_baseline.slurm)
#
# Expected runtime: ~18-22 hours
# Reasoning:
#   - Seq length: 20 (vs 30 for regression) → 33% less temporal data
#   - Classification: Simpler loss (CrossEntropy vs MSE), but per-class head
#   - Max epochs: 30 (same as regression)
#   - Early stopping likely to trigger around epoch 15-20
#   - GPUs: 6 RTX (same as regression baseline)
#   - Estimated: 18-22h (regression took ~6-8h for 30 epochs)

echo "=========================================="
echo "LSTM Classification Baseline Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Create necessary directories
mkdir -p logs
mkdir -p output/moviefmri

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Unbuffer output for real-time logging
export PYTHONUNBUFFERED=1

# Wandb API key
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

echo "=========================================="
echo "Configuration:"
echo "  Task: Classification (7 emotions, binary per emotion)"
echo "  Max epochs: 30"
echo "  Encoder: LSTM (2 layers, hidden_dim=256)"
echo "  Spatial pooling: Adaptive (16^3 = 4096)"
echo "  Decoder: Classification head (binary per emotion)"
echo "  GPUs: 6 RTX"
echo "  Batch size: 24 (4 per GPU)"
echo "  Strategy: DDP"
echo "  Dataset split: seed 2 (match mc3r4vhf)"
echo "  Sequence length: 20 TRs"
echo "  Optimizer: AdamW (lr=1e-4, wd=0.01)"
echo "  Scheduler: CosineAnnealingWarmUpRestarts (cycle=0.3)"
echo "=========================================="
echo ""

echo "[$(date '+%H:%M:%S')] Starting training..."

python -u src/train_lstm_baseline.py \
    --model lstm_encoder \
    --decoder lstm_classification_head \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type classification \
    --input_type movieDM \
    --seed 2 \
    --dataset_split_seed 2 \
    --stratified_params Sex Age \
    --img_size 96 96 96 20 \
    --sequence_length 20 \
    --stride_between_seq 1 \
    --stride_within_seq 1 \
    --num_targets 7 \
    --lstm_hidden_dim 256 \
    --lstm_num_layers 2 \
    --lstm_dropout 0.3 \
    --lstm_pooling adaptive \
    --lstm_pooled_dim 16 \
    --lstm_decoder_hidden 128 \
    --batch_size 24 \
    --eval_batch_size 24 \
    --num_workers 4 \
    --eval_num_workers 4 \
    --learning_rate 1e-4 \
    --optimizer AdamW \
    --weight_decay 0.01 \
    --use_scheduler \
    --cycle 0.3 \
    --max_epochs 30 \
    --precision 32 \
    --devices 6 \
    --num_nodes 1 \
    --strategy ddp \
    --loggername wandb \
    --project_name moviefmri \
    --experiment_name lstm_cls_baseline_seq20_seed2 \
    --label_scaling_method standardization \
    --train_split 0.7 \
    --val_split 0.15

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training SUCCEEDED!"
    echo "  Output: output/moviefmri/"
    echo ""
    echo "Results summary:"
    echo "  - Train/Val/Test metrics logged to wandb"
    echo "  - Best checkpoint saved based on validation accuracy"
    echo "  - Checkpoints: checkpt-epoch=XX-valid_acc=X.XX.ckpt"
else
    echo "✗ Training FAILED (exit code: $EXIT_CODE)"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Print resource usage
echo ""
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,CPUTime
echo ""
