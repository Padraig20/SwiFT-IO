#!/bin/bash
#
# Full test set evaluation of stratified metrics
# Tests on existing best checkpoint (opr6oq97, epoch 7)
# This will run on the ENTIRE test set to get accurate metrics
#

echo "=========================================="
echo "Quick Test: Stratified Metrics"
echo "Model: fd865zrm (epoch 17, seq_len=20)"
echo "Start Time: $(date)"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Verify environment
echo ""
echo "Environment Check:"
python -c "import torch; print('  PyTorch:', torch.__version__)"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
echo ""

# Set wandb environment
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

# Test configuration
CHECKPOINT_PATH="output/moviefmri/fd865zrm/checkpt-epoch=17-valid_mse=0.15.ckpt"

echo "Test Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Test batches: FULL TEST SET (all batches)"
echo "  Expected duration: ~20-30 minutes"
echo ""

# Construct arguments from original training script
TRAINER_ARGS="--accelerator gpu --precision 16 --num_nodes 1 --devices 1"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM'
DEFAULT_ARGS="--project_name moviefmri"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression'
RESUME_ARGS='--adjust_hrf --input_offset 0'
MODEL_ARGS='--model swin4d_ver9 --depth 2 2 6 2 --embed_dim 36'
ARCH_ARGS='--sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20'
DECODER_ARGS='--patch_size 4 4 4 1 --num_classes 1 --num_targets 7 --decoder series_decoder'

echo "Running test..."
echo "=========================================="

python src/main.py \
    --test_only \
    --test_ckpt_path $CHECKPOINT_PATH \
    $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS $MODEL_ARGS $ARCH_ARGS $DECODER_ARGS \
    --dataset_split_seed 2 --seed 2

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo ""
    echo "📊 Check wandb for new metrics:"
    echo "   - test_nonzero_mae_{emotion}"
    echo "   - test_nonzero_pearson_{emotion}"
    echo "   - test_avg_nonzero_mae"
    echo "   - test_avg_nonzero_pearson"
    echo ""
    echo "🔗 Wandb: https://wandb.ai/<your-entity>/moviefmri/runs/opr6oq97"
else
    echo "❌ Test failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
