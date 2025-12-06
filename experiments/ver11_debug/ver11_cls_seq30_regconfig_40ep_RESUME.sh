#!/bin/bash
#SBATCH --job-name ver11_cls_s30_RESUME
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o


echo "=========================================="
echo "RESUME: Ver11 Classification Seq30 (Regression Config)"
echo "Resuming from: output/moviefmri/i6kq8vu0/last.ckpt"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

echo "확인된 GPU 리스트:"
nvidia-smi -L

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Verify environment
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=52861

# wandb 환경변수
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

# Use same wandb run ID to continue logging
export WANDB_RUN_ID="i6kq8vu0"
export WANDB_RESUME="allow"

EXPERIMENT_NAME="ver11_cls_s30_regcfg_40ep_${SLURM_JOB_ID}"
CHECKPOINT_PATH="output/moviefmri/i6kq8vu0/last.ckpt"

TRAINER_ARGS="--accelerator gpu --max_epochs 40 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 4"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age Sex'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type classification --use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS="--adjust_hrf --input_offset 0 --resume_ckpt_path $CHECKPOINT_PATH"


echo ""
echo "=========================================="
echo "🔄 RESUME Configuration:"
echo "  Resuming from Epoch: 13"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  WandB Run ID: i6kq8vu0"
echo "  Target Epochs: 40 (27 epochs remaining)"
echo ""
echo "  Task Type: CLASSIFICATION"
echo "  Sequence Length: 30"
echo "  Batch Size: 2"
echo "  Learning Rate: 5e-5"
echo "  Model: swin4d_ver11"
echo ""
echo "  📊 Changes from original run:"
echo "  - wandb artifact upload DISABLED (to avoid disk space issues)"
echo "  - Continuing same wandb run for seamless tracking"
echo "=========================================="
echo ""

srun -N 1 -n 1 python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 5e-5 --model swin4d_ver11 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 30 --first_window_size 4 4 4 30 --window_size 4 4 4 30 --img_size 96 96 96 30 \
--patch_size 6 6 6 1 --num_classes 2 --num_targets 7 --decoder series_decoder

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Ver11 Classification Seq30 RESUMED and completed!"
    echo ""
    echo "📝 Next Steps:"
    echo "  1. Check WandB: https://wandb.ai/snu-connectome/moviefmri/runs/i6kq8vu0"
    echo "  2. Compare final results with Ver11 regression"
    echo "  3. Analyze if Ver11 encoder or regression head has issues"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "End Time: $(date)"
echo "=========================================="
