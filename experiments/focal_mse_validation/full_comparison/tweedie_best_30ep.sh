#!/bin/bash
#SBATCH --job-name tweedie_p1.5_30ep
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o


echo "=========================================="
echo "Full Comparison: Tweedie Loss (BEST)"
echo "Configuration: p=1.5, lr=3e-5, 30 epochs"
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
export MASTER_PORT=52846

# wandb 환경변수
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

EXPERIMENT_NAME="full_tweedie_p1.5_lr3e-5_30ep_${SLURM_JOB_ID}"


TRAINER_ARGS="--accelerator gpu --max_epochs 30 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 4"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age Sex'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression --use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='--adjust_hrf --input_offset 0'
LOSS_ARGS='--regression_loss_type tweedie --tweedie_p 1.5'


echo ""
echo "=========================================="
echo "Training Configuration:"
echo "  Loss Type: Tweedie Loss"
echo "  Tweedie p: 1.5 (BEST)"
echo "  Learning Rate: 3e-5"
echo "  Max Epochs: 30"
echo "  Batch Size: 2"
echo "  Gradient Accumulation: 4"
echo "  Sequence Length: 20"
echo "  Model: swin4d_ver9"
echo "  Purpose: Fair comparison to baseline"
echo "=========================================="
echo ""

srun -N 1 -n 1 bash -c "
python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS $LOSS_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 3e-5 --model swin4d_ver9 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 \
--patch_size 4 4 4 1 --num_classes 1 --num_targets 7 --decoder series_decoder
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Full Training (Tweedie p=1.5, 30 epochs) completed successfully!"
else
    echo "❌ Full Training failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
