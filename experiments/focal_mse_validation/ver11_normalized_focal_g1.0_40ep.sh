#!/bin/bash
#SBATCH --job-name ver11_norm_focal_g1.0_40ep
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o


echo "=========================================="
echo "Ver11 + Normalized Focal MSE Loss (Fair Comparison)"
echo "Configuration: gamma=1.0, lr=3e-5, 40 epochs"
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
export MASTER_PORT=52853

# wandb 환경변수
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

EXPERIMENT_NAME="ver11_norm_focal_g1.0_40ep_${SLURM_JOB_ID}"


TRAINER_ARGS="--accelerator gpu --max_epochs 40 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 4"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age Sex'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression --use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='--adjust_hrf --input_offset 0'
LOSS_ARGS='--regression_loss_type normalized_focal_mse --focal_gamma 1.0 --focal_eps 1e-6'


echo ""
echo "=========================================="
echo "Training Configuration:"
echo "  🎯 FAIR COMPARISON TO BASELINE"
echo "  Task Type: REGRESSION"
echo "  Loss Type: Normalized Focal MSE (gamma=1.0) ⭐"
echo "  Max Epochs: 40 (same as baseline)"
echo "  Sequence Length: 20"
echo "  Batch Size: 2"
echo "  Learning Rate: 3e-5 (Phase 1C best)"
echo "  Model: swin4d_ver11 ✨"
echo ""
echo "  Architecture (Ver11):"
echo "  Patch Size: [6, 6, 6, 1]"
echo "  Window Size: [4, 4, 4, 20] (full temporal!) 🌟"
echo "  Latents: 160"
echo "  Temporal dependency: MAXIMIZED"
echo ""
echo "  Seeds: dataset_split_seed=2, seed=2"
echo "  Stratified Split: Age, Sex ✅"
echo "  WandB Project: moviefmri"
echo ""
echo "  📊 Purpose: Fair comparison with baseline MSE"
echo "  Expected: Ver11 temporal + Normalized Focal scale-robust"
echo "  Key Advantage: Scale-invariant (Positive 0-27, Sad 0-5)"
echo "=========================================="
echo ""

srun -N 1 -n 1 python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS $LOSS_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 3e-5 --model swin4d_ver11 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 4 4 4 20 --window_size 4 4 4 20 --img_size 96 96 96 20 \
--patch_size 6 6 6 1 --num_classes 1 --num_targets 7 --decoder series_decoder

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Ver11 + Normalized Focal MSE training completed successfully!"
    echo ""
    echo "📝 Next Steps:"
    echo "  1. Check WandB: https://wandb.ai/snu-connectome/moviefmri"
    echo "  2. Compare with baseline MSE (Job 63946)"
    echo "  3. Compare with Ver11 + Focal & Tweedie"
    echo "  4. Analyze scale-robust performance across emotions"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "End Time: $(date)"
echo "=========================================="
