#!/bin/bash
#SBATCH --job-name ver11_emo_normfocal_c1
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o


echo "=========================================="
echo "Ver11 Emotions Regression - Config 1 + Normalized Focal MSE"
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

python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=52848

export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"
export WANDB_ANONYMOUS="allow"

EXPERIMENT_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}_$(date +'%Y-%m-%d_%H-%M-%S')"

TRAINER_ARGS="--accelerator gpu --max_epochs 40 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 4"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age Sex'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression --use_scheduler --gamma 0.5 --cycle 0.5'
LOSS_ARGS='--regression_loss_type normalized_focal_mse --focal_gamma 1.5 --focal_eps 1e-6'
RESUME_ARGS='--adjust_hrf --input_offset 0'

echo ""
echo "=========================================="
echo "Configuration:"
echo "  Task: Emotions Regression"
echo "  Model: swin4d_ver11"
echo "  Config: 1 (Full temporal resolution)"
echo "  Decoder: series_decoder"
echo "  Loss: Normalized Focal MSE (gamma=1.5, eps=1e-6) - Best from phase1a comparison"
echo "  Embed Dim: 60 (60/3=20 per head)"
echo "  Batch Size: 2"
echo "  Sequence Length: 20"
echo "  Patch Size: [6, 6, 6, 1] - Full temporal resolution"
echo "  Window Size: [4, 4, 4, 20] - FULL temporal sequence"
echo "  Num Targets: 7 (Anger, Happy, Fear, Sad, Excited, Positive, Negative)"
echo "  Dataset Split Seed: 2 (same as fxgvztr4)"
echo "  Seed: 2"
echo "  Stratified: Age, Sex (same as fxgvztr4)"
echo "=========================================="
echo ""

srun -N 1 -n 1 bash -c "
python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $LOSS_ARGS $RESUME_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 5e-5 --model swin4d_ver11 --depth 2 2 6 2 --embed_dim 60 \
--sequence_length 20 --first_window_size 4 4 4 20 --window_size 4 4 4 20 --img_size 96 96 96 20 \
--patch_size 6 6 6 1 --num_classes 1 --num_targets 7 --decoder series_decoder
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
