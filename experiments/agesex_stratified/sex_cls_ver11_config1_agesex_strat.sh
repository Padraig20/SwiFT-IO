#!/bin/bash
#SBATCH --job-name sex_v11_cfg1
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o

# =============================================================================
# Sex Classification - Ver11 Config 1 (Full Temporal) with Age+Sex Stratified
# Config 1: patch [6,6,6,1], window [4,4,4,20], embed_dim=60
# Reference: mwatswk4 - better balanced accuracy
# =============================================================================

echo "=========================================="
echo "Sex Classification - Ver11 Config 1 (Age+Sex Stratified)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=$((RANDOM % 10000 + 50000))

export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"

EXPERIMENT_NAME="${SLURM_JOB_ID}_sex_cls_v11_cfg1_agesex_strat_$(date +'%Y-%m-%d_%H-%M-%S')"

echo ""
echo "Configuration:"
echo "  Model: swin4d_ver11 (Config 1 - full temporal)"
echo "  Decoder: single_target_decoder"
echo "  Patch: [6,6,6,1], Window: [4,4,4,20]"
echo "  Embed Dim: 60"
echo "  Stratified: Age + Sex"
echo "  Task: Sex Classification"
echo "=========================================="

TRAINER_ARGS="--accelerator gpu --max_epochs 40 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 4"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age Sex'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task sex --downstream_task_type classification --use_scheduler --gamma 0.5 --cycle 0.5'

srun -N 1 -n 1 python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 5e-5 --model swin4d_ver11 --depth 2 2 6 2 --embed_dim 60 \
--sequence_length 20 --first_window_size 4 4 4 20 --window_size 4 4 4 20 --img_size 96 96 96 20 \
--patch_size 6 6 6 1 --num_classes 2 --num_targets 1 --decoder single_target_decoder

echo ""
echo "End Time: $(date)"
echo "=========================================="
