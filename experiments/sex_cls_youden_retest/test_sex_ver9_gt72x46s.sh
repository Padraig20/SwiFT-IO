#!/bin/bash
#SBATCH --job-name test_sex_v9_gt72x46s
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o

# =============================================================================
# TEST Sex Classification - Ver9 (gt72x46s) with Youden Optimal Threshold
# Ver9 baseline: patch [4,4,4,1], window [4,4,4,4], embed_dim=36
# =============================================================================

echo "=========================================="
echo "TEST Sex Classification - Ver9 (gt72x46s)"
echo "With Youden Optimal Threshold (new code)"
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

CHECKPOINT="output/moviefmri/gt72x46s/checkpt-epoch=00-valid_acc=0.80.ckpt"

echo ""
echo "Configuration:"
echo "  Model: swin4d_ver9"
echo "  Decoder: single_target_decoder"
echo "  Patch: [4,4,4,1], Window: [4,4,4,4]"
echo "  Checkpoint: $CHECKPOINT"
echo "  Mode: TEST ONLY with Youden threshold"
echo "=========================================="

TRAINER_ARGS="--accelerator gpu --precision 16 --num_nodes 1 --devices 1 --strategy ddp"
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 2 --eval_batch_size 2 --num_workers 4 --input_type movieDM --stratified_params Age'
DEFAULT_ARGS="--project_name moviefmri --experiment_name retest_sex_v9_gt72x46s_youden_$(date +'%Y%m%d_%H%M%S')"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task sex --downstream_task_type classification'

srun -N 1 -n 1 python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS \
--dataset_split_seed 2 --seed 2 --learning_rate 5e-5 --model swin4d_ver9 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 \
--patch_size 4 4 4 1 --num_classes 2 --num_targets 1 --decoder single_target_decoder \
--test_only --test_ckpt_path $CHECKPOINT

echo ""
echo "End Time: $(date)"
echo "=========================================="
