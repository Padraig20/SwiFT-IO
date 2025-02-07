#!/bin/bash
#SBATCH -N 1                       # 노드 1개 요청
#SBATCH -c 8                      # CPU 코어 32개 요청 (GPU당 8개 코어 기준)
#SBATCH --ntasks-per-node=1      # 노드당 2개의 task
#SBATCH --gpus-per-node=1          # 노드당 GPU 2개 요청
#SBATCH --mem-per-cpu=8G 
#SBATCH --chdir='/data/scratch/kimbo/SwiFT-IO'
#SBATCH --nodelist=gpu-1
#SBATCH --output=/data/scratch/kimbo/SwiFT-IO/logs/R-%j-%x.out
#SBATCH --error=/data/scratch/kimbo/SwiFT-IO/logs/R-%j-%x.out
#SBATCH -t 10:00:00


echo "GPUs for this job: $SLURM_JOB_GPUS"
echo "GPUs allocated: $SLURM_GPUS_PER_NODE"
echo "Tasks allocated: $SLURM_NTASKSs"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"

cd /data/scratch/kimbo/SwiFT-IO # move to where 'SwiFT is located'
# conda init
# source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /data/scratch/kimbo/.conda/myenv 

TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --dataset_name HBN --image_path /data/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 8 --eval_batch_size 8 --num_workers 8 --input_type rest'
DEFAULT_ARGS='--project_name movie-nersc'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression' #--use_scheduler --gamma 0.5 --cycle 0.5' 
RESUME_ARGS='--adjust_hrf'

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZTQ2NDY2ZS1mNGUwLTQ3YjQtYjVjNC02NWYzZTgwZjlkOTkifQ==" # when using neptune as a logger
 
 
python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 50 --first_window_size 2 2 2 2 --window_size 4 4 4 4 --img_size 96 96 96 50 \
--patch_size 4 4 4 1 --num_classes 1 --num_targets 7 --decoder series_decoder #single_target_decoder # series_decoder # --augment_during_training