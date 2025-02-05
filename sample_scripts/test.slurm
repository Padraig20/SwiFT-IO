#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -A m4727_g # project to charge for this job
#SBATCH -q regular # quality of service (regular, debug, ...)
#SBATCH -J HBN_emotion # name of job
#SBATCH -t 15:00:00 # max walltime
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

cd /data/scratch/kimbo/SwiFT-IO # move to where 'SwiFT is located'
# conda init
# source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /data/scratch/kimbo/.conda/myenv 
#conda init # doesnt work anyways
#conda activate SwiFT-BERT # make sure it's activated before already
 
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