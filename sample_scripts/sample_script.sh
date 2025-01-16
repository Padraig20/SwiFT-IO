cd .. # move to where 'SwiFT is located'
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio
 
TRAINER_ARGS='--accelerator gpu --max_epochs 1 --precision 16 --num_nodes 1 --devices 1 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --dataset_name Dummy --image_path {image_path}'
DATA_ARGS='--batch_size 8 --num_workers 8 --input_type rest'
DEFAULT_ARGS='--project_name {neptune_project_name}'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA} --clf_head_version v1 --downstream_task sex --downstream_task_type classification' #--use_scheduler --gamma 0.5 --cycle 0.5' 
RESUME_ARGS=''

export NEPTUNE_API_TOKEN="{neptune API token}" # when using neptune as a logger

export CUDA_VISIBLE_DEVICES={GPU number}

python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 2 2 2 2 --window_size 4 4 4 4 --img_size 96 96 96 20 \
--patch_size 4 4 4 1 --num_classes 2 --num_targets 7 --decoder series_decoder #single_target_decoder or series_decoder