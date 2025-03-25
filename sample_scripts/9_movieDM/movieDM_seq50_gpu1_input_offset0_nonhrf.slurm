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
#SBATCH -t 24:00:00


echo "GPUs for this job: $SLURM_JOB_GPUS"
echo "GPUs allocated: $SLURM_GPUS_PER_NODE"
echo "Tasks allocated: $SLURM_NTASKS"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"

cd /data/scratch/kimbo/SwiFT-IO # move to where 'SwiFT is located'
source /etc/profile.d/conda.sh
# conda activate myenv
conda activate /data/scratch/kimbo/.conda/swiftio_aica

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"  # GPU 개수 출력

# export NCCL_DEBUG=INFO # NCCL 통신 디버깅 로그 활성화: NCCL 관련 문제 발생 시 확인
# export NCCL_P2P_DISABLE=1 # GPU 간 P2P(Direct) 통신 비활성화	
# export NCCL_IB_DISABLE=1
# export PYTHONFAULTHANDLER=1

# export CUDA_VISIBLE_DEVICES=3

# 필요한 환경 변수 설정
export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=20750
export WANDB_API_KEY="ce3d72b255c21a2b99cd48915d63b62d36a17828"

EXPERIMENT_NAME="${SLURM_JOB_ID}_movieDM_offset0_seq30_nonhrf_$(date +'%Y-%m-%d_%H-%M-%S')"

TRAINER_ARGS="--accelerator gpu --max_epochs 80 --precision 16 --num_nodes 1 --devices 1 --strategy ddp --accumulate_grad_batches 2" # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername wandb --dataset_name HBN --image_path /data/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120'
DATA_ARGS='--batch_size 4 --eval_batch_size 4 --num_workers 4 --input_type movieDM'
DEFAULT_ARGS="--project_name moviefmri --experiment_name $EXPERIMENT_NAME"
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA --clf_head_version v1 --downstream_task emotions --downstream_task_type regression' #--use_scheduler --gamma 0.5 --cycle 0.5' 
RESUME_ARGS='--input_offset 0'
 
# TODO: srun으로 수정 
srun -N 1 -n 1 bash -c "
python src/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 50 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 50 \
--patch_size 4 4 4 1 --num_classes 1 --num_targets 7 --decoder series_decoder #single_target_decoder # series_decoder # --augment_during_training"