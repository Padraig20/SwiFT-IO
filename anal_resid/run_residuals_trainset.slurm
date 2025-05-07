#!/bin/bash
#SBATCH --job-name=resid_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --nodelist=node4
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o

# Conda 초기화
source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh

# 환경 명시적으로 활성화 (full path 사용)
conda activate /home/connectome/kimbo/.conda/envs/swiftio

# 환경 체크 (디버깅용)
echo "Using Python from: $(which python)"
pip list | grep wandb

# 작업 디렉토리 이동
cd /scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/src

start_time=$(date +%s)

python -u derive_residuals.py \
    --run_id pqslufat \
    --input_type movieDM \
    --seq_length 30 \
    --input_offset 3 \
    --dataset train

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "derive_residuals.py runtime: ${elapsed} sec ($(date -ud "@$elapsed" +%T))"
