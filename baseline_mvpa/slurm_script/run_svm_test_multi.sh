#!/bin/bash
#SBATCH --job-name=svm_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --nodelist=node3
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o

#  환경 세팅
source /usr/anaconda3/etc/profile.d/conda.sh
source activate swiftio  # 사용하는 conda 환경 이름으로 변경

# 작업 디렉토리로 이동 (필요시 변경)
cd /scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/baseline_mvpa

# 파이썬 코드 실행
python svm_test_multi.py --input_type movieDM --seq_length 50 --input_offset 3
