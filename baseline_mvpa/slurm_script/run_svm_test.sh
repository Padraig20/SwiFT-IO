#!/bin/bash
#SBATCH --job-name=svm_singles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00
#SBATCH --nodelist=node3
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%x-%A.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%x-%A.o

#  환경 세팅
source /usr/anaconda3/etc/profile.d/conda.sh
source activate swiftio  # 사용하는 conda 환경 이름으로 변경

# 작업 디렉토리로 이동 (필요시 변경)
cd /scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/baseline_mvpa

# 파이썬 코드 실행 전
start_time=$(date +%s)

python svm_test.py

# 실행 후
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "svm_test.py runtime: ${elapsed} sec ($(date -ud "@$elapsed" +%T))"
