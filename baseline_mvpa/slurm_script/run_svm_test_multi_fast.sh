#!/bin/bash
#SBATCH --job-name=svm_fast            # 잡 이름
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G              # 총 160 GB
#SBATCH --time=12:00:00                # 최대 12 시간 (필요 시 ↑)
#SBATCH --nodelist=node4
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/logs/%A-%x.o

# salloc   --nodes=1   --ntasks=1   --cpus-per-task=16   --mem=160G --nodelist=node4
# ────────── 환경 세팅 ──────────
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio     # 사용하는 conda 환경 이름으로 변경

# ────────── 작업 디렉토리 이동 ──────────
cd /scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/baseline_mvpa

# ────────── 실행 & runtime 기록 ──────────
start_time=$(date +%s)

python -u svm_test_multi_fast.py \
       --input_type movieDM \
       --seq_length 50 \
       --input_offset 3

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "svm_test_multi_fast.py runtime: ${elapsed} sec ($(date -ud "@$elapsed" +%T))"
