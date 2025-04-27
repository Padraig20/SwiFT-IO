#!/bin/bash
#SBATCH --job-name sub-13-18
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/logs/sub-13-18_%A.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/logs/sub-13-18_%A.o


# salloc --nodes=1 --qos=interactive --time=04:00:00 --constraint=gpu --gpus=1 --account=m4750_g -c 10
# bash sub-NDARCG073GZ6.slurm 2>&1 | tee logs/tubg3tim_sub-13-18_$(date +%Y%m%d_%H%M%S).out


source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

echo "===== Slurm Job Config ====="
echo "Job Name:      $SLURM_JOB_NAME"
echo "Job ID:        $SLURM_JOB_ID"
echo "GPU(s):        $SLURM_GPUS_PER_NODE"
echo "CPU core(s):   $SLURM_CPUS_PER_TASK"
echo "Time limit:    $SLURM_TIMELIMIT"
echo "============================"

cd /scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap

# 시작 시간 기록
start_time=$(date +%s)

python run_igmap_manual_mp_by_subject_2_initializer.py \
  --run_id tubg3tim \
  --subject_file sub-13-18.txt \
  --input_offset 3 \
  --seq_length 50 \
  --input_type movieDM \
  --n_jobs 6

# 종료 시간 기록
end_time=$(date +%s)

# 총 걸린 시간 계산
elapsed=$(( end_time - start_time ))

# HH:MM:SS 형식으로 출력
printf "✅ Done! Total elapsed time: %02d:%02d:%02d\n" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))