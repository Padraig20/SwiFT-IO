#!/bin/bash
#SBATCH --job-name 3
#SBATCH -t 48:00:00
#SBATCH --nodes 1
#SBATCH --nodelist node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/sub-3_%A.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/sub-3_%A.o

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

python run_igmap_manual_by_subject_group.py \
  --run_id tubg3tim \
  --subject_file sub-3.txt 
# 종료 시간 기록
end_time=$(date +%s)

# 소요 시간 계산
elapsed=$(( end_time - start_time ))

# 소요시간을 HH:MM:SS 형태로 출력
printf "✅ Done! Total elapsed time: %02d:%02d:%02d\n" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))
