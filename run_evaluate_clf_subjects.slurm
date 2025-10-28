#!/bin/bash
#SBATCH --job-name=eval_clf_subj
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node2
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o

echo "=========================================="
echo "Evaluating Classification Test Subjects"
echo "Run ID: mc3r4vhf"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Run evaluation
python evaluate_clf_test_subjects.py --run_id mc3r4vhf

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
