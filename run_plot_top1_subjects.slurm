#!/bin/bash
#SBATCH --job-name=plot_top1
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/plot_top1-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/plot_top1-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Plot predictions vs targets for top 1 subjects per emotion

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio

# Model path
MODEL_PATH="output/moviefmri/opr6oq97"
OUTPUT_DIR="analysis/plots/top1_subjects"

# Create output directory
mkdir -p $OUTPUT_DIR

# Top 1 subjects per emotion (from subject_performance analysis)
declare -A TOP_SUBJECTS
TOP_SUBJECTS[Anger]="NDARUC771VM5"
TOP_SUBJECTS[Happy]="NDARYR994ELB"
TOP_SUBJECTS[Fear]="NDARDU853XZ6"
TOP_SUBJECTS[Sad]="NDARTK357VHL"
TOP_SUBJECTS[Excited]="NDARJW697MYZ"
TOP_SUBJECTS[Positive]="NDARMZ366UY8"
TOP_SUBJECTS[Negative]="NDARZE850WXD"

echo "=========================================="
echo "Plotting predictions for top 1 subjects"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Loop through emotions and plot for top 1 subject
for emotion in Anger Happy Fear Sad Excited Positive Negative
do
    subject_id=${TOP_SUBJECTS[$emotion]}
    echo "[$(date)] Processing $emotion - Subject: $subject_id"

    python analysis/plot_predictions_vs_targets.py \
        --model_path $MODEL_PATH \
        --emotion $emotion \
        --subject_id $subject_id \
        --output_dir $OUTPUT_DIR \
        --split test

    if [ $? -eq 0 ]; then
        echo "  ✓ $emotion plot completed successfully"
    else
        echo "  ✗ $emotion plot failed"
    fi
    echo ""
done

echo "=========================================="
echo "All plots completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
