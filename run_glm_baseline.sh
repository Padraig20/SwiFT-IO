#!/bin/bash

# Run GLM Baseline Training Script
# This script trains GLM baseline using the same train/val/test split as SwiFT-IO

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

cd src/

python train_glm_baseline.py \
    --image_path /scratch/HBN/9.2.movieDM_ROI_timeseries \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type regression \
    --input_type movieDM \
    --dataset_split_seed 777 \
    --train_split 0.7 \
    --val_split 0.15 \
    --sequence_length 30 \
    --use_cross_validation \
    --use_emotion_rois_only \
    --standardize \
    --roi_timeseries_dir /scratch/HBN/9.2.movieDM_ROI_timeseries \
    --output_dir ../output/glm_baseline \
    --adjust_hrf

echo "GLM baseline training complete!"
echo "Results saved to output/glm_baseline/"
