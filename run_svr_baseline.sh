#!/bin/bash

# SVR Baseline Training Script
# Uses the same data split and input as SwiFT-IO

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio


cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

python src/train_svr_baseline.py \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --dataset_name HBN \
    --downstream_task emotions \
    --downstream_task_type regression \
    --input_type movieDM \
    --dataset_split_seed 777 \
    --sequence_length 30 \
    --stride_between_seq 1 \
    --stride_within_seq 1 \
    --kernel rbf \
    --C 1.0 \
    --epsilon 0.1 \
    --standardize \
    --batch_size 4 \
    --eval_batch_size 16 \
    --num_workers 0 \
    --output_dir output/svr_baseline \
    --experiment_name svr_seq30_rbf

echo "SVR Baseline training complete!"
