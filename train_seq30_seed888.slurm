#!/bin/bash
#SBATCH --job-name=clf30_s888
#SBATCH --output=clf30_s888-%j.out
#SBATCH --error=clf30_s888-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1

source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/src

# Seq 30 with seed 888 (validation experiment)
# Based on gajr5p1p run but with different seed

python main.py \
    --project_name moviefmri \
    --dataset_name HBN \
    --loggername wandb \
    --downstream_task emotions \
    --downstream_task_type classification \
    --model swin4d_ver9 \
    --decoder series_decoder \
    --clf_head_version v1 \
    --sequence_length 30 \
    --stride_within_seq 1 \
    --stride_between_seq 1 \
    --input_offset 0 \
    --img_size 96 96 96 30 \
    --patch_size 4 4 4 1 \
    --in_chans 1 \
    --embed_dim 36 \
    --depths 2 2 6 2 \
    --num_heads 3 6 12 24 \
    --window_size 4 4 4 4 \
    --first_window_size 4 4 4 4 \
    --last_layer_full_MSA \
    --num_targets 7 \
    --num_classes 2 \
    --image_path /scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120 \
    --label_scaling_method standardization \
    --default_root_dir output/moviefmri \
    --use_scheduler \
    --milestones 100 150 \
    --max_epochs 30 \
    --check_val_every_n_epoch 1 \
    --batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 0.00005 \
    --accumulate_grad_batches 4 \
    --optimizer AdamW \
    --weight_decay 0.01 \
    --momentum 0 \
    --num_workers 4 \
    --seed 888 \
    --log_every_n_steps 50 \
    --precision 16 \
    --devices 8 \
    --strategy ddp \
    --num_nodes 1 \
    --num_sanity_val_steps 2 \
    --adjust_hrf \
    --train_split 0.7 \
    --val_split 0.15 \
    --contrastive_type 0 \
    --cycle 0.5 \
    --temperature 0.1

echo "Training completed for seed 888!"
