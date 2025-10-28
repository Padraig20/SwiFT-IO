#!/bin/bash
#SBATCH --job-name=compare_clf
#SBATCH --output=compare_clf-%j.out
#SBATCH --error=compare_clf-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
python compare_classification_seq20_seq30.py
