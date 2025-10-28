#!/bin/bash
#SBATCH --job-name=opt_thr
#SBATCH --output=opt_thr-%j.out
#SBATCH --error=opt_thr-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
python find_optimal_thresholds.py
