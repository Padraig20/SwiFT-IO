#!/bin/bash
#SBATCH --job-name=clf_viz
#SBATCH --output=clf_viz-%j.out
#SBATCH --error=clf_viz-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Activate environment
source ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swiftio

# Run visualization
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

python visualize_classification_predictions.py

echo "Done!"
