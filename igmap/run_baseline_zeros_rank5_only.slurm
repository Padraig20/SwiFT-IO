#!/bin/bash
#SBATCH --job-name=ig_bl_zeros_r5
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_baseline_zeros_rank5-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_baseline_zeros_rank5-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Baseline Comparison - ZEROS (Rank 5 only)"
echo "Subject: sub-NDARMZ366UY8 (Positive top 1)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Create a temp script that processes only TR090 (rank 5)
cat << 'PYEOF' > /tmp/igmap_zeros_rank5_only.py
#!/usr/bin/env python3
"""Run IG for rank 5 only"""
import sys
from pathlib import Path
sys.path.append("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Import the main script
import igmap.igmap_baseline_comparison as igmap_module

# Override command line args
class Args:
    run_id = 'opr6oq97'
    subjects = ['sub-NDARMZ366UY8']
    baseline = 'zeros'
    emotion_peaks_json = 'analysis/4_IGmap/emotion_peak_sequences/opr6oq97_emotion_peak_sequences_top10.json'
    top_k_seqs = 5
    emotions = ['Positive']
    n_steps = 20
    project_root = '/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO'

args = Args()
project_root = Path(args.project_root)

# Initialize model
import torch
ckpt_path = project_root / f"output/moviefmri/{args.run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
args_model_dict = ckpt['hyper_parameters']

print(f"{'='*70}")
print(f"IG Map - Zeros Baseline - Rank 5 Only")
print(f"{'='*70}")
print(f"✅ Loaded checkpoint: {ckpt_path.name}")
print(f"   Subject: sub-NDARMZ366UY8")
print(f"   Emotion: Positive")
print(f"   Processing: TR090 (Rank 5)")

# Load emotion peaks
import json
peaks_json_path = project_root / args.emotion_peaks_json
with open(peaks_json_path, 'r') as f:
    peaks_data = json.load(f)

# Initialize model once
igmap_module.init_model_and_data(str(ckpt_path), args_model_dict, project_root)

# Use reference affine
affine_path = project_root / "analysis/4_IGmap/reference_affine_MNI152_2mm.nii.gz"
if not affine_path.exists():
    affine_path = project_root / "igmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

# Process only rank 5
subject = 'sub-NDARMZ366UY8'
if subject in peaks_data['subjects']:
    subject_peaks = peaks_data['subjects'][subject]

    # Get only rank 5 (index 4 = 5th sequence)
    selected_sequences = {'Positive': [subject_peaks['Positive'][4]]}

    print(f"\n🎯 Processing TR090 (Rank 5)...")
    igmap_module.process_subject((subject, selected_sequences, args, affine_path, project_root))
else:
    print(f"❌ No peak data for subject: {subject}")

print(f"\n{'='*70}")
print("✅ Rank 5 processing complete!")
print(f"{'='*70}")
PYEOF

# Run the script
/home/connectome/kimbo/.conda/envs/swiftio/bin/python /tmp/igmap_zeros_rank5_only.py

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ Baseline ZEROS Rank 5 complete"
else
    echo "❌ Baseline ZEROS Rank 5 FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
