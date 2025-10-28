#!/usr/bin/env python3
"""
Generate SLURM scripts for emotion-specific top performers
"""

import json
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Load emotion-specific top 5
with open(project_root / "analysis/4_IGmap/subject_performance/opr6oq97_emotion_specific_mse.json") as f:
    data = json.load(f)

emotion_labels = data['emotion_labels']
top5_per_emotion = data['top5_per_emotion']

print("Generating SLURM scripts for emotion-specific top performers...")
print("="*70)

# Create scripts directory if not exists
scripts_dir = project_root / "igmap/emotion_specific_jobs"
scripts_dir.mkdir(exist_ok=True)

for emotion in emotion_labels:
    subjects = top5_per_emotion[emotion]

    print(f"\n{emotion}: {len(subjects)} subjects")
    for i, subj in enumerate(subjects, 1):
        print(f"  {i}. {subj}")

    # Generate SLURM script for this emotion
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=ig_{emotion.lower()}
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_{emotion.lower()}_top5-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_{emotion.lower()}_top5-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - {emotion} Top 5 Performers"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process top 5 subjects for {emotion}
# Only process peak sequences for {emotion} (--emotions {emotion})
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \\
    --run_id opr6oq97 \\
    --subjects {' '.join(subjects)} \\
    --emotions {emotion} \\
    --top_k_seqs 5 \\
    --n_steps 20 \\
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ {emotion} processing complete"
else
    echo "❌ {emotion} processing FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
"""

    # Save SLURM script
    slurm_path = scripts_dir / f"run_ig_{emotion.lower()}_top5.slurm"
    with open(slurm_path, 'w') as f:
        f.write(slurm_content)

    print(f"  ✅ Created: {slurm_path.name}")

print("\n" + "="*70)
print("✅ All SLURM scripts generated!")
print(f"   Location: {scripts_dir}")
print("\nTo submit all jobs:")
print(f"  cd {scripts_dir}")
print("  for script in run_ig_*.slurm; do sbatch $script; done")
print("="*70)
