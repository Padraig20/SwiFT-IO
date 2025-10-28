#!/usr/bin/env python3
"""
Generate SLURM scripts for emotion-specific top performers
ONE subject per job for faster parallel processing
"""

import json
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Load emotion-specific top 5
with open(project_root / "analysis/4_IGmap/subject_performance/opr6oq97_emotion_specific_mse.json") as f:
    data = json.load(f)

emotion_labels = data['emotion_labels']
top5_per_emotion = data['top5_per_emotion']

print("Generating SLURM scripts for emotion-specific top performers (1 subject per job)...")
print("="*70)

# Create scripts directory if not exists
scripts_dir = project_root / "igmap/emotion_specific_jobs_single"
scripts_dir.mkdir(exist_ok=True)

total_scripts = 0
nodes = ['node2', 'node4']  # Alternate between node2 and node4

for emotion in emotion_labels:
    subjects = top5_per_emotion[emotion]

    print(f"\n{emotion}: {len(subjects)} subjects")

    for rank, subject in enumerate(subjects, 1):
        # Alternate nodes for load balancing
        node = nodes[rank % 2]
        print(f"  {rank}. {subject} → {node}")

        # Generate SLURM script for this subject-emotion pair
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=ig_{emotion[:3].lower()}_r{rank}
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_{emotion.lower()}_rank{rank}-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_{emotion.lower()}_rank{rank}-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist={node}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - {emotion} Rank {rank}: {subject}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Process single subject for {emotion}
/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \\
    --run_id opr6oq97 \\
    --subjects {subject} \\
    --emotions {emotion} \\
    --top_k_seqs 5 \\
    --n_steps 20 \\
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ {emotion} Rank {rank} ({subject}) complete"
else
    echo "❌ {emotion} Rank {rank} ({subject}) FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="
"""

        # Save SLURM script
        slurm_path = scripts_dir / f"run_ig_{emotion.lower()}_rank{rank}.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_content)

        total_scripts += 1

print("\n" + "="*70)
print(f"✅ {total_scripts} SLURM scripts generated!")
print(f"   Location: {scripts_dir}")
print("\nTo submit all jobs:")
print(f"  cd {scripts_dir}")
print("  for script in run_ig_*.slurm; do sbatch $script && sleep 0.5; done")
print("\nEstimated time per job: ~15-20 minutes")
print(f"Total jobs: {total_scripts} (parallel execution on available nodes)")
print("="*70)
