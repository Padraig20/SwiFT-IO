#!/usr/bin/env python3
"""
Generate SLURM scripts for classification emotion-specific top 5 performers
Using first_10sec baseline (proven to be better in regression)
"""

import json
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Load emotion-specific top 5 for classification
clf_perf_file = project_root / "analysis/4_IGmap/subject_performance/mc3r4vhf_clf_emotion_specific_performance.json"

if not clf_perf_file.exists():
    print(f"❌ Error: {clf_perf_file} not found!")
    print("Please run these steps first:")
    print("  1. python evaluate_clf_test_subjects.py --run_id mc3r4vhf")
    print("  2. python analyze_clf_emotion_performance.py")
    exit(1)

with open(clf_perf_file) as f:
    data = json.load(f)

emotion_labels = data['emotion_labels']
top5_per_emotion = data['top5_per_emotion']

print("Generating SLURM scripts for CLASSIFICATION emotion-specific top 5 IG maps")
print("Baseline: first_10sec (better interpretability)")
print("="*70)

# Create scripts directory
scripts_dir = project_root / "igmap/clf_emotion_top5_ig_jobs"
scripts_dir.mkdir(exist_ok=True)

total_scripts = 0
nodes = ['node1', 'node3']  # Nodes with GPU (though IG doesn't need GPU)

for emotion in emotion_labels:
    subjects = top5_per_emotion[emotion]

    print(f"\n{emotion}: {len(subjects)} subjects")

    for rank, subject in enumerate(subjects, 1):
        # Alternate nodes for load balancing
        node = nodes[(total_scripts) % 2]
        print(f"  Rank {rank}. {subject} → {node}")

        # Generate SLURM script for this subject-emotion pair
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=igc_{emotion[:3].lower()}_r{rank}
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/igc_{emotion.lower()}_rank{rank}-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/igc_{emotion.lower()}_rank{rank}-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist={node}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map (CLASSIFICATION) - {emotion} Rank {rank}: {subject}"
echo "Baseline: first_10sec"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Activate conda environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Process single subject for {emotion} with first_10sec baseline
python igmap/igmap_clf_baseline_comparison.py \\
    --run_id mc3r4vhf \\
    --subjects {subject} \\
    --baseline first_10sec \\
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
        slurm_path = scripts_dir / f"run_igc_{emotion.lower()}_rank{rank}.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_content)

        total_scripts += 1

print("\n" + "="*70)
print(f"✅ {total_scripts} SLURM scripts generated!")
print(f"   Location: {scripts_dir}")
print(f"   Task: CLASSIFICATION")
print(f"   Baseline: first_10sec")
print(f"   Nodes: node1, node3 (alternating)")
print("\nTo submit all jobs:")
print(f"  cd {scripts_dir}")
print("  for script in run_igc_*.slurm; do sbatch $script && sleep 0.5; done")
print("\nEstimated time per job: ~15-20 minutes")
print(f"Total jobs: {total_scripts}")
print("="*70)
