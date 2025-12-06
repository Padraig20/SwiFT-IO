#!/usr/bin/env python3
"""
Generate SLURM scripts for classification IG maps using peak sequences
Uses emotion-specific top 5 subjects and their peak sequences
"""

import json
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Load peak sequences
peak_file = project_root / "analysis/4_IGmap/clf_emotion_peak_sequences/mc3r4vhf_emotion_peak_sequences_top10.json"
with open(peak_file) as f:
    peak_data = json.load(f)

emotion_labels = peak_data['emotion_labels']

# Find top 5 subjects per emotion based on peak sequence scores
top5_per_emotion = {}
for emotion in emotion_labels:
    subject_peaks = []
    for subject_id, subject_data in peak_data['subjects'].items():
        if emotion in subject_data and len(subject_data[emotion]) > 0:
            top_seq = subject_data[emotion][0]
            subject_peaks.append({
                'subject': subject_id,
                'avg_score': top_seq['avg_score']
            })

    subject_peaks.sort(key=lambda x: x['avg_score'], reverse=True)
    top5_per_emotion[emotion] = [s['subject'] for s in subject_peaks[:5]]

print("="*80)
print("Generating Classification IG Map SLURM Scripts")
print("="*80)
print(f"Run ID: mc3r4vhf")
print(f"Using emotion-specific top 5 subjects with their peak sequences")
print(f"Baseline: first_10sec")
print(f"Nodes: node2, node4 (alternating)")
print(f"Time limit: 24:00:00")
print("="*80)

# Create scripts directory
scripts_dir = project_root / "igmap/clf_ig_jobs"
scripts_dir.mkdir(exist_ok=True)

total_scripts = 0
nodes = ['node2', 'node4']

for emotion in emotion_labels:
    subjects = top5_per_emotion[emotion]

    print(f"\n{emotion}: {len(subjects)} subjects")

    for rank, subject in enumerate(subjects, 1):
        # Check if subject has peak sequences
        if subject not in peak_data['subjects']:
            print(f"  Rank {rank}. {subject} → SKIP (no peak sequences)")
            continue

        if emotion not in peak_data['subjects'][subject]:
            print(f"  Rank {rank}. {subject} → SKIP (no {emotion} sequences)")
            continue

        # Get top 5 peak sequences for this subject and emotion
        peak_seqs = peak_data['subjects'][subject][emotion][:5]
        seq_indices = [seq['sequence_idx'] for seq in peak_seqs]

        # Alternate nodes
        node = nodes[total_scripts % 2]
        print(f"  Rank {rank}. {subject} → {node} (peak seqs: {seq_indices})")

        # Generate SLURM script
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
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "Classification IG Map - {emotion} Rank {rank}"
echo "Subject: {subject}"
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

# Run IG map computation
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

        # Make executable
        slurm_path.chmod(0o755)

        total_scripts += 1

print("\n" + "="*80)
print(f"✅ {total_scripts} SLURM scripts generated!")
print(f"   Location: {scripts_dir}")
print(f"   Task: Classification IG Maps")
print(f"   Baseline: first_10sec")
print(f"   Nodes: node2, node4 (alternating)")
print(f"   Time limit: 24:00:00")
print(f"   Memory: 32GB per job")
print(f"   CPUs: 4 per job")
print("\nTo submit all jobs:")
print(f"  cd {scripts_dir}")
print("  for script in run_igc_*.slurm; do sbatch $script && sleep 0.5; done")
print("\nEstimated time per job: ~2-4 hours (varies by emotion/subject)")
print(f"Total jobs: {total_scripts}")
print("="*80)
