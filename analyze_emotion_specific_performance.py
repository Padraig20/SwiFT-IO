#!/usr/bin/env python3
"""
Analyze per-emotion MSE and find top performers for each emotion
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
run_id = "opr6oq97"

# Load results
result_file = project_root / f"analysis/4_IGmap/subject_performance/{run_id}_test_subject_mse.json"
with open(result_file) as f:
    data = json.load(f)

# Emotion names
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

print("="*80)
print("EMOTION-SPECIFIC PERFORMANCE ANALYSIS")
print("="*80)
print()

# Calculate per-emotion MSE for each subject
emotion_mse = defaultdict(dict)  # {emotion: {subject: mse}}

for subject, result in data['all_subjects'].items():
    preds = np.array(result['predictions'])  # [num_sequences, 7]
    targets = np.array(result['targets'])    # [num_sequences, 7]
    
    # Calculate MSE per emotion
    for emo_idx in range(7):
        emotion_name = emotion_labels[emo_idx]
        mse = np.mean((preds[:, emo_idx] - targets[:, emo_idx]) ** 2)
        emotion_mse[emotion_name][subject] = float(mse)

# Print top 5 subjects per emotion
print("\n" + "="*80)
print("TOP 5 SUBJECTS PER EMOTION (Lowest MSE)")
print("="*80)

top5_per_emotion = {}
for emotion_name in emotion_labels:
    sorted_subjects = sorted(emotion_mse[emotion_name].items(), key=lambda x: x[1])
    top5 = sorted_subjects[:5]
    top5_per_emotion[emotion_name] = [s[0] for s in top5]
    
    print(f"\n{emotion_name}:")
    print(f"{'Rank':<6} {'Subject':<20} {'MSE':<12}")
    print("-"*40)
    for rank, (subject, mse) in enumerate(top5, 1):
        print(f"{rank:<6} {subject:<20} {mse:<12.6f}")

# Analyze overlap
print("\n" + "="*80)
print("OVERLAP ANALYSIS: Which subjects appear in multiple emotions?")
print("="*80)

from collections import Counter
all_top5_subjects = []
for subjects in top5_per_emotion.values():
    all_top5_subjects.extend(subjects)

subject_counts = Counter(all_top5_subjects)
frequent_subjects = [(subj, count) for subj, count in subject_counts.items() if count > 1]
frequent_subjects.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Subject':<20} {'# Emotions in Top 5':<20} {'Emotions'}")
print("-"*80)
for subject, count in frequent_subjects:
    emotions = [emo for emo, subjs in top5_per_emotion.items() if subject in subjs]
    print(f"{subject:<20} {count:<20} {', '.join(emotions)}")

# Compare with overall top 5
print("\n" + "="*80)
print("COMPARISON: Overall Top 5 vs Emotion-Specific Top 5")
print("="*80)

overall_top5 = data['top5_subjects']
print(f"\nOverall Top 5 (by average MSE across all emotions):")
for i, subj in enumerate(overall_top5, 1):
    print(f"{i}. {subj}")

print(f"\nSubjects appearing in ≥4 emotions' top 5:")
for subject, count in frequent_subjects:
    if count >= 4:
        print(f"  {subject}: {count}/7 emotions")

# Save emotion-specific results
output_path = project_root / f"analysis/4_IGmap/subject_performance/{run_id}_emotion_specific_mse.json"
with open(output_path, 'w') as f:
    json.dump({
        'run_id': run_id,
        'emotion_labels': emotion_labels,
        'top5_per_emotion': top5_per_emotion,
        'emotion_mse': {k: dict(v) for k, v in emotion_mse.items()},
        'overlap_analysis': {subj: count for subj, count in subject_counts.items()}
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(f"\n{'Emotion':<12} {'Min MSE':<12} {'Max MSE':<12} {'Mean MSE':<12} {'Std MSE':<12}")
print("-"*60)
for emotion_name in emotion_labels:
    mses = list(emotion_mse[emotion_name].values())
    print(f"{emotion_name:<12} {np.min(mses):<12.4f} {np.max(mses):<12.4f} {np.mean(mses):<12.4f} {np.std(mses):<12.4f}")

