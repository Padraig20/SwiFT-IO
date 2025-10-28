#!/usr/bin/env python3
"""
Analyze per-emotion accuracy/F1 score and find top performers for each emotion
For CLASSIFICATION task
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
run_id = "mc3r4vhf"

# Load results
result_file = project_root / f"analysis/4_IGmap/subject_performance/{run_id}_clf_test_subject_performance.json"
with open(result_file) as f:
    data = json.load(f)

# Emotion names
emotion_labels = data['emotion_labels']

print("="*80)
print("EMOTION-SPECIFIC PERFORMANCE ANALYSIS - CLASSIFICATION")
print("="*80)
print()

# Collect per-emotion F1 scores for each subject
emotion_f1 = defaultdict(dict)  # {emotion: {subject: f1_score}}
emotion_acc = defaultdict(dict)  # {emotion: {subject: accuracy}}

for subject, result in data['all_subjects'].items():
    for emotion_name in emotion_labels:
        metrics = result['emotion_metrics'][emotion_name]
        emotion_f1[emotion_name][subject] = metrics['f1_score']
        emotion_acc[emotion_name][subject] = metrics['accuracy']

# Print top 5 subjects per emotion (by F1 score)
print("\n" + "="*80)
print("TOP 5 SUBJECTS PER EMOTION (Highest F1 Score)")
print("="*80)

top5_per_emotion = {}
for emotion_name in emotion_labels:
    sorted_subjects = sorted(emotion_f1[emotion_name].items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_subjects[:5]
    top5_per_emotion[emotion_name] = [s[0] for s in top5]

    print(f"\n{emotion_name}:")
    print(f"{'Rank':<6} {'Subject':<20} {'F1 Score':<12} {'Accuracy':<12}")
    print("-"*50)
    for rank, (subject, f1) in enumerate(top5, 1):
        acc = emotion_acc[emotion_name][subject]
        print(f"{rank:<6} {subject:<20} {f1:<12.4f} {acc:<12.4f}")

# Analyze overlap
print("\n" + "="*80)
print("OVERLAP ANALYSIS: Which subjects appear in multiple emotions?")
print("="*80)

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
print(f"\nOverall Top 5 (by average F1 score across all emotions):")
for i, subj in enumerate(overall_top5, 1):
    avg_f1 = data['all_subjects'][subj]['avg_f1_score']
    print(f"{i}. {subj} (Avg F1: {avg_f1:.4f})")

print(f"\nSubjects appearing in ≥4 emotions' top 5:")
for subject, count in frequent_subjects:
    if count >= 4:
        avg_f1 = data['all_subjects'][subject]['avg_f1_score']
        print(f"  {subject}: {count}/7 emotions (Avg F1: {avg_f1:.4f})")

# Save emotion-specific results
output_path = project_root / f"analysis/4_IGmap/subject_performance/{run_id}_clf_emotion_specific_performance.json"
with open(output_path, 'w') as f:
    json.dump({
        'run_id': run_id,
        'task_type': 'classification',
        'emotion_labels': emotion_labels,
        'top5_per_emotion': top5_per_emotion,
        'emotion_f1': {k: dict(v) for k, v in emotion_f1.items()},
        'emotion_acc': {k: dict(v) for k, v in emotion_acc.items()},
        'overlap_analysis': {subj: count for subj, count in subject_counts.items()}
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(f"\n{'Emotion':<12} {'Min F1':<10} {'Max F1':<10} {'Mean F1':<10} {'Std F1':<10}")
print("-"*52)
for emotion_name in emotion_labels:
    f1_scores = list(emotion_f1[emotion_name].values())
    print(f"{emotion_name:<12} {np.min(f1_scores):<10.4f} {np.max(f1_scores):<10.4f} "
          f"{np.mean(f1_scores):<10.4f} {np.std(f1_scores):<10.4f}")

print("\n" + "="*80)
