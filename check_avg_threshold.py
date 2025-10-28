import nibabel as nib
import numpy as np
from pathlib import Path
import json

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def load_and_average_emotion(emotion):
    project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
    base_dir = project_root / "analysis/4_IGmap/baseline_first_10sec_selective/opr6oq97/nii_segments"
    
    json_path = project_root / "analysis/4_IGmap/subject_performance/opr6oq97_emotion_specific_mse.json"
    with open(json_path) as f:
        data = json.load(f)
    
    emotion_idx = emotion_labels.index(emotion)
    subjects = data['top5_per_emotion'][emotion]
    
    all_maps = []
    for subject in subjects:
        subject_dir = base_dir / subject / f"target{emotion_idx}_{emotion}"
        pos_files = sorted(subject_dir.glob(f"{subject}_{emotion}_TR*_rank*_AVGpred_positive.nii.gz"))
        
        for pos_path in pos_files:
            pos_data = nib.load(str(pos_path)).get_fdata()
            all_maps.append(pos_data)
    
    # Average across all maps
    mean_map = np.mean(all_maps, axis=0)
    return mean_map

# Check Anger emotion
print("Checking Anger emotion averaged IG map:")
mean_map = load_and_average_emotion('Anger')

pos_values = mean_map[mean_map > 0]
print(f"\nTotal positive voxels: {len(pos_values)}")
print(f"Mean: {pos_values.mean():.8f}")
print(f"Max: {pos_values.max():.8f}")
print()

# Check percentiles
for p in [90, 95, 99]:
    threshold = np.percentile(pos_values, p)
    above_threshold = np.sum(pos_values >= threshold)
    percentage = (above_threshold / len(pos_values)) * 100
    print(f"Percentile {p:5.1f}: threshold={threshold:.8f}, above={above_threshold:6d} ({percentage:5.2f}%)")

# Check cumulative distribution
print(f"\nCumulative distribution check:")
sorted_values = np.sort(pos_values)
indices_to_check = [int(len(sorted_values) * p) for p in [0.5, 0.8, 0.9, 0.95, 0.99]]
for idx in indices_to_check:
    percentile = (idx / len(sorted_values)) * 100
    value = sorted_values[idx]
    print(f"  {percentile:5.1f}%: value={value:.8f}")

