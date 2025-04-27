# ==========================================
# 예시 실행 방법:

# 조합 1A (subject 단위): residual 상위 20% (오차 큼)
# python select_subjects_by_residual.py --mode res_high --percentile 80 --emotion_mode avg --run_id tubg3tim --unit subject

# 조합 1B (subject 단위): residual 하위 20% (오차 작음)
# python select_subjects_by_residual.py --mode res_low --percentile 20 --emotion_mode avg --run_id tubg3tim --unit subject

# 조합 2A (subject 단위): residual 하위 20% & 감정 강도 높은 경우 (confidence 높음)
# python select_subjects_by_residual.py --mode conf_high --percentile 20 --emotion_mode avg --run_id tubg3tim --unit subject

# 조합 2B (subject 단위): residual 상위 20% & 감정 강도 높은 경우 (confidence 낮음)
# python select_subjects_by_residual.py --mode conf_low --percentile 80 --emotion_mode avg --run_id tubg3tim --unit subject

# 조합 1A (segment 단위): residual 상위 20%
# python select_subjects_by_residual.py --mode res_high --percentile 80 --emotion_mode avg --run_id tubg3tim --unit segment

# 조합 1B (segment 단위): residual 하위 20%
# python select_subjects_by_residual.py --mode res_low --percentile 20 --emotion_mode avg --run_id tubg3tim --unit segment

# 조합 2A (segment 단위): confidence 상위 20%
# python select_subjects_by_residual.py --mode conf_high --percentile 20 --emotion_mode avg --run_id tubg3tim --unit segment

# 조합 2B (segment 단위): confidence 하위 20%
# python select_subjects_by_residual.py --mode conf_low --percentile 80 --emotion_mode avg --run_id tubg3tim --unit segment
# ==========================================

import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Select subjects based on residuals or confidence for IG analysis.")
parser.add_argument('--mode', type=str, required=True, choices=['res_high', 'res_low', 'conf_high', 'conf_low'],
                    help='Selection mode: residual-based or confidence-based')
parser.add_argument('--percentile', type=int, required=True, help='Percentile threshold for selection')
parser.add_argument('--emotion_mode', type=str, required=True, choices=['avg', 'single'],
                    help='Use averaged emotion residuals or a specific emotion')
parser.add_argument('--emotion_idx', type=int, default=None, help='Index of emotion (required if emotion_mode is single)')
parser.add_argument('--run_id', type=str, required=True, help='Run ID to locate residual file')
parser.add_argument('--unit', type=str, choices=['subject', 'segment'], default='subject',
                    help='Unit of analysis: subject-level (default) or segment-level')
args = parser.parse_args()

save_dir = Path(f"/pscratch/sd/k/kimbo/SwiFT-IO/analysis/4_IGmap/results_each/{args.run_id}")
data = np.load(save_dir / "residuals.npy", allow_pickle=True).item()

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

subjects = data['subject']

if args.unit == 'subject':
    subject_to_res = {}
    subject_to_pred = {}

    for i in range(len(subjects)):
        subj = subjects[i]
        if subj not in subject_to_res:
            subject_to_res[subj] = []
            subject_to_pred[subj] = []
        subj_res = []
        subj_pred = []
        for j, emo in enumerate(emotion_labels):
            r_values = data[f"residual_{j}_{emo}"][i]
            p_values = data[f"prediction_{j}_{emo}"][i]
            if args.emotion_mode == 'avg':
                subj_res.extend(r_values)
                subj_pred.extend(p_values)
            elif args.emotion_mode == 'single' and args.emotion_idx == j:
                subj_res = r_values
                subj_pred = p_values
        subject_to_res[subj].extend(subj_res)
        subject_to_pred[subj].extend(subj_pred)

    subject_list = []
    residuals = []
    predictions = []
    for subj in subject_to_res:
        mean_res = np.mean(np.abs(subject_to_res[subj]))
        mean_pred = np.mean(np.abs(subject_to_pred[subj]))
        subject_list.append(subj)
        residuals.append(mean_res)
        predictions.append(mean_pred)

else:  # segment 단위
    residuals = []
    predictions = []
    subject_list = []
    for i in range(len(subjects)):
        subj_res = []
        subj_pred = []
        for j, emo in enumerate(emotion_labels):
            r_values = data[f"residual_{j}_{emo}"][i]
            p_values = data[f"prediction_{j}_{emo}"][i]
            if args.emotion_mode == 'avg':
                subj_res.extend(r_values)
                subj_pred.extend(p_values)
            elif args.emotion_mode == 'single' and args.emotion_idx == j:
                subj_res = r_values
                subj_pred = p_values
        mean_res = np.mean(np.abs(subj_res))
        mean_pred = np.mean(np.abs(subj_pred))
        residuals.append(mean_res)
        predictions.append(mean_pred)
        subject_list.append(subjects[i])

residuals = np.array(residuals)
predictions = np.array(predictions)
subject_list = np.array(subject_list)

valid_idx = ~np.isnan(residuals)
residuals = residuals[valid_idx]
predictions = predictions[valid_idx]
subject_list = subject_list[valid_idx]

# 선택 기준에 따른 그룹 선택
if args.mode == 'res_high':
    threshold = np.percentile(residuals, args.percentile)
    selected_subjects = subject_list[residuals >= threshold]
elif args.mode == 'res_low':
    threshold = np.percentile(residuals, args.percentile)
    selected_subjects = subject_list[residuals <= threshold]
elif args.mode == 'conf_high':
    confidence = -residuals * (1 / (predictions + 1e-6))
    threshold = np.percentile(confidence, 100 - args.percentile)
    selected_subjects = subject_list[confidence >= threshold]
elif args.mode == 'conf_low':
    confidence = -residuals * (1 / (predictions + 1e-6))
    threshold = np.percentile(confidence, 100 - args.percentile)  # ✅ fix 적용
    selected_subjects = subject_list[confidence <= threshold]

# 저장 경로 설정
suffix = f"{args.unit}"
if args.emotion_mode == 'avg':
    save_path = save_dir / f"selected_subjects_{args.mode}_p{args.percentile}_emomode-avg_{suffix}.txt"
else:
    emo_label = emotion_labels[args.emotion_idx]
    save_path = save_dir / f"selected_subjects_{args.mode}_p{args.percentile}_emo-{emo_label}_{suffix}.txt"

# 저장
with open(save_path, 'w') as f:
    for s in selected_subjects:
        f.write(f"{s}\n")

print(f"Saved selected subject list to {save_path}")
