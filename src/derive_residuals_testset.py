import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

# ===========================
# 예시 실행 방법:
# python derive_residuals.py --run_id tubg3tim --input_type movieDM --seq_length 50
# ===========================

# CLI 인자 받기
parser = argparse.ArgumentParser(description="Extract residuals from model predictions and save with frame info.")
parser.add_argument('--run_id', type=str, required=True, help='Run ID (e.g., cvy8kv4o)')
parser.add_argument('--input_type', type=str, choices=['movieDM', 'movieTP'], required=True, help='Input type')
parser.add_argument('--seq_length', type=int, default=50, help='Sequence length (default: 50)')
args_cli = parser.parse_args()

run_id = args_cli.run_id
input_type = args_cli.input_type
seq_length = args_cli.seq_length

# 감정 레이블
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

# 프로젝트 경로 설정
project_root = Path("/pscratch/sd/k/kimbo/SwiFT-IO")
project_main = project_root / "src"
sys.path.append(str(project_main))

from module.models.encoder.swin4d_transformer_ver7 import SwinTransformer4D
from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

# 체크포인트 경로 설정
project_id = "moviefmri"
ckpt_dir = project_root / f"output/{project_id}/{run_id}"
ckpt_path = list(ckpt_dir.glob("checkpt*"))[0]
ckpt = torch.load(ckpt_path, map_location='cpu')

# 하이퍼파라미터 설정 및 덮어쓰기
ckpt['hyper_parameters']['input_type'] = input_type
ckpt['hyper_parameters']['seq_length'] = seq_length
ckpt['hyper_parameters']['image_path'] = "/global/cfs/cdirs/m4750/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
ckpt['hyper_parameters']['default_root_dir'] = str(project_root / "output/moviefmri")
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1
ckpt['hyper_parameters']['input_offset'] = 3
ckpt['hyper_parameters']['bad_subj_path'] = None
ckpt['hyper_parameters']['limit_training_samples'] = 0
ckpt['hyper_parameters']['img_size'] = [96, 96, 96, seq_length]
args = ckpt['hyper_parameters']

# total_frame 설정
total_frame = 750 if input_type == 'movieDM' else 300

# 데이터 모듈 초기화
data_module = fMRIDataModule(**args)
data_module.setup()
data_module.prepare_data()
test_loader = data_module.test_dataloader()

# 모델 초기화
model = LitClassifier(data_module=data_module, **args)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model.cpu()

# 결과 저장 경로
save_dir = Path(f"/pscratch/sd/k/kimbo/SwiFT-IO/analysis/4_IGmap/results_each/{run_id}")
save_dir.mkdir(parents=True, exist_ok=True)

# 예측 및 오차 저장 함수
def save_predictions_and_residuals(model, test_loader, save_dir):
    rows = []
    segment_counter = {}

    # 전체 저장용 dict 초기화
    full_data = {
        "subject": [],
        "start_frame": [],
        "end_frame": [],
    }
    for i, label in enumerate(emotion_labels):
        full_data[f"residual_{i}_{label}"] = []
        full_data[f"prediction_{i}_{label}"] = []

    for idx, data in enumerate(tqdm(test_loader)):
        subj_name = data['subject_name'][0]
        input_ts = data['fmri_sequence'].float().cpu()
        target = data['target'].float().cpu()  # shape: (1, 50, 7)

        with torch.no_grad():
            pred = model(input_ts)  # shape: (50, 7)

        residual = torch.abs(pred - target.squeeze(0))  # shape: (50, 7)

        seg_idx = segment_counter.get(subj_name, 0)
        start = seg_idx * seq_length
        end = start + seq_length - 1
        segment_counter[subj_name] = seg_idx + 1

        residual_np = residual.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        row = {
            "subject": subj_name,
            "start_frame": start,
            "end_frame": end
        }

        full_data["subject"].append(subj_name)
        full_data["start_frame"].append(start)
        full_data["end_frame"].append(end)

        for i, emotion in enumerate(emotion_labels):
            r_list = residual_np[:, i].tolist()
            p_list = pred_np[:, i].tolist()
            row[f"residual_{i}_{emotion}"] = r_list
            row[f"prediction_{i}_{emotion}"] = p_list
            full_data[f"residual_{i}_{emotion}"].append(r_list)
            full_data[f"prediction_{i}_{emotion}"].append(p_list)

        rows.append(row)

    # CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "residuals.csv", index=False)
    print(f"Saved residuals.csv to {save_dir}")

    # npy 저장
    np.save(save_dir / "residuals.npy", full_data)
    print(f"Saved residuals.npy with subject and frame info to {save_dir}")

# 실행
save_predictions_and_residuals(model, test_loader, save_dir)
