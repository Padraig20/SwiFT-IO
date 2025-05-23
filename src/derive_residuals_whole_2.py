import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# ======== 설정값 ========
run_id = "tubg3tim"
input_type = "movieDM"
seq_length = 50
input_offset = 3
total_frames = 750  # 전체 프레임 수

# ======== 감정 레이블 ========
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

# ======== 경로 설정 ========
project_root = Path("/pscratch/sd/k/kimbo/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule
from torch.utils.data import DataLoader, ConcatDataset

# ======== 체크포인트 로드 ========
project_id = "moviefmri"
ckpt_dir = project_root / f"output/{project_id}/{run_id}"
ckpt_path = list(ckpt_dir.glob("checkpt*"))[0]
ckpt = torch.load(ckpt_path, map_location='cpu')

# ======== 하이퍼파라미터 설정 ========
args = ckpt['hyper_parameters']
args.update({
    'input_type': input_type,
    'seq_length': seq_length,
    'image_path': "/global/cfs/cdirs/m4750/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
    'default_root_dir': str(project_root / "output/moviefmri"),
    'shuffle_time_sequence': False,
    'time_as_channel': False,
    'eval_batch_size': 1,
    'input_offset': input_offset,
    'bad_subj_path': None,
    'limit_training_samples': 0,
    'img_size': [96, 96, 96, seq_length],
    'eval_num_workers': 1
})

# # ======== 데이터 모듈 초기화 ========
# data_module = fMRIDataModule(**args)

# # 🔥 split 파일을 무시하고 모든 참가자 데이터셋을 불러오기 위해 커스텀 데이터로더 구성
# data_module.setup(stage='fit')  # 원본 데이터셋 전체 로딩

# # 전체 데이터셋 합치기 (train+val+test)
# full_dataset = ConcatDataset([
#     data_module.train_set,
#     data_module.val_set,
#     data_module.test_set
# ])
# ======== 데이터 모듈 초기화 ========
dummy_split_path = '/pscratch/sd/k/kimbo/SwiFT-IO/tmp/7_checkpoint/kimbo_aica/split_fixed_1.txt'
data_module = fMRIDataModule(**args)
data_module.split_file_path = str(dummy_split_path)
data_module.prepare_data()
data_module.setup(stage='fit')  # 추가된 부분 🚩

# 전체 데이터셋 합치기 (train+val+test)
full_dataset = ConcatDataset([
    data_module.train_set,
    data_module.val_set,
    data_module.test_set
])


# 모든 참가자를 한 번에 처리하는 로더 생성
whole_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=1)

# ======== 모델 초기화 ========
model = LitClassifier(data_module=data_module, **args)
model.load_state_dict(ckpt['state_dict'])
model.eval().cpu()

# ======== 결과 저장 경로 ========
save_dir = project_root / f"analysis/residuals/{run_id}"
save_dir.mkdir(parents=True, exist_ok=True)

# ======== segment 개수 계산 ========
max_segments = (total_frames - input_offset) // seq_length
print(f"✅ 각 subject당 segment 개수: {max_segments}개")

# ======== residual 및 prediction 저장 함수 ========
def save_predictions_and_residuals(model, loader, save_dir, seq_length, input_offset, max_segments):
    rows = []

    full_data = {"split": [], "subject": [], "start_frame": [], "end_frame": []}
    for i, label in enumerate(emotion_labels):
        full_data[f"residual_{i}_{label}"] = []
        full_data[f"prediction_{i}_{label}"] = []

    print(f"Processing all subjects ({len(loader)} batches)...")

    segment_counter = {}

    for idx, data in enumerate(tqdm(loader)):
        subj_name = data['subject_name'][0]
        seg_idx = segment_counter.get(subj_name, 0)

        if seg_idx >= max_segments:
            continue  # 최대 segment 수를 초과하면 넘어감

        input_ts = data['fmri_sequence'].float().cpu()
        target = data['target'].float().cpu()

        with torch.no_grad():
            pred = model(input_ts)

        residual = torch.abs(pred - target.squeeze(0))

        start = seg_idx * seq_length
        input_start = start + input_offset
        end = input_start + seq_length - 1
        segment_counter[subj_name] = seg_idx + 1

        residual_np = residual.numpy()
        pred_np = pred.numpy()

        row = {"split": "whole", "subject": subj_name, "start_frame": input_start, "end_frame": end}

        full_data["split"].append("whole")
        full_data["subject"].append(subj_name)
        full_data["start_frame"].append(input_start)
        full_data["end_frame"].append(end)

        for i, emotion in enumerate(emotion_labels):
            row[f"residual_{i}_{emotion}"] = residual_np[:, i].tolist()
            row[f"prediction_{i}_{emotion}"] = pred_np[:, i].tolist()
            full_data[f"residual_{i}_{emotion}"].append(residual_np[:, i].tolist())
            full_data[f"prediction_{i}_{emotion}"].append(pred_np[:, i].tolist())

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "residuals_whole_2.csv", index=False)
    np.save(save_dir / "residuals_whole_2.npy", full_data)

    print(f"✅ Saved residuals_whole.csv and residuals_whole.npy to {save_dir}")

# ======== 실행 ========
save_predictions_and_residuals(model, whole_loader, save_dir, seq_length, input_offset, max_segments)
print("✅ 전체 segment에 대한 residuals 및 predictions 저장 완료")
