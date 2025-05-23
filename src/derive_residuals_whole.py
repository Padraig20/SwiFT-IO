import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import pdb

# CLI 인자 받기
parser = argparse.ArgumentParser(description="Extract residuals from model predictions and save with frame info.")
parser.add_argument('--run_id', type=str, required=True, help='Run ID (e.g., cvy8kv4o, tubg3tim)')
parser.add_argument('--input_type', type=str, choices=['movieDM', 'movieTP'], required=True, help='Input type')
parser.add_argument('--seq_length', type=int, default=50, help='Sequence length (default: 50)')
parser.add_argument('--input_offset', type=int, default=0, help='input offset (default: 0)')
parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'whole'], default='whole', help='Dataset type to process (default: whole)')
# --run_id tubg3tim --input_type movieDM --seq_length 50 --input_offset 3 --dataset test
# --run_id tubg3tim --input_type movieDM --seq_length 50 --input_offset 3 --dataset val
# --run_id tubg3tim --input_type movieDM --seq_length 50 --input_offset 3 --dataset train
# --run_id tubg3tim --input_type movieDM --seq_length 50 --input_offset 3 --dataset whole
args_cli = parser.parse_args()

run_id = args_cli.run_id
input_type = args_cli.input_type
seq_length = args_cli.seq_length
input_offset = args_cli.input_offset
dataset_choice = args_cli.dataset

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
ckpt['hyper_parameters']['input_offset'] = input_offset
ckpt['hyper_parameters']['bad_subj_path'] = None
ckpt['hyper_parameters']['limit_training_samples'] = 0
ckpt['hyper_parameters']['img_size'] = [96, 96, 96, seq_length]
ckpt['hyper_parameters']['eval_num_workers'] = 1

args = ckpt['hyper_parameters']

# 데이터 모듈 초기화
data_module = fMRIDataModule(**args)
# pdb.set_trace()

# custom_split_path = '/pscratch/sd/k/kimbo/SwiFT-IO/output/moviefmri/tubg3tim/split_fixed_1.txt'
# custom_split_path = '/pscratch/sd/k/kimbo/kimbo_aica/kimbo/SwiFT-IO/data/splits/HBN/split_fixed_1.txt'
# custom_split_path = '/pscratch/sd/k/kimbo/kimbo_aica/kimbo/SwiFT-IO/src/data/splits/HBN/split_fixed_1.txt'
# custom_split_path = '/pscratch/sd/k/kimbo/SwiFT-IO/output/moviefmri/tubg3tim/labserver/split_fixed_1.txt'
custom_split_path = '/pscratch/sd/k/kimbo/SwiFT-IO/tmp/7_checkpoint/kimbo_aica/split_fixed_1.txt'
data_module.split_file_path = custom_split_path

data_module.setup()
data_module.prepare_data()

# 모델 초기화
model = LitClassifier(data_module=data_module, **args)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model.cpu()

# 결과 저장 경로
save_dir = Path(f"/pscratch/sd/k/kimbo/SwiFT-IO/analysis/residuals/{run_id}")
save_dir.mkdir(parents=True, exist_ok=True)


def save_predictions_and_residuals_with_split(model, data_module, save_dir, seq_length, input_offset=3, dataset_choice='whole'):
    
    # 여기서 데이터셋을 선택해서 처리할 수 있도록 변경
    if dataset_choice == 'train':
        loaders = {'train': data_module.train_dataloader()}
    elif dataset_choice == 'val':
        loaders = {'val': data_module.val_dataloader()}
    elif dataset_choice == 'test':
        loaders = {'test': data_module.test_dataloader()}
    elif dataset_choice == 'whole':
        loaders = {
            'train': data_module.train_dataloader(),
            'val': data_module.val_dataloader(),
            'test': data_module.test_dataloader()
        }

    rows = []
    segment_counter = {}

    full_data = {"split": [], "subject": [], "start_frame": [], "end_frame": []}

    for i, label in enumerate(emotion_labels):
        full_data[f"residual_{i}_{label}"] = []
        full_data[f"prediction_{i}_{label}"] = []

    for split_name, loader in loaders.items():
        print(f"Processing split: {split_name}")
        for idx, data in enumerate(tqdm(loader)):
            subj_name = data['subject_name'][0]
            input_ts = data['fmri_sequence'].float().cpu()
            target = data['target'].float().cpu()

            with torch.no_grad(): 
                pred = model(input_ts)

            residual = torch.abs(pred - target.squeeze(0))

            seg_idx = segment_counter.get((split_name, subj_name), 0)
            start = seg_idx * seq_length
            input_start = start + input_offset
            end = input_start + seq_length - 1
            segment_counter[(split_name, subj_name)] = seg_idx + 1

            residual_np = residual.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            row = {"split": split_name, "subject": subj_name, "start_frame": input_start, "end_frame": end}

            full_data["split"].append(split_name)
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
    df.to_csv(save_dir / f"residuals_{dataset_choice}.csv", index=False)
    np.save(save_dir / f"residuals_{dataset_choice}.npy", full_data)

    print(f"Saved residuals_{dataset_choice}.csv and residuals_{dataset_choice}.npy to {save_dir}")


# 실행
save_predictions_and_residuals_with_split(model, data_module, save_dir, seq_length, input_offset=input_offset, dataset_choice=dataset_choice)
