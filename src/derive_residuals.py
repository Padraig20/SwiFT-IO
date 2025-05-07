import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import pdb
import wandb
# --run_id pqslufat --input_type movieDM --seq_length 30 --input_offset 3 --dataset test
# --run_id pqslufat --input_type movieDM --seq_length 30 --input_offset 3 --dataset val
# --run_id pqslufat --input_type movieDM --seq_length 30 --input_offset 3 --dataset train
# --run_id pqslufat --input_type movieDM --seq_length 30 --input_offset 3 --dataset whole
# CLI 인자 받기
parser = argparse.ArgumentParser(description="Extract residuals from model predictions and save with frame info.")
parser.add_argument('--run_id', type=str, required=True, help='Run ID (e.g., cvy8kv4o, tubg3tim)')
parser.add_argument('--input_type', type=str, choices=['movieDM', 'movieTP'], required=True, help='Input type')
parser.add_argument('--seq_length', type=int, default=30, help='Sequence length (default: 30)')
parser.add_argument('--input_offset', type=int, default=0, help='Input offset (default: 0)')
parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'whole'], default='whole', help='Dataset type (default: whole)')
args_cli = parser.parse_args()

run_id = args_cli.run_id
input_type = args_cli.input_type
seq_length = args_cli.seq_length
input_offset = args_cli.input_offset
dataset_choice = args_cli.dataset

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO")
project_main = project_root / "src"
sys.path.append(str(project_main))

from module.models.encoder.swin4d_transformer_ver7 import SwinTransformer4D
from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

project_id = "moviefmri"
ckpt_dir = project_root / f"output/{project_id}/{run_id}"
ckpt_path = list(ckpt_dir.glob("checkpt*"))[0]
ckpt = torch.load(ckpt_path, map_location='cpu')

ckpt['hyper_parameters'].update({
    'input_type': input_type,
    'seq_length': seq_length,
    'image_path': "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
    'default_root_dir': str(project_root / "output/moviefmri"),
    'shuffle_time_sequence': False,
    'time_as_channel': False,
    'eval_batch_size': 1,
    'input_offset': input_offset,
    'bad_subj_path': None,
    'limit_training_samples': 0,
    'img_size': [96, 96, 96, seq_length],
    'eval_num_workers': 4
})

args = ckpt['hyper_parameters']

data_module = fMRIDataModule(**args)
data_module.setup()
data_module.prepare_data()

model = LitClassifier(data_module=data_module, **args)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model.cpu()

save_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/anal_resid/residuals/{run_id}")
save_dir.mkdir(parents=True, exist_ok=True)

def save_predictions_and_residuals_with_split(model, data_module, save_dir, seq_length, input_offset=3, dataset_choice='whole', save_interval=0.1):
    loaders = {
        'train': data_module.train_dataloader(),
        'val': data_module.val_dataloader(),
        'test': data_module.test_dataloader()
    } if dataset_choice == 'whole' else {dataset_choice: getattr(data_module, f"{dataset_choice}_dataloader")()}

    rows, segment_counter = [], {}
    full_data = {"split": [], "subject": [], "start_frame": [], "end_frame": []}

    for i, label in enumerate(emotion_labels):
        full_data[f"residual_{i}_{label}"] = []
        full_data[f"prediction_{i}_{label}"] = []

    for split_name, loader in loaders.items():
        total_batches = len(loader)
        save_batch_interval = max(int(total_batches * save_interval), 1)
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

            residual_np = residual.numpy()
            pred_np = pred.numpy()

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

            if (idx + 1) % save_batch_interval == 0 or (idx + 1) == total_batches:
                percent_done = int(100 * (idx + 1) / total_batches)
                interim_df = pd.DataFrame(rows)
                interim_df.to_csv(save_dir / f"residuals_{dataset_choice}_{percent_done}pct.csv", index=False)
                np.save(save_dir / f"residuals_{dataset_choice}_{percent_done}pct.npy", full_data)
                print(f"✅ Interim results ({percent_done}%) saved.")

    print(f"All results saved to {save_dir}")

save_predictions_and_residuals_with_split(
    model, data_module, save_dir, seq_length,
    input_offset=input_offset, dataset_choice=dataset_choice
)
