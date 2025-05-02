import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
from torch.utils.data import DataLoader

# ===========================
# Example Usage:
# python derive_residuals.py --run_id tubg3tim --input_type movieDM --seq_length 50 --data test
# ===========================

# CLI Setup
parser = argparse.ArgumentParser(description="Extract residuals from model predictions and save with frame info.")
parser.add_argument('--run_id',
                    type=str,
                    required=True,
                    help='Run ID (e.g., cvy8kv4o)')
parser.add_argument('--input_type',
                    type=str,
                    choices=['movieDM', 'movieTP'],
                    required=True,
                    help='Input type')
parser.add_argument('--seq_length',
                    type=int,
                    default=50,
                    help='Sequence length (default: 50)')
parser.add_argument('--data',
                    type=str,
                    required=True,
                    choices=['train', 'val', 'test'],
                    help='Data type to load (train, val, test)')
args_cli = parser.parse_args()

RUN_ID = args_cli.run_id
INPUT_TYPE = args_cli.input_type
SEQ_LENGTH = args_cli.seq_length
DATA_SPLIT = args_cli.data

EMOTION_LABELS = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

PROJECT_ROOT = Path("/scratch/connectome/patrickstyll/kimbo/SwiFT-IO")
PROJECT_MAIN = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_MAIN))

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

# checkpoint-specific
PROJECT_ID = "moviefmri"
CKPT_DIR = PROJECT_ROOT / f"output/{PROJECT_ID}/{RUN_ID}"
CKPT_PATH = list(CKPT_DIR.glob("checkpt*"))[0]

ckpt = torch.load(CKPT_PATH, map_location='cpu')

# override hyperparams of checkpoint
# TODO where do these values come from? e.g. input_offset is 3, why? Why would it need to be overridden?
ckpt['hyper_parameters']['input_type'] = INPUT_TYPE
ckpt['hyper_parameters']['seq_length'] = SEQ_LENGTH
ckpt['hyper_parameters']['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
ckpt['hyper_parameters']['default_root_dir'] = str(PROJECT_ROOT / "output/moviefmri")
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1
#ckpt['hyper_parameters']['input_offset'] = 3
ckpt['hyper_parameters']['bad_subj_path'] = None
ckpt['hyper_parameters']['limit_training_samples'] = 0
ckpt['hyper_parameters']['img_size'] = [96, 96, 96, SEQ_LENGTH]
args = ckpt['hyper_parameters']

# data loading
data_module = fMRIDataModule(**args)
data_module.setup()
data_module.prepare_data()

if DATA_SPLIT == "train":
    test_loader = data_module.train_dataloader()
elif DATA_SPLIT == "val":
    test_loader = data_module.val_dataloader()[0]
elif DATA_SPLIT == "test":
    test_loader = data_module.test_dataloader()
else:
    raise ValueError(f"Invalid data split: {DATA_SPLIT}. Choose from 'train', 'val', or 'test'.")

# model loading
model = LitClassifier(data_module=data_module, **args)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model.cpu()

# result directory
save_dir = Path(f"/scratch/connectome/patrickstyll/kimbo/SwiFT-IO/results_each/{RUN_ID}")
save_dir.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def save_predictions_and_residuals(model: LitClassifier,
                                   test_loader: DataLoader,
                                   save_dir: Path) -> None:
    """ Save model predictions and residuals to CSV and npy files.
    Args:
        model (LitClassifier): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        save_dir (Path): Directory to save the results.
    """
    
    # pre-allocate space
    
    rows = [] # corresponds to each segment of a subject
    segment_counter = {}
    
    full_data = {
        "subject": [],
        "start_frame": [],
        "end_frame": [],
    }
    
    for i, label in enumerate(EMOTION_LABELS):
        full_data[f"residual_{i}_{label}"] = []
        full_data[f"prediction_{i}_{label}"] = []

    for data in tqdm(test_loader, desc="Processing test data", unit="sequence"): # batch size = 1
        # data['fmri_sequence'] shape: (1, 96, 96, 96, SEQ_LENGTH)
        subj_name = data['subject_name'][0]
        input_ts = data['fmri_sequence'].float().cpu()
        target = data['target'].float().cpu()  # shape: (1, SEQ_LENGTH, EMOTIONS)

        pred = model(input_ts)  # shape: (SEQ_LENGTH, EMOTIONS)
        residual = torch.abs(pred - target.squeeze(0))  # shape: (SEQ_LENGTH, EMOTIONS)

        # segment the data
        # segment_counter keeps track of how many segments have been processed for each subject
        seg_idx = segment_counter.get(subj_name, 0)
        start = seg_idx * SEQ_LENGTH
        end = start + SEQ_LENGTH - 1
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

        # save residuals and predictions for each emotion
        for i, emotion in enumerate(EMOTION_LABELS):
            r_list = residual_np[:, i].tolist()
            p_list = pred_np[:, i].tolist()
            row[f"residual_{i}_{emotion}"] = r_list
            row[f"prediction_{i}_{emotion}"] = p_list
            full_data[f"residual_{i}_{emotion}"].append(r_list)
            full_data[f"prediction_{i}_{emotion}"].append(p_list)

        rows.append(row)

    # save as csv
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / f"residuals_{DATA_SPLIT}.csv", index=False)
    print(f"Saved residuals_{DATA_SPLIT}.csv to {save_dir}")

    # save as npy file
    np.save(save_dir / f"residuals_{DATA_SPLIT}.npy", full_data)
    print(f"Saved residuals_{DATA_SPLIT}.npy with subject and frame info to {save_dir}")

save_predictions_and_residuals(model, test_loader, save_dir)
