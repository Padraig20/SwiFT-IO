import os
import glob
import torch
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

torch.multiprocessing.set_sharing_strategy('file_system')

class Dict2Class:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

print("Starting...")

# experiment_id와 경로 설정
experiment_id = "12rj2gw9"
source_input = "/data/scratch/kimbo/SwiFT-IO/output/moviefmri/"
source_output = "/data/scratch/kimbo/SwiFT-IO/analysis/3_scatter_plot"

# source_input 아래 experiment_id 폴더에서 체크포인트 파일 검색 (파일명: checkpt-epoch*ckpt)
ckpt_pattern = os.path.join(source_input, experiment_id, "checkpt-epoch*ckpt")
ckpt_files = glob.glob(ckpt_pattern)
if len(ckpt_files) == 0:
    raise FileNotFoundError(f"No checkpoint file found in {os.path.join(source_input, experiment_id)} with pattern 'checkpt-epoch*ckpt'")
# 여러 파일이 있다면 정렬 후 마지막 파일을 선택 (예: 최신 파일)
ckpt_files.sort()
pretrained_ckpt_path = ckpt_files[-1]
print(f"Using checkpoint: {pretrained_ckpt_path}")

emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

ckpt = torch.load(pretrained_ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
hparams = Dict2Class(**ckpt['hyper_parameters'])

# 설정 변경
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['batch_size'] = 1
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1
ckpt['hyper_parameters']['workers'] = 1
ckpt['hyper_parameters']['input_offset'] = 0
ckpt['hyper_parameters']['input_type'] = 'movieDM'
ckpt['hyper_parameters']['bad_subj_path'] = ''
ckpt['hyper_parameters']['limit_training_samples'] = 0
args = ckpt['hyper_parameters']

data_module = fMRIDataModule(**args)
data_module.setup()
data_module.prepare_data()
test_loader = data_module.test_dataloader()

from sklearn.preprocessing import StandardScaler
target_values = data_module.train_dataset.target_values
scaler = StandardScaler()
normalized_target_values = scaler.fit_transform(target_values)
print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')

model = LitClassifier(data_module=data_module, **args)
if torch.cuda.is_available():
    model.cuda(0)
model.load_state_dict(ckpt['state_dict'])

kwargs = {
    "nt_samples": 1,
    "nt_samples_batch_size": 1,
    "nt_type": "smoothgrad_sq",  # 1
    # "stdevs": 0.05,
    "internal_batch_size": 1,
}

all_subjects = []

model.eval()
for idx, data in enumerate(tqdm(test_loader), 0):
    input_ts = data['fmri_sequence'].float().cuda(0)
    pred = model.forward(input_ts)
    # inverse transform: 원래 스케일로 복원
    pred = pred * scaler.scale_[0] + scaler.mean_[0]
    target = data['target'].squeeze()

    all_subjects.append({
        "subject": data['subject_name'],
        "TR": data['TR'],
        "target": target.detach().cpu(),
        "prediction": pred.detach().cpu()
    })

unique_subject_names = np.unique([subj['subject'] for subj in all_subjects])
one_subject = [subject for subject in all_subjects if subject['subject'][0] == unique_subject_names[0]]
one_subject_sorted = sorted(one_subject, key=lambda x: x['TR'])
trs = [frame['TR'] for frame in one_subject_sorted]
targets = [frame['target'] for frame in one_subject_sorted]
targets = torch.stack(targets)
if len(targets.shape) == 3:
    target = rearrange(targets, 'b t c -> (b t) c')
else:
    target = targets.flatten().unsqueeze(1)

num_targets = target.shape[1]
time = torch.arange(0, target.shape[0], 1)

# source_output 아래 experiment_id 폴더 생성
save_dir = os.path.join(source_output, experiment_id)
os.makedirs(save_dir, exist_ok=True)

for i in range(num_targets):
    plt.figure(figsize=(12, 6))
    for subj in unique_subject_names:
        subj_frames = [subject for subject in all_subjects if subject['subject'][0] == subj]
        subj_frames_sorted = sorted(subj_frames, key=lambda x: x['TR'])
        subj_predictions = [frame['prediction'] for frame in subj_frames_sorted]
        subj_predictions = torch.stack(subj_predictions)
        if len(subj_predictions.shape) == 3:
            subj_predictions = rearrange(subj_predictions, 'b t c -> (b t) c').numpy()
        else:
            subj_predictions = subj_predictions.flatten().unsqueeze(1).numpy()
        if subj_predictions.shape[0] != target.shape[0]:
            print(f"Skipping subject {subj} because of shape mismatch {subj_predictions.shape[0]} vs {target.shape[0]}")
            continue
        subj_predictions = subj_predictions[:, i]
        plt.plot(time, subj_predictions, label=f"Subject {subj}", alpha=0.5)

    plt.plot(time, target[:, i].numpy(), label="Target", linestyle="dashed", alpha=1, color='black', linewidth=1.5)

    for tr in trs:  # TR 표시 선
        plt.axvline(x=tr, color='black', linestyle='dotted', linewidth=1, alpha=0.5)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Predicted vs Target {emotions[i]}")
    plt.grid(True)

    filename = os.path.join(save_dir, f"timeseries_{emotions[i]}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("Done!")
