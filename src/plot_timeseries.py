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

emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

pretrained_ckpt_path='/data/scratch/patrickstyll/SwiFT-IO/output/HBN/HBN-55/checkpt-epoch=37-valid_mse=0.03.ckpt'
ckpt = torch.load(pretrained_ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
hparams = Dict2Class(**ckpt['hyper_parameters'])

ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['batch_size'] = 1
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1
ckpt['hyper_parameters']['workers'] = 1
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
model.cuda(0) if torch.cuda.is_available() else model
model.load_state_dict(ckpt['state_dict'])

kwargs = {
    "nt_samples": 1,
    "nt_samples_batch_size": 1,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 1,
}

all_subjects = []

model.eval()
for idx, data in enumerate(tqdm(test_loader),0):
    input_ts = data['fmri_sequence'].float().cuda(0)

    pred = model.forward(input_ts)
    pred = (pred - scaler.mean_[0]) / (scaler.scale_[0])
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

    plt.plot(time, target[:, i].numpy(), label="Target", linestyle="dashed", alpha=1, color='black', linewidth=2.5)

    for tr in trs: # to track changes when stitching together TRs
        plt.axvline(x=tr, color='black', linestyle='dotted', linewidth=3)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Predicted vs Target {emotions[i]}")
    plt.grid(True)

    filename = f"timeseries_{emotions[i]}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

print("Done!")