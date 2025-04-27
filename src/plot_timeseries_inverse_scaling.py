# plot_timeseries_inverse_scaling.py
# ─────────────── fMRI emotion-regression: test-set timeseries plot ───────────────
# 실행 전 확인:
#   1) module.pl_classifier.LitClassifier
#   2) module.utils.data_module.fMRIDataModule
#   3) 이미지·타깃 폴더 경로가 실제 환경과 일치하는지
# ------------------------------------------------------------------------------
import os
import glob
import torch
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from module.pl_classifier import LitClassifier
from module.utils.data_module import fMRIDataModule

torch.multiprocessing.set_sharing_strategy("file_system")

# ───────────────────────────── 설정 ─────────────────────────────
experiment_id = "tubg3tim"
source_input  = "/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/output/moviefmri"
source_output = "/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/3_scatter_plot"
image_path    = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
emotions      = ["Anger", "Happy", "Fear", "Sad", "Excited", "Positive", "Negative"]

# ─────────────────── 체크포인트 로드 ───────────────────
ckpt_files = sorted(glob.glob(os.path.join(source_input,
                                           experiment_id,
                                           "checkpt-epoch*ckpt")))
if not ckpt_files:
    raise FileNotFoundError("No checkpoint found.")
ckpt_path = ckpt_files[-1]
print(f"Using checkpoint: {ckpt_path}")

ckpt   = torch.load(ckpt_path, map_location="cpu")
hparam = ckpt["hyper_parameters"]

# 하이퍼파라미터 런타임 재설정
hparam.update({
    "shuffle_time_sequence" : False,
    "batch_size"            : 1,
    "eval_batch_size"       : 1,
    "workers"               : 1,
    "time_as_channel"       : False,
    "input_offset"          : 0,
    "input_type"            : "movieDM",
    "bad_subj_path"         : "",
    "limit_training_samples": 0,
    "image_path"            : image_path,
    "default_root_dir"      : str(Path(source_input)),
    "eval_num_workers": 0
})

# ───────────────────── DataModule ─────────────────────
data_module = fMRIDataModule(**hparam)

data_module.prepare_data()

data_module.setup(stage="test")                # test split 로드
test_loader = data_module.test_dataloader()

data_module.setup(stage="fit")                 # train/val split 로드
scaler = StandardScaler().fit(
    data_module.train_dataset.target_values)   # train 통계로 딱 한번 fit
print(f"[Scaler] mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")


# ───────────────────── 모델 ─────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = LitClassifier(data_module=data_module, **hparam).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ───────────────────── 추론 ─────────────────────
all_subjects = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predict"):
        # ---- subject 이름 추출 ----
        subj_raw = batch["subject_name"]          # 예: ['sub-0001']  또는 tensor(['sub-0001'])
        if isinstance(subj_raw, (list, tuple, np.ndarray)):
            subj = str(subj_raw[0])
        elif torch.is_tensor(subj_raw):
            subj = str(subj_raw[0])
        else:
            subj = str(subj_raw)

        input_ts = batch["fmri_sequence"].to(device, dtype=torch.float32)
        pred     = model(input_ts)
        pred     = pred * scaler.scale_[0] + scaler.mean_[0]  # 역정규화

        all_subjects.append({
            "subject"   : batch["subject_name"],
            "TR"        : batch["TR"],
            "target"    : batch["target"].squeeze().cpu(),
            "prediction": pred.cpu(),
        })

# ───────────────────── 시각화 준비 ─────────────────────
# unique_subjects = np.unique([d["subject"] for d in all_subjects])
unique_subjects = np.unique([str(d["subject"]) for d in all_subjects])

ref_subject     = unique_subjects[0]                        # 첫 번째 subject 기준 time index
ref_frames      = sorted([d for d in all_subjects if d["subject"] == ref_subject],
                         key=lambda x: x["TR"])
trs             = [f["TR"] for f in ref_frames]

targets = torch.stack([f["target"] for f in ref_frames])
targets = rearrange(targets, "b t -> (b t) 1") if targets.ndim == 2 else \
          rearrange(targets, "b t c -> (b t) c")
num_targets = targets.shape[1]
time_axis   = torch.arange(targets.shape[0])

# 결과 저장 디렉터리
save_dir = os.path.join(source_output, experiment_id)
os.makedirs(save_dir, exist_ok=True)

# ───────────────────── 시각화 ─────────────────────
for i in range(num_targets):
    plt.figure(figsize=(12, 6))

    for subj in unique_subjects:
        subj_frames = sorted([d for d in all_subjects if d["subject"] == subj],
                             key=lambda x: x["TR"])
        subj_preds  = torch.stack([f["prediction"] for f in subj_frames])
        if subj_preds.ndim == 3:
            subj_preds = rearrange(subj_preds, "b t c -> (b t) c")
        else:
            subj_preds = rearrange(subj_preds, "b -> b 1")
        if subj_preds.shape[0] != targets.shape[0]:
            print(f"[Skip] {subj}: length mismatch")
            continue
        plt.plot(time_axis, subj_preds[:, i], alpha=0.5, label=f"{subj}")

    plt.plot(time_axis,
             targets[:, i].numpy(),
             color="black",
             linestyle="dashed",
             linewidth=1.5,
             label="Target")

    for tr in trs:
        plt.axvline(tr, color="black", linestyle="dotted", linewidth=1, alpha=0.5)

    plt.xlabel("Time (TR)")
    plt.ylabel("Value")
    plt.title(f"Predicted vs Target – {emotions[i]}")
    plt.grid(True)
    plt.legend(fontsize=6, ncol=4, loc="upper center")

    out_png = os.path.join(save_dir, f"timeseries_{emotions[i]}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

print("Done!")
