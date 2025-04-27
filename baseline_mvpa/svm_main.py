import torch
import glob
import numpy as np
import os
from sklearn.svm import SVR
from nilearn.decoding import FREMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import nibabel as nib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Parameters 설정 =========
input_type = 'movieDM'
seq_length = 50

if input_type == 'movieDM':
    total_length = 750
elif input_type == 'movieTP':
    total_length = 360
assert total_length is not None, "total_length must be defined."

# === 파일 경로 정의 ===
split_file_path = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/output/moviefmri/tubg3tim/split_fixed_1.txt'
emotion_timeseries_file = "/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/metadata/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv"
data_base_dir = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img"

affine_path = Path("/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed.nii.gz")
affine = nib.load(affine_path).affine

# === Subject 순서 및 그룹 로딩 ===
with open(split_file_path, "r") as f:
    subject_order = f.read().splitlines()

train_index = np.argmax(["train" in line for line in subject_order])
val_index = np.argmax(["val" in line for line in subject_order])
test_index = np.argmax(["test" in line for line in subject_order])

train_names = subject_order[train_index + 1 : val_index]
val_names = subject_order[val_index + 1 : test_index]
test_names = subject_order[test_index + 1 :]

train_names = train_names[:10]
val_names = val_names[:10]
test_names = test_names[:10]

print("Length of Train subjects:", len(train_names))
print("Length of Validation subjects:", len(val_names))
print("Length Test subjects:", len(test_names))

# === Emotion Timeseries 데이터 로딩 ===
print('------Loading emotion outcome------')
emotion_df = pd.read_csv(emotion_timeseries_file)
emotion_vars = ['Positive', 'Negative']
emotion_df = emotion_df[emotion_vars]

target = 'Positive'  # 원하는 타겟 설정

# === 타겟 값 생성 ===
window_size = seq_length
num_windows = total_length // window_size
target_values = [emotion_df[target].iloc[i:i + window_size].mean() for i in range(0, num_windows * window_size, window_size)]
target_values = np.array(target_values)

# =========== 데이터 로딩 함수 ===========
def load_4d_fmri_from_pt(subject, base_dir, total_length, seq_length, affine):
    subject_dir = os.path.join(base_dir, subject)
    frame_paths = sorted(
        glob.glob(os.path.join(subject_dir, 'frame_*.pt')),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )

    if len(frame_paths) < total_length:
        raise ValueError(f"{subject}: Expected at least {total_length} frames, found {len(frame_paths)} frames.")

    fmri_data = []
    for frame_path in frame_paths[:total_length]:
        data = torch.load(frame_path).float()

        if data.ndim == 4:
            data = data.squeeze(-1)

        if data.shape != (96, 96, 96):
            if data.shape == (81, 95, 81):
                background_value = data.flatten()[0].item()
                data = data.unsqueeze(0).unsqueeze(0)

                data = torch.nn.functional.pad(
                    data,
                    (7, 8, 0, 1, 7, 8),
                    value=background_value
                ).squeeze(0).squeeze(0)

                if data.shape != (96, 96, 96):
                    raise ValueError(f"{frame_path}: padding failed, resulting shape {data.shape}")
            else:
                raise ValueError(f"{frame_path}: unexpected shape {data.shape}")

        fmri_data.append(data.numpy().astype(np.float32))

    fmri_data = np.stack(fmri_data, axis=-1)
    fmri_sequences = np.split(fmri_data, num_windows, axis=-1)

    niimg_sequences = [nib.Nifti1Image(seq.mean(axis=-1).astype(np.float32), affine) for seq in fmri_sequences]

    return niimg_sequences

# === 데이터 로딩 ===
def prepare_data(names):
    X, y = [], []
    for subj in names:
        niimg_seqs = load_4d_fmri_from_pt(subj, data_base_dir, total_length, seq_length, affine)
        X.extend(niimg_seqs)
        y.extend(target_values)
    return X, np.array(y)

print('------Loading Input data------')
print('------1. train -------')
X_train, y_train = prepare_data(train_names)
print('------2. validation -------')
X_val, y_val = prepare_data(val_names)
print('------3. test -------')
X_test, y_test = prepare_data(test_names)

print('------SVM regression-----')
decoder = FREMRegressor(
    estimator=SVR(kernel='linear'),
    standardize='zscore_sample',
    smoothing_fwhm=None,
    screening_percentile=20,
    scoring='neg_mean_squared_error',
    mask=None
)

# 모델 학습
decoder.fit(X_train, y_train)

# 평가
train_pred = decoder.predict(X_train)
val_pred = decoder.predict(X_val)
test_pred = decoder.predict(X_test)

for name, true, pred in zip(['Train', 'Validation', 'Test'], [y_train, y_val, y_test], [train_pred, val_pred, test_pred]):
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{name} MSE: {mse:.4f}, R²: {r2:.4f}")