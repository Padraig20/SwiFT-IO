import torch
import glob
import numpy as np
import os
from sklearn.svm import SVR
from nilearn.decoding import FREMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import nibabel as nib
from nilearn.image import resample_to_img
import pandas as pd
from pathlib import Path
import joblib

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
affine_img = nib.load(affine_path)
affine = affine_img.affine

# === Mask 로딩 및 resampling ===
brain_mask_path = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
brain_mask_img = nib.load(brain_mask_path)
resampled_mask = resample_to_img(brain_mask_img, affine_img, interpolation='nearest')

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

# === Emotion Timeseries 데이터 로딩 ===
emotion_df = pd.read_csv(emotion_timeseries_file)
emotion_vars = ['Positive', 'Negative']
emotion_df = emotion_df[emotion_vars]

target = 'Positive'

# === 타겟 값 생성 ===
window_size = seq_length
num_windows = total_length // window_size
target_values = [emotion_df[target].iloc[i:i + window_size].mean() for i in range(0, num_windows * window_size, window_size)]
target_values = np.array(target_values)

# === 데이터 로딩 함수 ===
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
                    data, (7, 8, 0, 1, 7, 8), value=background_value
                ).squeeze(0).squeeze(0)

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

X_train, y_train = prepare_data(train_names)
X_val, y_val = prepare_data(val_names)
X_test, y_test = prepare_data(test_names)

# === GridSearch 파라미터 세트 정의 ===
param_grid = {
    'estimator__C': [0.1, 1, 10],
    'estimator__epsilon': [0.01, 0.1],
}

# === GridSearchCV 객체 생성 ===
grid_decoder = GridSearchCV(
    FREMRegressor(
        estimator=SVR(kernel='linear'),
        standardize='zscore_sample',
        smoothing_fwhm=None,
        screening_percentile=20,
        scoring='neg_mean_squared_error',
        mask=resampled_mask
    ),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# === GridSearchCV로 모델 학습 ===
grid_decoder.fit(X_train, y_train)

# === 최적의 하이퍼파라미터 출력 ===
print("Best parameters:", grid_decoder.best_params_)

# === 최적의 모델 저장 ===
joblib.dump(grid_decoder.best_estimator_, 'best_frem_regressor_model.joblib')

# === 평가 ===
for name, X, y in zip(['Train', 'Validation', 'Test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
    pred = grid_decoder.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name} MSE: {mse:.4f}, R²: {r2:.4f}")
