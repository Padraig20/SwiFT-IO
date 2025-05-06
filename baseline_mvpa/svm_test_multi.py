#!/usr/bin/env python
# svm_test_multi.py
"""
Baseline MVPA: Multi-output SVM regression (Positive, Negative, Anger, Happy, Fear, Sad, Excited).
Subjects with insufficient frames (movieDM < 750 or movieTP < 360) are excluded.
Allows specifying seq_length, input_type, and input_offset from command line.
"""

# ========= Imports =========
import os, glob, argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nilearn.decoding import FREMRegressor
from nilearn.image import resample_to_img
import joblib

# ========= Command Line Arguments =========
parser = argparse.ArgumentParser(description='Baseline MVPA: Multi-output SVR')
parser.add_argument('--input_type', choices=['movieDM', 'movieTP'], default='movieDM', help='Type of movie input')
parser.add_argument('--seq_length', type=int, default=50, help='Number of frames per window')
parser.add_argument('--input_offset', type=int, default=3, help='Input offset to adjust HRF delay (frames)')
args = parser.parse_args()

input_type = args.input_type
seq_length = args.seq_length
input_offset = args.input_offset
total_length = 750 if input_type == 'movieDM' else 360

# ========= File paths =========
split_file_path = '/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/data/splits/HBN/split_fixed_1.txt'
emotion_timeseries_csv = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/metadata/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv'
data_base_dir = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img'

affine_img_path = Path('/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed.nii.gz')
brain_mask_path = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

# ========= Load affine & mask =========
affine_img = nib.load(affine_img_path)
affine = affine_img.affine
brain_mask_img = nib.load(brain_mask_path)
resampled_mask = resample_to_img(brain_mask_img, affine_img, interpolation='nearest')

# ========= Subject split =========
with open(split_file_path) as f:
    lines = f.read().splitlines()

train_idx = lines.index('train_subjects')
val_idx = lines.index('val_subjects')
test_idx = lines.index('test_subjects')

train_names = lines[train_idx+1 : val_idx]
val_names = lines[val_idx+1 : test_idx]
test_names = lines[test_idx+1 :]

# ========= Filter subjects =========
usable_length = total_length - input_offset
num_windows = usable_length // seq_length
required_frames = input_offset + num_windows * seq_length

def filter_subjects(names, base_dir, required_frames):
    kept = []
    for subj in names:
        n_frames = len(glob.glob(os.path.join(base_dir, subj, 'frame_*.pt')))
        if n_frames >= required_frames:
            kept.append(subj)
        else:
            print(f'⏩ Excluded {subj} ({n_frames}/{required_frames} frames)')
    return kept

train_names = filter_subjects(train_names, data_base_dir, required_frames)
val_names = filter_subjects(val_names, data_base_dir, required_frames)
test_names = filter_subjects(test_names, data_base_dir, required_frames)

print(f'Subject counts after filtering ➜ Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}')

# ========= Emotion targets (with offset) =========
emotion_vars = ['Positive','Negative','Anger','Happy','Fear','Sad','Excited']
emotion_df = pd.read_csv(emotion_timeseries_csv)[emotion_vars]

target_vals = np.stack([
    [emotion_df[var].iloc[i : i+seq_length].mean()
     for i in range(input_offset, input_offset + num_windows*seq_length, seq_length)]
    for var in emotion_vars
]).T.astype(np.float32)

# ========= Load & preprocess fMRI with offset =========
def load_4d_fmri_from_pt(subject, base_dir, offset, n_frames, window, aff):
    frame_paths = sorted(
        glob.glob(os.path.join(base_dir, subject, 'frame_*.pt')),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
    )[offset : offset + n_frames]

    vol_list = []
    for fp in frame_paths:
        data = torch.load(fp).float()
        if data.ndim == 4:
            data = data.squeeze(-1)
        elif data.shape == (81,95,81):
            pad_val = data.flatten()[0].item()
            data = torch.nn.functional.pad(
                data.unsqueeze(0).unsqueeze(0),
                (7,8, 0,1, 7,8),
                value=pad_val
            ).squeeze()
        vol_list.append(data.numpy().astype(np.float32))

    vol_4d = np.stack(vol_list, axis=-1)
    seqs_4d = np.split(vol_4d, num_windows, axis=-1)
    return [nib.Nifti1Image(seq.mean(-1), aff) for seq in seqs_4d]

# ========= Prepare datasets =========
def prepare(names, split_name):
    X, y = [], []
    print(f'\n🚩 Starting {split_name} data preparation. Total subjects: {len(names)}')
    for idx, subj in enumerate(names, 1):
        print(f'[{split_name}] Processing subject {idx}/{len(names)}: {subj}')
        subj_X = load_4d_fmri_from_pt(subj, data_base_dir, input_offset, num_windows * seq_length, seq_length, affine)
        X.extend(subj_X)
        y.extend(target_vals)
        print(f'[{split_name}] Subject {subj} done. Total samples so far: {len(X)}')
    print(f'✅ Finished preparing {split_name} dataset. Total samples: {len(X)}')
    return X, np.asarray(y, dtype=np.float32)

X_train, y_train = prepare(train_names, 'Train')
X_val, y_val = prepare(val_names, 'Validation')
X_test, y_test = prepare(test_names, 'Test')

# ========= Train multi-output SVR =========
print('\n🚩 Starting model training...')
base_decoder = FREMRegressor(
    estimator=SVR(kernel='linear'),
    standardize='zscore_sample',
    screening_percentile=20,
    smoothing_fwhm=None,
    scoring='neg_mean_squared_error',
    mask=resampled_mask
)

decoder = MultiOutputRegressor(base_decoder, n_jobs=1)
decoder.fit(X_train, y_train)
print('✅ Model training completed.')

# ========= Evaluate & save performance =========
performance = []
for split, X, y in [('Train', X_train, y_train),
                    ('Validation', X_val, y_val),
                    ('Test', X_test, y_test)]:
    print(f'\n🚩 Evaluating {split} set...')
    pred = decoder.predict(X)
    mse = mean_squared_error(y, pred, multioutput='raw_values')
    r2 = r2_score(y, pred, multioutput='raw_values')
    for var, m, r in zip(emotion_vars, mse, r2):
        performance.append({'Split': split, 'Emotion': var, 'MSE': m, 'R2': r})
        print(f'[{split}] {var:<8} ➜ MSE: {m:.4f}, R²: {r:.4f}')
print('✅ Evaluation completed.')

performance_df = pd.DataFrame(performance)

model_save_dir = '/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/baseline_mvpa/models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(
    model_save_dir, f'multioutput_svr_{input_type}_offset{input_offset}_seq{seq_length}.pkl'
)
performance_csv_path = model_save_path.replace('.pkl', '_performance.csv')

print('\n🚩 Saving model and performance results...')
joblib.dump(decoder, model_save_path)
performance_df.to_csv(performance_csv_path, index=False)

print(f'\n✅ Model saved: {model_save_path}')
print(f'✅ Performance saved: {performance_csv_path}')

