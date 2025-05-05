#!/usr/bin/env python
# svm_test.py
"""
Baseline MVPA: SVM regression on window-averaged movieDM fMRI.
Subjects with fewer than `total_length` frames are automatically excluded.
"""

# ========= Imports =========
import os, glob
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from nilearn.decoding import FREMRegressor
from nilearn.image import resample_to_img

# ========= Parameters =========
input_type   = 'movieDM'   # 'movieDM' or 'movieTP'
seq_length   = 50          # frames per window
total_length = 750 if input_type == 'movieDM' else 360
assert total_length is not None, "total_length must be defined."

# ========= File paths =========
split_file_path        = '/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/data/splits/HBN/split_fixed_1.txt'
emotion_timeseries_csv = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/metadata/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv'
data_base_dir          = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img'

affine_img_path  = Path('/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed.nii.gz')
brain_mask_path  = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

# ========= Load affine & mask =========
affine_img     = nib.load(affine_img_path)
affine         = affine_img.affine
brain_mask_img = nib.load(brain_mask_path)
resampled_mask = resample_to_img(brain_mask_img, affine_img, interpolation='nearest')

# ========= Subject split =========
with open(split_file_path) as f:
    lines = f.read().splitlines()

train_idx = lines.index('train_subjects')
val_idx   = lines.index('val_subjects')
test_idx  = lines.index('test_subjects')

train_names = lines[train_idx+1 : val_idx]
val_names   = lines[val_idx+1   : test_idx]
test_names  = lines[test_idx+1  :]

print(f'Initial counts ➜ train {len(train_names)}, val {len(val_names)}, test {len(test_names)}')

# ========= Filter subjects by frame count =========
def filter_subjects(names, base_dir, min_frames):
    """Keep only subjects with ≥min_frames *.pt files."""
    kept = []
    for subj in names:
        n_frames = len(glob.glob(os.path.join(base_dir, subj, 'frame_*.pt')))
        if n_frames >= min_frames:
            kept.append(subj)
        else:
            print(f'⏩  {subj} skipped ({n_frames}/{min_frames} frames)')
    return kept

train_names = filter_subjects(train_names, data_base_dir, total_length)
val_names   = filter_subjects(val_names,   data_base_dir, total_length)
test_names  = filter_subjects(test_names,  data_base_dir, total_length)

print(f'After filtering ➜ train {len(train_names)}, val {len(val_names)}, test {len(test_names)}')

# ========= Emotion target (window-mean) =========
emotion_df   = pd.read_csv(emotion_timeseries_csv)[['Positive','Negative','Anger','Happy','Fear','Sad','Excited']]
target_var   = 'Positive'
num_windows  = total_length // seq_length
target_vals  = [
    emotion_df[target_var].iloc[i : i + seq_length].mean()
    for i in range(0, num_windows * seq_length, seq_length)
]
target_vals = np.asarray(target_vals, dtype=np.float32)

# ========= Helper: load a subject & convert to 4-D NIfTI sequences =========
def load_4d_fmri_from_pt(subject, base_dir, min_frames, window, aff):
    frame_paths = sorted(
        glob.glob(os.path.join(base_dir, subject, 'frame_*.pt')),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
    )[:min_frames]

    vol_list = []
    for fp in frame_paths:
        data = torch.load(fp).float()
        if data.ndim == 4:            # (96,96,96,1) → (96,96,96)
            data = data.squeeze(-1)
        elif data.shape == (81,95,81):  # pad to (96,96,96)
            pad_val = data.flatten()[0].item()
            data = torch.nn.functional.pad(
                data.unsqueeze(0).unsqueeze(0),
                (7,8, 0,1, 7,8),
                value=pad_val
            ).squeeze()
        vol_list.append(data.numpy().astype(np.float32))

    vol_4d  = np.stack(vol_list, axis=-1)               # (96,96,96,T)
    seqs_4d = np.split(vol_4d, num_windows, axis=-1)    # list of (96,96,96,window)
    # average within each window then wrap as NIfTI
    return [nib.Nifti1Image(seq.mean(-1), aff) for seq in seqs_4d]

# ========= Prepare datasets =========
def prepare(names):
    X, y = [], []
    for subj in names:
        X.extend(load_4d_fmri_from_pt(subj, data_base_dir, total_length, seq_length, affine))
        y.extend(target_vals)
    return X, np.asarray(y, dtype=np.float32)

X_train, y_train = prepare(train_names)
X_val,   y_val   = prepare(val_names)
X_test,  y_test  = prepare(test_names)

# ========= SVM regression (FREM wrapper) =========
decoder = FREMRegressor(
    estimator             = SVR(kernel='linear'),
    standardize           = 'zscore_sample',
    screening_percentile  = 20,
    smoothing_fwhm        = None,
    scoring               = 'neg_mean_squared_error',
    mask                  = resampled_mask
)

decoder.fit(X_train, y_train)

# ========= Evaluation =========
for split, X, y in [('Train', X_train, y_train),
                    ('Val',   X_val,   y_val),
                    ('Test',  X_test,  y_test)]:
    preds = decoder.predict(X)
    mse   = mean_squared_error(y, preds)
    r2    = r2_score(y, preds)
    print(f'{split:5s}  MSE {mse:8.4f}   R² {r2:6.4f}')
