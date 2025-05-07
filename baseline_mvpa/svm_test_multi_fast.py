#!/usr/bin/env python
# svm_test_multi_fast.py
"""
Multi-output SVR 회귀 (7 emotions) — FREM + SVR(linear) 안전 버전.
"""

# ========= Imports =========
import os, glob, argparse
import numpy as np
import pandas as pd
import torch, nibabel as nib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nilearn.decoding import FREMRegressor
from nilearn.image import resample_to_img
import joblib

# ========= CLI =========
parser = argparse.ArgumentParser()
parser.add_argument('--input_type',  choices=['movieDM','movieTP'], default='movieDM')
parser.add_argument('--seq_length',  type=int, default=50)
parser.add_argument('--input_offset',type=int, default=3)
args = parser.parse_args()

input_type, seq_len, offset = args.input_type, args.seq_length, args.input_offset
total_len = 750 if input_type=='movieDM' else 360

# ========= Paths =========
split_file  = '/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/data/splits/HBN/split_fixed_1.txt'
emotion_csv = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/metadata/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv'
data_dir    = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img'

affine_nii  = Path('/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed.nii.gz')
mask_nii    = '/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

# ========= Load affine & mask =========
aff_img = nib.load(affine_nii); affine = aff_img.affine
mask_img = resample_to_img(nib.load(mask_nii), aff_img, interpolation='nearest')

# ========= Subject split =========
with open(split_file) as f: lines = f.read().splitlines()
tr_i,va_i,te_i = map(lines.index,['train_subjects','val_subjects','test_subjects'])
train,val,test = lines[tr_i+1:va_i], lines[va_i+1:te_i], lines[te_i+1:]

# ========= Frame 요구량 & 필터 =========
usable  = total_len - offset
num_win = usable // seq_len
need_fr = offset + num_win*seq_len

def filter_subjs(names):
    keep=[]
    for s in names:
        n=len(glob.glob(os.path.join(data_dir,s,'frame_*.pt')))
        print(f'⏩ {s}({n}/{need_fr}) skip' if n<need_fr else f'✅ {s} OK', flush=True)
        if n>=need_fr: keep.append(s)
    return keep

train,val,test = map(filter_subjs,[train,val,test])
print(f'After filter  Train {len(train)}  Val {len(val)}  Test {len(test)}', flush=True)

# ========= Emotion targets =========
vars = ['Positive','Negative','Anger','Happy','Fear','Sad','Excited']
emo_df = pd.read_csv(emotion_csv)[vars]
tar = np.stack([[emo_df[v].iloc[i:i+seq_len].mean()
                 for i in range(offset, offset+num_win*seq_len, seq_len)]
                for v in vars]).T.astype(np.float32)     # (num_win,7)

# ========= fMRI loader =========
def load_subj(s):
    fps = sorted(glob.glob(os.path.join(data_dir,s,'frame_*.pt')),
                 key=lambda p:int(os.path.basename(p).split('_')[1].split('.')[0])
                )[offset:offset+num_win*seq_len]
    vols=[]
    for fp in fps:
        d=torch.load(fp).float()
        if d.ndim==4: d=d.squeeze(-1)
        elif d.shape==(81,95,81):
            pad=d.flatten()[0].item()
            d=torch.nn.functional.pad(d.unsqueeze(0).unsqueeze(0),
                                      (7,8,0,1,7,8),value=pad).squeeze()
        vols.append(d.numpy().astype(np.float32))
    v4d=np.stack(vols,-1)
    return [nib.Nifti1Image(v4d[...,i*seq_len:(i+1)*seq_len].mean(-1),affine)
            for i in range(num_win)]

def prep(names):
    X,Y=[],[]
    for s in names: X.extend(load_subj(s)); Y.extend(tar)
    return X,np.asarray(Y,np.float32)

X_tr,y_tr=prep(train); X_va,y_va=prep(val); X_te,y_te=prep(test)

# ========= FREM + SVR(linear) =========
base = FREMRegressor(
        estimator            = SVR(kernel='linear'),
        standardize          = 'zscore_sample',
        screening_percentile = 5,
        cv                   = 5,
        mask                 = mask_img,
        n_jobs               = 16
)
model = MultiOutputRegressor(base, n_jobs=1)

print('🚩 Fit model...', flush=True)
model.fit(X_tr,y_tr)
print('✅ Fit done.', flush=True)

# ========= Evaluation =========
def report(tag,X,y):
    p=model.predict(X)
    mse=mean_squared_error(y,p,multioutput='raw_values')
    r2 =r2_score(y,p,multioutput='raw_values')
    for v,m,r in zip(vars,mse,r2):
        print(f'[{tag}] {v:<8}  MSE {m:7.4f}  R² {r:6.4f}', flush=True)
for t,X,y in [('Train',X_tr,y_tr),('Val',X_va,y_va),('Test',X_te,y_te)]: report(t,X,y)

# ========= Save =========
out_dir='/scratch/connectome/kimbo/SwiFT-IO-3/SwiFT-IO/baseline_mvpa/models'
os.makedirs(out_dir,exist_ok=True)
stem=f'svr_FREM_scr5_cv3_{input_type}_off{offset}_seq{seq_len}'
joblib.dump(model, f'{out_dir}/{stem}.pkl')
