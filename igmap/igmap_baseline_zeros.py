#!/usr/bin/env python3
"""
IG Map with baseline = zeros (기존 방식)

Model: opr6oq97
Baseline: torch.zeros_like(input_ts)
"""

import argparse
import time
from pathlib import Path
import torch
import nibabel as nib
from multiprocessing import Pool
import sys
import os
from torch.utils.data import DataLoader, Subset

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

# Global variable for shared model
model = None
data_module = None

def init_model_and_data(ckpt_path_str, args_model_dict, project_root):
    global model, data_module

    sys.path.append(str(project_root / "src"))
    from module.pl_classifier import LitClassifier
    from module.utils.data_module import fMRIDataModule

    ckpt = torch.load(ckpt_path_str, map_location="cpu")
    args_model_dict["num_workers"] = 0
    args_model_dict["eval_num_workers"] = 0
    args_model_dict['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args_model_dict['default_root_dir'] = str(project_root / "output/moviefmri")
    args_model_dict['shuffle_time_sequence'] = False
    args_model_dict['time_as_channel'] = False
    args_model_dict['eval_batch_size'] = 1
    args_model_dict['bad_subj_path'] = None
    args_model_dict['limit_training_samples'] = 0
    args_model_dict['downstream_task'] = 'emotions'
    args_model_dict['decoder'] = 'series_decoder'

    # Add missing data module parameters
    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 777)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = None
    if 'train_split' not in args_model_dict:
        args_model_dict['train_split'] = 0.7
    if 'val_split' not in args_model_dict:
        args_model_dict['val_split'] = 0.15

    print("🚀 Initializing model & data...", flush=True)
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

    model = LitClassifier(data_module=data_module, **args_model_dict)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cpu()
    print("✅ Model & Data initialized.")

def compute_ig_on_prediction_average(input_ts, baseline, i, n_steps=10):
    global model
    alphas = torch.linspace(0, 1.0, steps=n_steps).view(-1, 1, 1, 1, 1, 1).to(input_ts.device)
    delta = input_ts - baseline
    scaled_inputs = baseline + alphas * delta
    grads = []
    for step_idx, s_input in enumerate(scaled_inputs):
        s_input = s_input.unsqueeze(0).requires_grad_(True)
        output = model(s_input)

        # Debug: print output shape on first step
        if step_idx == 0:
            print(f"  [DEBUG] Model output shape: {output.shape}", flush=True)

        print(f"📈 IG step {step_idx+1}/{n_steps}, emotion: {emotion_labels[i]}", flush=True)

        # Handle different output shapes
        if output.dim() == 2:
            # Output shape: [batch, emotions]
            scalar = output[:, i].mean()
        elif output.dim() == 3:
            # Output shape: [batch, time, emotions]
            scalar = output[:, :, i].mean()
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        grad = torch.autograd.grad(outputs=scalar, inputs=s_input)[0]
        grads.append(grad)
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grads = delta * avg_grads
    return integrated_grads.detach()

def process_subject(args_tuple):
    subject, args, affine_path, project_root = args_tuple
    global model, data_module

    print(f"\n🚀 Start subject: {subject} | PID: {os.getpid()}", flush=True)
    overall_start = time.time()

    affine = nib.load(str(affine_path)).affine

    testset = model.data_module.test_dataset
    # Use dataset.data to find subject indices without loading actual fMRI data
    # data format: (i, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex)
    subj_indices = [
        idx for idx, data_tuple in enumerate(testset.data)
        if str(data_tuple[1]) == subject
    ]
    if not subj_indices:
        print(f"❌ No matching data for subject: {subject}", flush=True)
        return
    print(f"✅ Found {len(subj_indices)} sequences for {subject}", flush=True)

    test_loader = DataLoader(Subset(testset, subj_indices), batch_size=1, shuffle=False, num_workers=0)
    instance_count = 0

    for data in test_loader:
        subj = data['subject_name'] if isinstance(data['subject_name'], str) else data['subject_name'][0]
        TR_index = int(data['TR'])
        input_ts = data['fmri_sequence'].float().cpu()

        # ========== BASELINE: ZEROS (기존 방식) ==========
        baseline = torch.zeros_like(input_ts)
        print(f"  [Baseline] zeros, shape: {baseline.shape}", flush=True)
        # ===============================================

        for i, emotion_label in enumerate(emotion_labels):
            out_dir = project_root / f"analysis/4_IGmap/baseline_zeros/{args.run_id}/nii_segments" / subject / f"target{i}_{emotion_label}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 양수와 음수를 분리하여 각 영역 평균
            result = compute_ig_on_prediction_average(input_ts, baseline, i=i, n_steps=20)
            # result shape: [B, C, X, Y, Z, T]
            result_tensor = result[0, 0, :, :, :, :]  # [X, Y, Z, T]

            pos_mask = (result_tensor > 0).float()
            neg_mask = (result_tensor < 0).float()
            pos_sum = (result_tensor * pos_mask).sum(dim=-1)
            pos_count = pos_mask.sum(dim=-1) + 1e-8
            avgpred_positive = pos_sum / pos_count

            neg_sum = (result_tensor * neg_mask).sum(dim=-1)
            neg_count = neg_mask.sum(dim=-1) + 1e-8
            avgpred_negative = neg_sum / neg_count

            out_path_pos = out_dir / f"{subject}_{emotion_label}_TR{TR_index:03d}_AVGpred_positive.nii.gz"
            out_path_neg = out_dir / f"{subject}_{emotion_label}_TR{TR_index:03d}_AVGpred_negative.nii.gz"
            nib.save(nib.Nifti1Image(avgpred_positive.cpu().numpy(), affine), out_path_pos)
            nib.save(nib.Nifti1Image(avgpred_negative.cpu().numpy(), affine), out_path_neg)
            print(f"[IG OK] {subject} - {emotion_label} TR{TR_index:03d} (baseline=zeros)", flush=True)

        instance_count += 1
        # Process all sequences (not just first 5)

    print(f"✅ Total time for {subject}: {time.time() - overall_start:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='opr6oq97')
    parser.add_argument('--subject', type=str, default=None, help="Subject name to process (optional, will auto-select from test set if not provided)")
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--project_root', type=str, default='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO')
    parser.add_argument('--dry-run', action='store_true', help="Only list test subjects without processing")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{args.run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*"))[0]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    print(f"✅ Loaded checkpoint: {ckpt_path.name}")
    print(f"   seq_len={args_model_dict.get('sequence_length')}, offset={args_model_dict.get('input_offset')}")

    # Initialize model and data to get test set subjects
    print("🚀 Initializing model & data to find test subjects...")
    init_model_and_data(str(ckpt_path), args_model_dict, project_root)

    # Get unique test subjects from test dataset (faster method using dataset's data list)
    testset = data_module.test_dataset
    if hasattr(testset, 'data'):
        # Extract unique subjects from data list
        # data format: (i, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex)
        test_subjects = sorted(list(set([str(x[1]) for x in testset.data])))
        print(f"✅ Extracted {len(test_subjects)} subjects from dataset.data")
    else:
        print("❌ Error: dataset.data attribute not found")
        sys.exit(1)

    print(f"\n📊 Found {len(test_subjects)} unique subjects in test set")
    print(f"   First 10 subjects: {test_subjects[:10]}")

    # If dry-run, just print subjects and exit
    if args.dry_run:
        print("\n" + "="*50)
        print("DRY RUN MODE - Test Subjects Only")
        print("="*50)
        for i, subj in enumerate(test_subjects, 1):
            print(f"  {i:3d}. {subj}")
        print("="*50)
        print(f"\nTotal: {len(test_subjects)} subjects in test set")
        sys.exit(0)

    # Select subject
    if args.subject is None:
        selected_subject = test_subjects[0]
        print(f"\n✅ Auto-selected first test subject: {selected_subject}")
    else:
        if args.subject in test_subjects:
            selected_subject = args.subject
            print(f"\n✅ Using specified subject: {selected_subject}")
        else:
            print(f"\n❌ Error: Subject '{args.subject}' not found in test set!")
            print(f"   Available subjects: {test_subjects[:20]}...")
            sys.exit(1)

    # Use reference affine
    affine_path = project_root / "analysis/4_IGmap/reference_affine_MNI152_2mm.nii.gz"
    if not affine_path.exists():
        print("⚠ Reference affine not found, creating from data...")
        affine_path = project_root / "igmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    # Process the selected subject
    process_subject((selected_subject, args, affine_path, project_root))

    print("✅ All processing complete!")
