#!/usr/bin/env python3
"""
IG Map with selective sequences based on emotion peaks.
Processes only specified subjects and their peak emotion sequences.

Model: opr6oq97
Baseline: torch.zeros_like(input_ts)
"""

import argparse
import time
from pathlib import Path
import torch
import nibabel as nib
import sys
import os
import json
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
    print("✅ Model & Data initialized.", flush=True)

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
            # Output shape: [time, emotions]
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
    subject, selected_sequences, args, affine_path, project_root = args_tuple
    global model, data_module

    print(f"\n{'='*70}", flush=True)
    print(f"🚀 Start subject: {subject} | PID: {os.getpid()}", flush=True)
    print(f"{'='*70}", flush=True)
    overall_start = time.time()

    affine = nib.load(str(affine_path)).affine

    testset = model.data_module.test_dataset

    # Get all sequence indices for this subject
    # data format: (i, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex)
    subject_data_indices = [
        idx for idx, data_tuple in enumerate(testset.data)
        if str(data_tuple[1]) == subject
    ]

    if not subject_data_indices:
        print(f"❌ No matching data for subject: {subject}", flush=True)
        return

    print(f"✅ Total sequences in dataset: {len(subject_data_indices)}", flush=True)

    # Process selected sequences for each emotion
    total_sequences_processed = 0

    for emotion_name in selected_sequences.keys():
        emotion_idx = emotion_labels.index(emotion_name)
        peak_sequences = selected_sequences[emotion_name]

        print(f"\n{'-'*70}", flush=True)
        print(f"Processing {len(peak_sequences)} sequences for emotion: {emotion_name}", flush=True)
        print(f"{'-'*70}", flush=True)

        for rank, seq_info in enumerate(peak_sequences, 1):
            seq_idx = seq_info['sequence_idx']

            # Get the actual data index
            if seq_idx >= len(subject_data_indices):
                print(f"⚠️ Sequence index {seq_idx} out of range, skipping", flush=True)
                continue

            data_idx = subject_data_indices[seq_idx]

            # Load this specific sequence
            test_loader = DataLoader(Subset(testset, [data_idx]),
                                    batch_size=1, shuffle=False, num_workers=0)

            for data in test_loader:
                subj = data['subject_name'] if isinstance(data['subject_name'], str) else data['subject_name'][0]
                TR_index = int(data['TR'])
                input_ts = data['fmri_sequence'].float().cpu()

                print(f"\n  Rank {rank}: TR{seq_info['start_frame']:03d}-{seq_info['end_frame']:03d} "
                      f"(avg: {seq_info['avg_score']:.3f})", flush=True)

                # ========== BASELINE: ZEROS ==========
                baseline = torch.zeros_like(input_ts)
                print(f"  [Baseline] zeros, shape: {baseline.shape}", flush=True)
                # =====================================

                out_dir = project_root / f"analysis/4_IGmap/baseline_zeros_selective/{args.run_id}/nii_segments" / subject / f"target{emotion_idx}_{emotion_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Compute IG
                result = compute_ig_on_prediction_average(input_ts, baseline, i=emotion_idx, n_steps=args.n_steps)
                # result shape: [B, C, X, Y, Z, T]
                result_tensor = result[0, 0, :, :, :, :]  # [X, Y, Z, T]

                # Average over time: positive and negative
                pos_mask = (result_tensor > 0).float()
                neg_mask = (result_tensor < 0).float()
                pos_sum = (result_tensor * pos_mask).sum(dim=-1)
                pos_count = pos_mask.sum(dim=-1) + 1e-8
                avgpred_positive = pos_sum / pos_count

                neg_sum = (result_tensor * neg_mask).sum(dim=-1)
                neg_count = neg_mask.sum(dim=-1) + 1e-8
                avgpred_negative = neg_sum / neg_count

                # Save with rank information
                out_path_pos = out_dir / f"{subject}_{emotion_name}_TR{TR_index:03d}_rank{rank:02d}_AVGpred_positive.nii.gz"
                out_path_neg = out_dir / f"{subject}_{emotion_name}_TR{TR_index:03d}_rank{rank:02d}_AVGpred_negative.nii.gz"
                nib.save(nib.Nifti1Image(avgpred_positive.cpu().numpy(), affine), out_path_pos)
                nib.save(nib.Nifti1Image(avgpred_negative.cpu().numpy(), affine), out_path_neg)
                print(f"  ✅ [IG OK] {subject} - {emotion_name} TR{TR_index:03d} rank{rank} (baseline=zeros)", flush=True)

                total_sequences_processed += 1

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"✅ Subject {subject} complete!", flush=True)
    print(f"   Total sequences processed: {total_sequences_processed}", flush=True)
    print(f"   Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='opr6oq97')
    parser.add_argument('--subjects', type=str, nargs='+', required=True,
                       help='List of subjects to process')
    parser.add_argument('--emotion_peaks_json', type=str,
                       default='analysis/4_IGmap/emotion_peak_sequences/opr6oq97_emotion_peak_sequences_top10.json',
                       help='Path to emotion peak sequences JSON file')
    parser.add_argument('--top_k_seqs', type=int, default=5,
                       help='Number of top sequences to process per emotion (default: 5)')
    parser.add_argument('--emotions', type=str, nargs='+', default=None,
                       help='Specific emotions to process (default: all 7 emotions)')
    parser.add_argument('--n_steps', type=int, default=20,
                       help='Number of IG steps (default: 20)')
    parser.add_argument('--project_root', type=str, default='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO')
    args = parser.parse_args()

    project_root = Path(args.project_root)

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{args.run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*"))[0]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    print(f"{'='*70}")
    print(f"IG Map - Selective Sequences (Baseline: Zeros)")
    print(f"{'='*70}")
    print(f"✅ Loaded checkpoint: {ckpt_path.name}")
    print(f"   seq_len={args_model_dict.get('sequence_length')}, offset={args_model_dict.get('input_offset')}")
    print(f"   IG steps: {args.n_steps}")
    print(f"   Top-K sequences per emotion: {args.top_k_seqs}")
    print(f"   Subjects to process: {', '.join(args.subjects)}")

    # Load emotion peak sequences
    peaks_json_path = project_root / args.emotion_peaks_json
    print(f"\n📊 Loading emotion peak sequences from:")
    print(f"   {peaks_json_path}")

    with open(peaks_json_path, 'r') as f:
        peaks_data = json.load(f)

    # Determine which emotions to process
    emotions_to_process = args.emotions if args.emotions else emotion_labels
    print(f"\n🎯 Emotions to process: {', '.join(emotions_to_process)}")

    # Initialize model once
    init_model_and_data(str(ckpt_path), args_model_dict, project_root)

    # Use reference affine
    affine_path = project_root / "analysis/4_IGmap/reference_affine_MNI152_2mm.nii.gz"
    if not affine_path.exists():
        print("⚠ Reference affine not found, using fallback...")
        affine_path = project_root / "igmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    # Process each subject
    for subject in args.subjects:
        if subject not in peaks_data['subjects']:
            print(f"\n❌ No peak data for subject: {subject}")
            continue

        # Get selected sequences for this subject
        subject_peaks = peaks_data['subjects'][subject]

        # Filter by emotions and top-k
        selected_sequences = {}
        for emotion in emotions_to_process:
            if emotion in subject_peaks:
                selected_sequences[emotion] = subject_peaks[emotion][:args.top_k_seqs]

        process_subject((subject, selected_sequences, args, affine_path, project_root))

    print(f"\n{'='*70}")
    print("✅ All processing complete!")
    print(f"{'='*70}")
