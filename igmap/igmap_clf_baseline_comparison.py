#!/usr/bin/env python3
"""
IG Map for CLASSIFICATION task with baseline comparison
Computes gradients with respect to positive class (class 1) logits/probabilities
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
    args_model_dict['downstream_task_type'] = 'classification'
    args_model_dict['decoder'] = 'series_decoder'

    # Add missing data module parameters
    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 2)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = ['Age', 'Sex']
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
    """
    Compute Integrated Gradients for classification task.
    We compute gradients w.r.t. positive class (class 1) logits for emotion i.

    Args:
        input_ts: Input fMRI sequence [1, C, X, Y, Z, T]
        baseline: Baseline fMRI sequence [1, C, X, Y, Z, T]
        i: Emotion index (0-6)
        n_steps: Number of IG steps

    Returns:
        Integrated gradients [1, C, X, Y, Z, T]
    """
    global model
    alphas = torch.linspace(0, 1.0, steps=n_steps).view(-1, 1, 1, 1, 1, 1).to(input_ts.device)
    delta = input_ts - baseline
    scaled_inputs = baseline + alphas * delta
    grads = []

    for step_idx, s_input in enumerate(scaled_inputs):
        s_input = s_input.unsqueeze(0).requires_grad_(True)
        output = model(s_input)  # [1, time, 7, 2] - logits for binary classification

        # Debug: print output shape on first step
        if step_idx == 0:
            print(f"  [DEBUG] Model output shape: {output.shape}", flush=True)

        print(f"📈 IG step {step_idx+1}/{n_steps}, emotion: {emotion_labels[i]}", flush=True)

        # For classification, output is [batch, time, 7, 2]
        # We want gradient w.r.t. positive class (class 1) logits
        if output.dim() == 4:
            # Output shape: [batch, time, emotions, 2]
            # Select positive class logits for emotion i and average over time
            scalar = output[:, :, i, 1].mean()  # Average positive class logits over time
        else:
            raise ValueError(f"Unexpected output shape for classification: {output.shape}. "
                           f"Expected [batch, time, 7, 2]")

        grad = torch.autograd.grad(outputs=scalar, inputs=s_input)[0]
        grads.append(grad)

    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grads = delta * avg_grads
    return integrated_grads.detach()

def process_subject(args_tuple):
    subject, selected_sequences, args, affine_path, project_root = args_tuple
    global model, data_module

    print(f"\n{'='*70}", flush=True)
    print(f"🚀 Start subject: {subject} | Baseline: {args.baseline} | PID: {os.getpid()}", flush=True)
    print(f"{'='*70}", flush=True)
    overall_start = time.time()

    affine = nib.load(str(affine_path)).affine

    testset = model.data_module.test_dataset

    # Get all sequence indices for this subject
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
                input_ts = data['fmri_sequence'].float().cpu()  # [B, C, X, Y, Z, T]

                print(f"\n🔍 Rank {rank}/{len(peak_sequences)} | Seq {seq_idx} | "
                      f"TR {TR_index:03d}-{TR_index + input_ts.shape[-1] - 1:03d} | "
                      f"Avg Prob: {seq_info.get('avg_prob', 0):.4f}", flush=True)

                # Create baseline
                if args.baseline == 'zeros':
                    baseline = torch.zeros_like(input_ts)
                elif args.baseline == 'first_10sec':
                    baseline = input_ts.clone()
                    baseline[:, :, :, :, :, 5:] = 0
                else:
                    raise ValueError(f"Unknown baseline: {args.baseline}")

                # Compute IG
                seq_start = time.time()
                ig = compute_ig_on_prediction_average(
                    input_ts,
                    baseline,
                    emotion_idx,
                    n_steps=args.n_steps
                )
                seq_duration = time.time() - seq_start

                # Sum over time dimension and channel
                ig_spatial = ig.squeeze(0).sum(dim=(0, -1)).numpy()  # [X, Y, Z]

                # Save as NIfTI
                output_dir = project_root / f"analysis/4_IGmap/clf_results/{args.run_id}/{args.baseline}/{emotion_name}"
                output_dir.mkdir(parents=True, exist_ok=True)

                nii_path = output_dir / f"{subject}_rank{rank}_seq{seq_idx}_TR{TR_index:03d}.nii.gz"
                nii = nib.Nifti1Image(ig_spatial, affine)
                nib.save(nii, str(nii_path))

                print(f"✅ Saved: {nii_path.name} ({seq_duration:.1f}s)", flush=True)
                total_sequences_processed += 1

    overall_duration = time.time() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"✅ Subject {subject} complete!", flush=True)
    print(f"   Total sequences: {total_sequences_processed}", flush=True)
    print(f"   Total time: {overall_duration:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='Model run ID (classification)')
    parser.add_argument('--subjects', type=str, nargs='+', required=True, help='Subject names')
    parser.add_argument('--baseline', type=str, required=True, choices=['zeros', 'first_10sec'])
    parser.add_argument('--emotions', type=str, nargs='+', default=None,
                       help='Specific emotions to process (default: all)')
    parser.add_argument('--top_k_seqs', type=int, default=5,
                       help='Number of top sequences to process per emotion')
    parser.add_argument('--n_steps', type=int, default=20, help='IG steps')
    parser.add_argument('--project_root', type=str,
                       default='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO')
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sys.path.append(str(project_root / "src"))

    print("="*70)
    print(f"IG Map Generation - CLASSIFICATION")
    print(f"Run ID: {args.run_id}")
    print(f"Baseline: {args.baseline}")
    print(f"Subjects: {args.subjects}")
    print(f"Emotions: {args.emotions if args.emotions else 'all'}")
    print(f"Top K seqs per emotion: {args.top_k_seqs}")
    print(f"IG steps: {args.n_steps}")
    print("="*70)

    # Load checkpoint
    ckpt_path = project_root / f"output/moviefmri/{args.run_id}/checkpt-epoch=08-valid_acc=1.00.ckpt"
    if not ckpt_path.exists():
        ckpt_files = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found for {args.run_id}")
        ckpt_path = ckpt_files[0]

    print(f"\n✅ Found checkpoint: {ckpt_path.name}")

    # Load peak sequences
    peak_seq_file = project_root / f"analysis/4_IGmap/clf_peak_sequences/{args.run_id}_clf_peak_sequences_top10.json"
    if not peak_seq_file.exists():
        raise FileNotFoundError(f"Peak sequences file not found: {peak_seq_file}")

    with open(peak_seq_file) as f:
        peak_data = json.load(f)

    print(f"✅ Loaded peak sequences from: {peak_seq_file.name}")

    # Initialize model
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']
    init_model_and_data(str(ckpt_path), args_model_dict, project_root)

    # Get affine from a sample subject
    testset = data_module.test_dataset
    sample_path = Path(str(testset.data[0][2]))
    affine_path = sample_path

    # Process each subject
    emotions_to_process = args.emotions if args.emotions else emotion_labels

    for subject in args.subjects:
        if subject not in peak_data['subjects']:
            print(f"⚠️ Subject {subject} not in peak sequences, skipping")
            continue

        # Filter sequences by emotion and top_k
        selected_sequences = {}
        for emotion in emotions_to_process:
            if emotion in peak_data['subjects'][subject]:
                selected_sequences[emotion] = peak_data['subjects'][subject][emotion][:args.top_k_seqs]

        if not selected_sequences:
            print(f"⚠️ No sequences for {subject}, skipping")
            continue

        # Process this subject
        process_subject((subject, selected_sequences, args, affine_path, project_root))

    print("\n" + "="*70)
    print("ALL SUBJECTS COMPLETED!")
    print("="*70)

if __name__ == '__main__':
    main()
