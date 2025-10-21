"""
Train GLM baseline for emotion prediction

This script:
1. Reuses train/val/test splits from SwiFT-IO data_module
2. Loads ROI timeseries for each subject
3. Trains GLM baseline
4. Evaluates on val and test sets
5. Saves results for comparison with SwiFT-IO

Usage:
    python train_glm_baseline.py --downstream_task emotions --dataset_split_seed 777
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.data_module import fMRIDataModule
from baselines.glm_baseline import GLMBaseline
import json


def load_emotion_labels(task_name: str, input_type: str, adjust_hrf: bool = False) -> pd.DataFrame:
    """
    Load emotion labels from CSV files

    Args:
        task_name: 'emotions', 'contents', or 'features'
        input_type: 'movieDM' or 'movieTP'
        adjust_hrf: Whether to use HRF-adjusted labels

    Returns:
        DataFrame with emotion labels
    """
    emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    contents = ['Closeup', 'Body', 'Face', 'NumberCharacters', 'SpokenWords', 'WrittenWords']
    features = ['Brightness', 'SaliencyFraction', 'Sharpness', 'Vibrance', 'Loudness', 'Motion', 'Tempo', 'LowLevelChange']

    if task_name == 'emotions':
        label_names = emotions
    elif task_name == 'contents':
        label_names = contents
    elif task_name == 'features':
        label_names = features
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Adjust for HRF if needed
    if adjust_hrf:
        label_names = [x + "_conv" for x in label_names]

    # Load CSV
    if input_type == 'movieDM':
        csv_path = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/data_behavior/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv"
    elif input_type == 'movieTP':
        csv_path = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/data_behavior/ThePresent_summary_codes_1.2Hz_intuitivenames_260120.csv"
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    df = pd.read_csv(csv_path)

    # Select label columns and sort by frame
    df = df[label_names + ['frame']].dropna()
    df = df.sort_values('frame').reset_index(drop=True)

    # Return only label columns (frame is used for sorting only)
    return df[label_names]


def main():
    # Parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Data arguments (reuse from data_module)
    parser.add_argument("--image_path", type=str, required=True, help="Path to fMRI image data")
    parser.add_argument("--dataset_name", type=str, default="HBN", help="Dataset name")
    parser.add_argument("--downstream_task", type=str, default="emotions", help="Task: emotions, contents, features")
    parser.add_argument("--downstream_task_type", type=str, default="regression", help="Task type")
    parser.add_argument("--input_type", type=str, default="movieDM", choices=['movieDM', 'movieTP'], help="Movie type")
    parser.add_argument("--dataset_split_seed", type=int, default=777, help="Random seed for data split")
    parser.add_argument("--stratified_params", nargs="+", default=None, type=str, help="Stratified split parameters")
    parser.add_argument("--train_split", type=float, default=0.7, help="Training set proportion")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set proportion")
    parser.add_argument("--bad_subj_path", type=str, default=None, help="Path to bad subjects file")
    parser.add_argument("--adjust_hrf", action='store_true', help="Use HRF-adjusted emotion labels")

    # GLM-specific arguments
    parser.add_argument("--sequence_length", type=int, default=20, help="Length of fMRI sequence")
    parser.add_argument("--use_cross_validation", action='store_true', default=True, help="Use CV for alpha selection")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (if not using CV)")
    parser.add_argument("--standardize", action='store_true', default=True, help="Standardize features")
    parser.add_argument("--use_emotion_rois_only", action='store_true', default=True, help="Use only emotion ROIs")
    parser.add_argument("--roi_timeseries_dir", type=str, default="/scratch/HBN/9.2.movieDM_ROI_timeseries",
                       help="Path to ROI timeseries directory")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output/glm_baseline", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # Dummy arguments required by data_module
    parser.add_argument("--decoder", type=str, default="series_decoder", help="Decoder type (for data_module)")
    parser.add_argument("--num_targets", type=int, default=7, help="Number of targets (for data_module)")
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--with_voxel_norm", action='store_true', default=False)
    parser.add_argument("--shuffle_time_sequence", action='store_true', default=False)
    parser.add_argument("--label_scaling_method", type=str, default="standardization")
    parser.add_argument("--use_contrastive", action='store_true', default=False)
    parser.add_argument("--contrastive_type", type=int, default=0)
    parser.add_argument("--limit_training_samples", type=int, default=None)
    parser.add_argument("--input_offset", type=int, default=0)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("GLM Baseline Training")
    print("="*80)
    print(f"Task: {args.downstream_task}")
    print(f"Input type: {args.input_type}")
    print(f"Split seed: {args.dataset_split_seed}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Use emotion ROIs only: {args.use_emotion_rois_only}")
    print("="*80)

    # ===== 1. Get train/val/test splits =====
    print("\n[Step 1] Getting train/val/test splits...")

    # Get available subjects from ROI timeseries directory
    roi_files = os.listdir(args.roi_timeseries_dir)
    available_subjects = []
    for f in roi_files:
        if f.endswith('_roi_temporal_activity.csv'):
            # Extract subject ID: sub-NDARXXXXX_movieDM_roi_temporal_activity.csv -> sub-NDARXXXXX
            subj_id = f.replace(f'_{args.input_type}_roi_temporal_activity.csv', '')
            available_subjects.append(subj_id)

    print(f"Found {len(available_subjects)} subjects with ROI timeseries")

    # Create minimal subject_dict for split function (value doesn't matter for GLM)
    subject_dict = {subj: (0, 0) for subj in available_subjects}

    # Use data_module's stratified split function
    metadata_csv_path = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/data_behavior/split_fixed_1.w.Dx.csv"
    data_module = fMRIDataModule(**vars(args))

    train_names, val_names, test_names = data_module.determine_stratified_split(
        subject_dict,
        args.dataset_split_seed,
        args.stratified_params,
        metadata_csv_path,
        args.train_split,
        args.val_split
    )

    # Remove bad subjects if specified
    if args.bad_subj_path:
        bad_subjects = open(args.bad_subj_path, "r").readlines()
        bad_subjects = [s.strip() for s in bad_subjects]
        train_names = [s for s in train_names if s not in bad_subjects]
        val_names = [s for s in val_names if s not in bad_subjects]
        test_names = [s for s in test_names if s not in bad_subjects]

    # Create subject dicts (values don't matter for GLM, only keys are used)
    train_dict = {key: (0, 0) for key in train_names if key in subject_dict}
    val_dict = {key: (0, 0) for key in val_names if key in subject_dict}
    test_dict = {key: (0, 0) for key in test_names if key in subject_dict}

    print(f"Train subjects: {len(train_dict)}")
    print(f"Val subjects: {len(val_dict)}")
    print(f"Test subjects: {len(test_dict)}")

    # ===== 2. Load emotion labels =====
    print("\n[Step 2] Loading emotion labels...")

    emotion_labels = load_emotion_labels(
        args.downstream_task,
        args.input_type,
        args.adjust_hrf
    )

    print(f"Emotion labels shape: {emotion_labels.shape}")
    print(f"Emotion names: {list(emotion_labels.columns)}")

    # ===== 3. Initialize GLM baseline =====
    print("\n[Step 3] Initializing GLM baseline...")

    glm = GLMBaseline(
        roi_timeseries_dir=args.roi_timeseries_dir,
        num_emotions=emotion_labels.shape[1],
        sequence_length=args.sequence_length,
        use_cross_validation=args.use_cross_validation,
        alpha=args.alpha,
        standardize=args.standardize,
        use_emotion_rois_only=args.use_emotion_rois_only
    )

    # ===== 4. Train GLM =====
    print("\n[Step 4] Training GLM...")

    train_metrics = glm.fit(train_dict, emotion_labels)

    # ===== 5. Evaluate on validation set =====
    print("\n[Step 5] Evaluating on validation set...")

    val_metrics = glm.evaluate(val_dict, emotion_labels, mode='valid')

    # ===== 6. Evaluate on test set =====
    print("\n[Step 6] Evaluating on test set...")

    test_metrics = glm.evaluate(test_dict, emotion_labels, mode='test')

    # ===== 7. Save results =====
    print("\n[Step 7] Saving results...")

    # Combine all metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "glm_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Save model
    model_path = os.path.join(args.output_dir, "glm_model.pkl")
    glm.save(model_path)

    # Save configuration
    config_path = os.path.join(args.output_dir, "glm_config.json")
    config = {
        'task': args.downstream_task,
        'input_type': args.input_type,
        'split_seed': args.dataset_split_seed,
        'sequence_length': args.sequence_length,
        'use_emotion_rois_only': args.use_emotion_rois_only,
        'use_cross_validation': args.use_cross_validation,
        'alpha': args.alpha,
        'standardize': args.standardize,
        'num_train_subjects': len(train_dict),
        'num_val_subjects': len(val_dict),
        'num_test_subjects': len(test_dict),
        'selected_roi_names': glm.selected_roi_names
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # ===== 8. Print summary =====
    print("\n" + "="*80)
    print("GLM Baseline Training Complete!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Train MSE: {train_metrics['train_mse']:.4f}")
    print(f"  Train MAE: {train_metrics['train_mae']:.4f}")
    print(f"  Valid MSE: {val_metrics['valid_mse']:.4f}")
    print(f"  Valid MAE: {val_metrics['valid_mae']:.4f}")
    print(f"  Test MSE:  {test_metrics['test_mse']:.4f}")
    print(f"  Test MAE:  {test_metrics['test_mae']:.4f}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
