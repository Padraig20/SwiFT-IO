#!/usr/bin/env python3
"""
Evaluate LSTM Baseline on Test Set

Loads trained LSTM checkpoint and evaluates on test set.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

# Add src to path (same as other evaluation scripts)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.module.utils.data_module import fMRIDataModule
from src.module.pl_classifier import LitClassifier


def custom_collate_fn(batch):
    """Custom collate function that handles string fields"""
    import torch
    import numpy as np
    from torch.utils.data._utils.collate import default_collate

    # Separate string fields from tensors
    batch_dict = {}
    for key in batch[0].keys():
        if key in ['subject_name', 'sex']:
            # Keep as list for string fields
            batch_dict[key] = [item[key] for item in batch]
        else:
            try:
                # Use default collate for tensors/arrays
                batch_dict[key] = default_collate([item[key] for item in batch])
            except TypeError as e:
                # Handle numpy object dtype arrays
                items = [item[key] for item in batch]
                if isinstance(items[0], np.ndarray) and items[0].dtype == np.object_:
                    print(f"Warning: Skipping batch with object dtype in field '{key}'")
                    raise
                raise

    return batch_dict


def main():
    parser = ArgumentParser()

    # Checkpoint path
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to LSTM checkpoint')

    # Data arguments (same as training)
    parser.add_argument("--image_path", type=str,
                       default='/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120',
                       help="Path to fMRI data")
    parser.add_argument("--dataset_name", type=str, default="HBN")
    parser.add_argument("--downstream_task", type=str, default="emotions")
    parser.add_argument("--downstream_task_type", type=str, default="regression")
    parser.add_argument("--input_type", type=str, default="movieDM")
    parser.add_argument("--dataset_split_seed", type=int, default=777)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    # Dummy arguments for data module compatibility
    parser.add_argument("--decoder", type=str, default="lstm_series_regression_head")
    parser.add_argument("--num_targets", type=int, default=7)
    parser.add_argument("--with_voxel_norm", action='store_true', default=False)
    parser.add_argument("--shuffle_time_sequence", action='store_true', default=False)
    parser.add_argument("--label_scaling_method", type=str, default="standardization")
    parser.add_argument("--use_contrastive", action='store_true', default=False)
    parser.add_argument("--contrastive_type", type=int, default=0)
    parser.add_argument("--limit_training_samples", type=int, default=None)
    parser.add_argument("--input_offset", type=int, default=0)
    parser.add_argument("--img_size", nargs="+", default=[96, 96, 96, 30], type=int)
    parser.add_argument("--stride_between_seq", type=int, default=1)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--stratified_params", nargs="+", default=None)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--bad_subj_path", type=str, default=None)
    parser.add_argument("--adjust_hrf", action='store_true', default=False)
    parser.add_argument("--augment_during_training", action='store_true', default=False)

    args = parser.parse_args()

    print("="*80)
    print("LSTM Baseline - Test Set Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split seed: {args.dataset_split_seed}")
    print()

    # Setup data module
    print("[Step 1] Setting up data module...")
    data_module = fMRIDataModule(
        image_path=args.image_path,
        dataset_name=args.dataset_name,
        dataset_split_num=1,
        dataset_split_seed=args.dataset_split_seed,
        downstream_task=args.downstream_task,
        downstream_task_type=args.downstream_task_type,
        input_type=args.input_type,
        sequence_length=args.sequence_length,
        stride_between_seq=args.stride_between_seq,
        stride_within_seq=args.stride_within_seq,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_contrastive=args.use_contrastive,
        contrastive_type=args.contrastive_type,
        with_voxel_norm=args.with_voxel_norm,
        augment_during_training=args.augment_during_training,
        shuffle_time_sequence=args.shuffle_time_sequence,
        label_scaling_method=args.label_scaling_method,
        decoder=args.decoder,
        adjust_hrf=args.adjust_hrf,
        input_offset=args.input_offset,
        stratified_params=args.stratified_params,
        train_split=args.train_split,
        val_split=args.val_split,
        bad_subj_path=args.bad_subj_path,
        limit_training_samples=args.limit_training_samples,
    )
    # Setup both 'fit' (for train_dataset/scaler) and 'test' stages
    data_module.setup(stage='fit')
    data_module.setup(stage='test')

    print(f"Train set size: {len(data_module.train_dataset)}")
    print(f"Test set size: {len(data_module.test_dataset)}")
    print()

    # Load model from checkpoint
    print("[Step 2] Loading model from checkpoint...")
    model = LitClassifier.load_from_checkpoint(
        args.checkpoint_path,
        data_module=data_module
    )
    print("Model loaded successfully!")
    print()

    # Create test dataloader with custom collate
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        data_module.test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Setup trainer
    print("[Step 3] Running test evaluation...")
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
    )

    # Run test
    results = trainer.test(model, dataloaders=test_loader)

    print()
    print("="*80)
    print("TEST RESULTS")
    print("="*80)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("="*80)
    print()
    print("✓ Evaluation complete!")


if __name__ == '__main__':
    main()
