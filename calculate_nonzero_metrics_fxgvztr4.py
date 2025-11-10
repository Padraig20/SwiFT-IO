"""
Calculate non-zero metrics for fxgvztr4 (Ver 9 Regression)

This script loads the trained model checkpoint and re-evaluates on train/val/test sets,
computing metrics for both overall and non-zero targets only.

IMPORTANT: fxgvztr4 was trained with old SeriesDecoder initialization
(num_latents=embed_dim, num_latent_channels=dims). We need to temporarily
use old-style loading for this specific checkpoint.

Author: Claude
Date: 2025-11-10
"""

import torch
import pytorch_lightning as pl
import sys
import os

# Temporarily monkey-patch load_model to use old initialization for fxgvztr4
import src.module.models.load_model as load_model_module
_original_load_model = load_model_module.load_model

def _old_style_load_model(model_name, hparams=None):
    """Load model with old-style SeriesDecoder initialization for fxgvztr4 compatibility."""
    if model_name == "series_decoder":
        from src.module.models.decoder.series_decoder import SeriesDecoder

        # Calculate dimensions (copied from load_model.py)
        h, w, d, t_orig = hparams.img_size
        hp, wp, dp, tp = hparams.patch_size
        h = h // (hp*8) if hp != 1 else h
        w = w // (wp*8) if wp != 1 else w
        d = d // (dp*8) if dp != 1 else d
        t = t_orig // tp if tp != 1 else t_orig
        embed_dim = hparams.embed_dim * 8
        dims = h * w * d * t

        num_classes = 1 if hparams.downstream_task_type == 'regression' else hparams.num_classes

        # OLD-STYLE initialization for fxgvztr4 compatibility
        net = SeriesDecoder(
            num_latents=embed_dim,  # OLD: 288
            num_latent_channels=dims,  # OLD: 540
            num_output_queries=t_orig,
            num_classes=num_classes,
            num_targets=hparams.num_targets,
            downstream_task_type=hparams.downstream_task_type
        )
        return net
    else:
        # For all other models, use original load_model
        return _original_load_model(model_name, hparams)

# Apply monkey patch
load_model_module.load_model = _old_style_load_model

from src.module.pl_classifier import LitClassifier
from src.module.utils.data_module import fMRIDataModule
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def calculate_metrics(predictions, targets, split_name="test"):
    """Calculate both overall and non-zero metrics."""

    # Emotion names
    emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    n_emotions = len(emotions)

    results = {
        'split': split_name,
        'overall_mae': None,
        'overall_mse': None,
        'overall_nonzero_mae': None,
        'overall_nonzero_mse': None,
    }

    # Overall metrics
    results['overall_mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
    results['overall_mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
    results['overall_r2'] = r2_score(targets.flatten(), predictions.flatten())

    # Correlation
    try:
        corr, _ = pearsonr(targets.flatten(), predictions.flatten())
        results['overall_correlation'] = corr
    except:
        results['overall_correlation'] = None

    # Distribution statistics
    results['target_mean'] = float(np.mean(targets))
    results['target_std'] = float(np.std(targets))
    results['target_min'] = float(np.min(targets))
    results['target_max'] = float(np.max(targets))
    results['target_variance'] = float(np.var(targets))

    results['pred_mean'] = float(np.mean(predictions))
    results['pred_std'] = float(np.std(predictions))
    results['pred_min'] = float(np.min(predictions))
    results['pred_max'] = float(np.max(predictions))
    results['pred_variance'] = float(np.var(predictions))

    # Flatness indicator: ratio of prediction variance to target variance
    if results['target_variance'] > 0:
        results['variance_ratio'] = results['pred_variance'] / results['target_variance']
    else:
        results['variance_ratio'] = None

    # Non-zero overall metrics
    nonzero_mask = targets.flatten() != 0
    if nonzero_mask.sum() > 0:
        results['overall_nonzero_mae'] = mean_absolute_error(
            targets.flatten()[nonzero_mask],
            predictions.flatten()[nonzero_mask]
        )
        results['overall_nonzero_mse'] = mean_squared_error(
            targets.flatten()[nonzero_mask],
            predictions.flatten()[nonzero_mask]
        )
        results['overall_nonzero_r2'] = r2_score(
            targets.flatten()[nonzero_mask],
            predictions.flatten()[nonzero_mask]
        )

        # Non-zero correlation
        try:
            corr, _ = pearsonr(
                targets.flatten()[nonzero_mask],
                predictions.flatten()[nonzero_mask]
            )
            results['overall_nonzero_correlation'] = corr
        except:
            results['overall_nonzero_correlation'] = None

    # Per-emotion metrics
    for i, emotion in enumerate(emotions):
        emotion_targets = targets[:, i]
        emotion_preds = predictions[:, i]

        # Overall metrics for this emotion
        results[f'{emotion}_mae'] = mean_absolute_error(emotion_targets, emotion_preds)
        results[f'{emotion}_mse'] = mean_squared_error(emotion_targets, emotion_preds)
        results[f'{emotion}_r2'] = r2_score(emotion_targets, emotion_preds)

        # Correlation
        try:
            corr, _ = pearsonr(emotion_targets, emotion_preds)
            results[f'{emotion}_correlation'] = corr
        except:
            results[f'{emotion}_correlation'] = None

        # Distribution statistics
        results[f'{emotion}_target_mean'] = float(np.mean(emotion_targets))
        results[f'{emotion}_target_std'] = float(np.std(emotion_targets))
        results[f'{emotion}_target_variance'] = float(np.var(emotion_targets))

        results[f'{emotion}_pred_mean'] = float(np.mean(emotion_preds))
        results[f'{emotion}_pred_std'] = float(np.std(emotion_preds))
        results[f'{emotion}_pred_variance'] = float(np.var(emotion_preds))

        # Variance ratio (flatness indicator)
        if results[f'{emotion}_target_variance'] > 0:
            results[f'{emotion}_variance_ratio'] = results[f'{emotion}_pred_variance'] / results[f'{emotion}_target_variance']
        else:
            results[f'{emotion}_variance_ratio'] = None

        # Peak detection (threshold-based)
        peak_threshold = 5.0
        peak_mask = emotion_targets > peak_threshold
        if peak_mask.sum() > 0:
            results[f'{emotion}_peak_mae'] = mean_absolute_error(
                emotion_targets[peak_mask],
                emotion_preds[peak_mask]
            )
            results[f'{emotion}_peak_count'] = int(peak_mask.sum())
            results[f'{emotion}_peak_percent'] = float(peak_mask.sum() / len(emotion_targets) * 100)

            # Peak prediction quality: how many peaks did we predict?
            pred_peak_mask = emotion_preds > peak_threshold
            true_positives = (peak_mask & pred_peak_mask).sum()
            false_positives = (~peak_mask & pred_peak_mask).sum()
            false_negatives = (peak_mask & ~pred_peak_mask).sum()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[f'{emotion}_peak_precision'] = float(precision)
            results[f'{emotion}_peak_recall'] = float(recall)
            results[f'{emotion}_peak_f1'] = float(f1)
        else:
            results[f'{emotion}_peak_mae'] = None
            results[f'{emotion}_peak_count'] = 0
            results[f'{emotion}_peak_percent'] = 0.0
            results[f'{emotion}_peak_precision'] = None
            results[f'{emotion}_peak_recall'] = None
            results[f'{emotion}_peak_f1'] = None

        # Non-zero metrics for this emotion
        nonzero_mask = emotion_targets != 0
        if nonzero_mask.sum() > 0:
            results[f'{emotion}_nonzero_mae'] = mean_absolute_error(
                emotion_targets[nonzero_mask],
                emotion_preds[nonzero_mask]
            )
            results[f'{emotion}_nonzero_mse'] = mean_squared_error(
                emotion_targets[nonzero_mask],
                emotion_preds[nonzero_mask]
            )
            results[f'{emotion}_nonzero_r2'] = r2_score(
                emotion_targets[nonzero_mask],
                emotion_preds[nonzero_mask]
            )

            # Non-zero correlation
            try:
                corr, _ = pearsonr(
                    emotion_targets[nonzero_mask],
                    emotion_preds[nonzero_mask]
                )
                results[f'{emotion}_nonzero_correlation'] = corr
            except:
                results[f'{emotion}_nonzero_correlation'] = None

            results[f'{emotion}_nonzero_count'] = int(nonzero_mask.sum())
            results[f'{emotion}_zero_count'] = int((~nonzero_mask).sum())
            results[f'{emotion}_nonzero_percent'] = float(nonzero_mask.sum() / len(emotion_targets) * 100)
        else:
            results[f'{emotion}_nonzero_mae'] = None
            results[f'{emotion}_nonzero_mse'] = None
            results[f'{emotion}_nonzero_r2'] = None
            results[f'{emotion}_nonzero_correlation'] = None
            results[f'{emotion}_nonzero_count'] = 0
            results[f'{emotion}_zero_count'] = int(len(emotion_targets))
            results[f'{emotion}_nonzero_percent'] = 0.0

    return results


def evaluate_model(checkpoint_path, data_module, device='cuda', save_predictions=True, output_dir='docs/analysis'):
    """Load model and evaluate on all splits."""

    print(f"Loading checkpoint: {checkpoint_path}")
    # Monkey-patched load_model ensures old-style SeriesDecoder initialization
    model = LitClassifier.load_from_checkpoint(checkpoint_path, data_module=data_module)
    model = model.to(device)

    # Convert to float16 if checkpoint was trained with precision=16
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    use_fp16 = ckpt['hyper_parameters'].get('precision') == 16
    if use_fp16:
        model = model.half()
        print("Model converted to float16 (precision=16)")

    model.eval()

    results = {}
    predictions_dict = {}

    # Evaluate on each split
    # Note: val_dataloader() returns [val_loader, test_loader], so we need to handle it
    val_loaders = data_module.val_dataloader()
    if isinstance(val_loaders, list):
        val_loader = val_loaders[0]  # First is val, second is test
    else:
        val_loader = val_loaders

    for split_name, dataloader in [
        ('train', data_module.train_dataloader()),
        ('val', val_loader),
        ('test', data_module.test_dataloader())
    ]:
        print(f"\nEvaluating on {split_name} set...")

        all_preds = []
        all_targets = []
        all_subject_ids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    print(f"  Processing batch {batch_idx}/{len(dataloader)}")

                images = batch['fmri_sequence'].to(device)
                if use_fp16:
                    images = images.half()
                targets = batch['target'].cpu().numpy()

                # Get subject IDs if available
                if 'subject_name' in batch:
                    subject_ids = batch['subject_name']
                    all_subject_ids.extend(subject_ids)

                # Forward pass
                outputs = model(images)
                preds = outputs.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        print(f"  {split_name} set: {len(all_preds)} samples")
        print(f"  Predictions shape: {all_preds.shape}")
        print(f"  Targets shape: {all_targets.shape}")

        # Save predictions and targets
        if save_predictions:
            predictions_dict[split_name] = {
                'predictions': all_preds,
                'targets': all_targets,
                'subject_ids': all_subject_ids if all_subject_ids else None
            }

        # Calculate metrics
        split_results = calculate_metrics(all_preds, all_targets, split_name)
        results[split_name] = split_results

    # Save predictions to file
    if save_predictions:
        output_path = Path(output_dir) / '2025-11-10-fxgvztr4_predictions.npz'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        save_data = {}
        for split_name, data in predictions_dict.items():
            save_data[f'{split_name}_predictions'] = data['predictions']
            save_data[f'{split_name}_targets'] = data['targets']
            if data['subject_ids'] is not None:
                save_data[f'{split_name}_subject_ids'] = np.array(data['subject_ids'])

        np.savez_compressed(output_path, **save_data)
        print(f"\n✅ Predictions and targets saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return results


def print_results_table(results):
    """Print results in a formatted table."""

    emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    print("\n" + "="*120)
    print("FXGVZTR4 NON-ZERO PERFORMANCE METRICS TABLE")
    print("="*120)

    for split in ['train', 'val', 'test']:
        if split not in results:
            continue

        r = results[split]

        print(f"\n{'='*120}")
        print(f"{split.upper()} SET METRICS")
        print(f"{'='*120}")

        print(f"\nOverall Metrics:")
        print(f"  Overall MAE: {r['overall_mae']:.6f}")
        print(f"  Overall MSE: {r['overall_mse']:.6f}")
        print(f"  Overall MAE (non-zero only): {r['overall_nonzero_mae']:.6f}")
        print(f"  Overall MSE (non-zero only): {r['overall_nonzero_mse']:.6f}")

        print(f"\nPer-Emotion Metrics:")
        print(f"  {'Emotion':>12} | {'MAE':>10} | {'MSE':>10} | {'MAE(nz)':>10} | {'MSE(nz)':>10} | {'% Non-Zero':>10}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for emotion in emotions:
            mae = r[f'{emotion}_mae']
            mse = r[f'{emotion}_mse']
            mae_nz = r[f'{emotion}_nonzero_mae']
            mse_nz = r[f'{emotion}_nonzero_mse']
            pct_nz = r[f'{emotion}_nonzero_percent']

            mae_str = f"{mae:.6f}" if mae is not None else "N/A"
            mse_str = f"{mse:.6f}" if mse is not None else "N/A"
            mae_nz_str = f"{mae_nz:.6f}" if mae_nz is not None else "N/A"
            mse_nz_str = f"{mse_nz:.6f}" if mse_nz is not None else "N/A"
            pct_str = f"{pct_nz:.2f}%" if pct_nz is not None else "N/A"

            print(f"  {emotion:>12} | {mae_str:>10} | {mse_str:>10} | {mae_nz_str:>10} | {mse_nz_str:>10} | {pct_str:>10}")


def save_results_to_csv(results, output_path):
    """Save results to CSV file."""

    rows = []
    for split, split_results in results.items():
        for key, value in split_results.items():
            rows.append({
                'split': split,
                'metric': key,
                'value': value
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='output/moviefmri/fxgvztr4/checkpt-epoch=06-valid_mse=0.06.ckpt',
                       help='Path to checkpoint file')
    parser.add_argument('--output', type=str,
                       default='docs/analysis/2025-11-10-fxgvztr4_nonzero_metrics.csv',
                       help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Load checkpoint hyperparameters
    print("Loading checkpoint hyperparameters...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    hparams = ckpt['hyper_parameters']

    # Setup data module using checkpoint hyperparameters
    print("Setting up data module...")
    data_module = fMRIDataModule(
        dataset_name=hparams['dataset_name'],
        image_path=hparams['image_path'],
        input_type=hparams['input_type'],
        batch_size=args.batch_size,  # Override for faster evaluation
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=hparams['sequence_length'],
        stride_between_seq=hparams['stride_between_seq'],
        stride_within_seq=hparams['stride_within_seq'],
        with_voxel_norm=hparams['with_voxel_norm'],
        shuffle_time_sequence=hparams['shuffle_time_sequence'],
        adjust_hrf=hparams['adjust_hrf'],
        input_offset=hparams['input_offset'],
        dataset_split_seed=hparams['dataset_split_seed'],
        train_split=hparams['train_split'],
        val_split=hparams['val_split'],
        stratified_params=hparams['stratified_params'],
        downstream_task=hparams['downstream_task'],
        downstream_task_type=hparams['downstream_task_type'],
        decoder=hparams['decoder'],
        use_contrastive=hparams['use_contrastive'],
        contrastive_type=hparams['contrastive_type'],
        label_scaling_method=hparams['label_scaling_method'],
        bad_subj_path=None,
        limit_training_samples=None,
        img_size=hparams['img_size'],
    )
    data_module.setup('fit')
    data_module.setup('test')

    # Evaluate model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = evaluate_model(
        args.checkpoint,
        data_module,
        device=device,
        save_predictions=True,
        output_dir='docs/analysis'
    )

    # Print results
    print_results_table(results)

    # Save to CSV
    save_results_to_csv(results, args.output)

    print("\n" + "="*120)
    print("DONE! Non-zero metrics calculation complete.")
    print("="*120)


if __name__ == '__main__':
    main()
