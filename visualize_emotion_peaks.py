#!/usr/bin/env python3
"""
Visualize emotion peak sequences selection.
Shows the entire emotion timeseries with highlighted selected sequences.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from matplotlib.cm import get_cmap

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")
sys.path.append(str(project_root / "src"))

from module.utils.data_module import fMRIDataModule

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

def visualize_subject_peaks(subject, run_id='opr6oq97', output_dir=None):
    """
    Visualize emotion timeseries with selected peak sequences for one subject.

    Args:
        subject: Subject ID
        run_id: Model run ID
        output_dir: Directory to save figures
    """
    print(f"\n{'='*70}")
    print(f"Visualizing Emotion Peaks for {subject}")
    print(f"{'='*70}")

    # Load checkpoint to get configuration
    ckpt_path = project_root / f"output/moviefmri/{run_id}/checkpt-epoch=07-valid_mse=0.05.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((project_root / f"output/moviefmri/{run_id}").glob("checkpt*"))[0]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    # Setup data module
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

    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 777)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = None
    if 'train_split' not in args_model_dict:
        args_model_dict['train_split'] = 0.7
    if 'val_split' not in args_model_dict:
        args_model_dict['val_split'] = 0.15

    print("\n🚀 Initializing data module...")
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

    # Get test dataset
    testset = data_module.test_dataset

    # Get sequences for this subject
    subject_sequences = [
        (idx, d) for idx, d in enumerate(testset.data)
        if str(d[1]) == subject
    ]

    if not subject_sequences:
        print(f"❌ No sequences found for {subject}")
        return

    print(f"✅ Found {len(subject_sequences)} sequences")

    # Reconstruct full timeseries
    # Assume all sequences are non-overlapping and cover the full scan
    first_data = subject_sequences[0][1]
    total_frames = first_data[5]  # num_frames

    print(f"Total frames: {total_frames}")

    # Initialize full timeseries array
    full_timeseries = np.zeros((total_frames, len(emotion_labels)))

    # Fill in the timeseries from sequences
    for seq_idx, (data_idx, data_tuple) in enumerate(subject_sequences):
        start_frame = data_tuple[3]
        seq_length = data_tuple[4]
        target = data_tuple[6]

        if torch.is_tensor(target):
            target = target.numpy()

        # Fill in the corresponding frames
        full_timeseries[start_frame:start_frame+seq_length, :] = target

    # Load selected peak sequences
    peaks_path = project_root / f"analysis/4_IGmap/emotion_peak_sequences/{run_id}_emotion_peak_sequences_top10.json"

    with open(peaks_path, 'r') as f:
        peaks_data = json.load(f)

    if subject not in peaks_data['subjects']:
        print(f"❌ No peak data for {subject}")
        return

    subject_peaks = peaks_data['subjects'][subject]

    # Create figure with subplots for each emotion
    fig, axes = plt.subplots(7, 1, figsize=(20, 14), sharex=True)
    fig.suptitle(f'Emotion Timeseries with Selected Peak Sequences\nSubject: {subject}',
                 fontsize=16, fontweight='bold')

    # Color map for ranking
    cmap = get_cmap('Reds')
    colors = [cmap(0.3 + 0.07*i) for i in range(10)]  # 10 different intensities

    for emotion_idx, emotion_name in enumerate(emotion_labels):
        ax = axes[emotion_idx]

        # Plot full timeseries
        ax.plot(range(total_frames), full_timeseries[:, emotion_idx],
                color='gray', linewidth=1, alpha=0.6, label='Full timeseries')

        # Highlight selected sequences
        peak_sequences = subject_peaks[emotion_name]

        for rank, seq_info in enumerate(peak_sequences):
            start = seq_info['start_frame']
            end = seq_info['end_frame']

            # Add shaded region
            ax.axvspan(start, end, alpha=0.3, color=colors[rank],
                      label=f'Rank {rank+1}' if rank < 3 else '')

            # Add text annotation for top 3
            if rank < 3:
                mid_point = (start + end) / 2
                y_pos = full_timeseries[start:end+1, emotion_idx].max()
                ax.text(mid_point, y_pos, f'#{rank+1}\n{seq_info["avg_score"]:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[rank], alpha=0.7))

        # Formatting
        ax.set_ylabel(emotion_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, total_frames)

        # Add legend only for first subplot
        if emotion_idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    # X-axis label
    axes[-1].set_xlabel('TR (Time)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = project_root / f"analysis/4_IGmap/emotion_peak_sequences/visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{subject}_emotion_peaks.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")

    # Also create a summary table figure
    fig_table, ax_table = plt.subplots(figsize=(14, 10))
    ax_table.axis('tight')
    ax_table.axis('off')

    # Create table data
    table_data = []
    table_data.append(['Emotion', 'Rank', 'TR Range', 'Avg Score', 'Max Score', 'Std Score'])

    for emotion_name in emotion_labels:
        peak_sequences = subject_peaks[emotion_name]
        for rank, seq_info in enumerate(peak_sequences[:5], 1):  # Show top 5
            tr_range = f"TR{seq_info['start_frame']:03d}-{seq_info['end_frame']:03d}"
            table_data.append([
                emotion_name if rank == 1 else '',
                str(rank),
                tr_range,
                f"{seq_info['avg_score']:.4f}",
                f"{seq_info['max_score']:.4f}",
                f"{seq_info['std_score']:.4f}"
            ])

    table = ax_table.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color emotion names
    row_idx = 1
    for emotion_idx, emotion_name in enumerate(emotion_labels):
        for rank in range(5):
            if row_idx < len(table_data):
                if rank == 0:
                    table[(row_idx, 0)].set_facecolor('#E8F5E9')
                    table[(row_idx, 0)].set_text_props(weight='bold')
                table[(row_idx, 1)].set_facecolor(colors[rank])
                row_idx += 1

    plt.title(f'Top 5 Peak Sequences per Emotion\nSubject: {subject}',
              fontsize=14, fontweight='bold', pad=20)

    table_path = output_dir / f"{subject}_emotion_peaks_table.png"
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    print(f"✅ Table saved to: {table_path}")

    plt.close('all')

    return output_path, table_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='opr6oq97')
    parser.add_argument('--subjects', type=str, nargs='+', default=['sub-NDARAH793FBF'],
                       help='Subjects to visualize (default: sub-NDARAH793FBF)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    for subject in args.subjects:
        try:
            visualize_subject_peaks(
                subject=subject,
                run_id=args.run_id,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"❌ Error processing {subject}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETED!")
    print("="*70)
