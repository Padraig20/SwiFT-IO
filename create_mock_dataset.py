"""
Create mock fMRI dataset for local testing

This creates synthetic data that has the same structure as real data
but is randomly generated, so it's much smaller and faster to create.

Perfect for:
- Testing code logic
- Debugging data loading
- CI/CD pipelines
- Quick local development

Usage:
    python create_mock_dataset.py --num_subjects 3 --output_dir mock_data
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json


def create_mock_subject(subject_id: str, output_dir: Path, num_frames: int = 750):
    """
    Create mock data for one subject

    Args:
        subject_id: Subject identifier
        output_dir: Output directory
        num_frames: Number of frames (default: 750)
    """
    subject_dir = output_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Create mock fMRI frames (96, 96, 96)
    for frame_idx in range(num_frames):
        # Random brain activation pattern
        frame = np.random.randn(96, 96, 96).astype(np.float32)

        # Add some structure (not completely random)
        # Simulate brain regions with correlated activity
        frame = np.clip(frame, -3, 3)  # Clip outliers

        # Save as .pt file (same format as real data)
        frame_path = subject_dir / f"frame_{frame_idx}.pt"
        torch.save(torch.from_numpy(frame), frame_path)

    return subject_dir


def create_mock_labels(output_dir: Path, num_subjects: int, num_frames: int = 750):
    """
    Create mock emotion labels

    Args:
        output_dir: Output directory
        num_subjects: Number of subjects
        num_frames: Number of frames per subject
    """
    # 7 emotions: Anger, Happy, Fear, Sad, Excited, Positive, Negative
    num_emotions = 7
    emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    for subj_idx in range(num_subjects):
        subject_id = f"sub-{subj_idx+1:04d}"

        # Create random emotion time series
        # Simulate smooth temporal dynamics (not too random)
        emotions = np.random.randn(num_frames, num_emotions).astype(np.float32)

        # Apply smoothing to simulate realistic temporal dynamics
        from scipy.ndimage import gaussian_filter1d
        for e in range(num_emotions):
            emotions[:, e] = gaussian_filter1d(emotions[:, e], sigma=5)

        # Normalize to [-1, 1] range
        emotions = np.clip(emotions, -2, 2)

        # Save as numpy file
        labels_path = output_dir / f"{subject_id}_emotions.npy"
        np.save(labels_path, emotions)

    print(f"Created emotion labels for {num_subjects} subjects")


def create_mock_dataset(
    output_dir: str,
    num_subjects: int = 3,
    num_frames: int = 750
):
    """
    Create complete mock dataset

    Args:
        output_dir: Output directory
        num_subjects: Number of subjects to create
        num_frames: Number of frames per subject
    """
    print("="*80)
    print("Creating Mock Dataset for Local Testing")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {num_subjects} mock subjects with {num_frames} frames each...")

    # Create mock subjects
    subject_ids = []
    for i in tqdm(range(num_subjects), desc="Creating subjects"):
        subject_id = f"sub-{i+1:04d}"
        subject_ids.append(subject_id)
        create_mock_subject(subject_id, output_path, num_frames)

    # Create mock labels
    create_mock_labels(output_path, num_subjects, num_frames)

    # Create metadata
    metadata = {
        'dataset_type': 'mock',
        'num_subjects': num_subjects,
        'num_frames_per_subject': num_frames,
        'frame_shape': [96, 96, 96],
        'num_emotions': 7,
        'emotion_names': ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative'],
        'subjects': subject_ids,
        'note': 'This is synthetic data for testing purposes only'
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())

    print(f"\nMock dataset created!")
    print(f"  Location: {output_path}")
    print(f"  Size: {total_size / 1e9:.2f} GB")
    print(f"  Subjects: {num_subjects}")
    print(f"  Frames per subject: {num_frames}")

    print("\n" + "="*80)
    print("You can now test your code with this mock dataset locally!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Create mock fMRI dataset for testing"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="mock_data_for_local",
        help="Output directory"
    )

    parser.add_argument(
        "--num_subjects",
        type=int,
        default=3,
        help="Number of subjects (default: 3)"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=750,
        help="Number of frames per subject (default: 750)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 subjects, 100 frames each"
    )

    args = parser.parse_args()

    if args.quick:
        args.num_subjects = 2
        args.num_frames = 100
        print("Quick mode: Creating 2 subjects with 100 frames each")

    create_mock_dataset(
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        num_frames=args.num_frames
    )


if __name__ == "__main__":
    main()
