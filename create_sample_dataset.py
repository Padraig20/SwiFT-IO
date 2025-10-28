"""
Create a small sample dataset for local testing

This script:
1. Samples a few subjects from the full dataset
2. Copies them to a smaller directory
3. Creates a tarball for easy download to local machine

Usage:
    python create_sample_dataset.py --num_subjects 5 --output_dir sample_data
"""

import os
import shutil
import argparse
from pathlib import Path
import random
import tarfile
from tqdm import tqdm


def create_sample_dataset(
    full_data_path: str,
    output_dir: str,
    num_subjects: int = 5,
    create_tarball: bool = True
):
    """
    Create a small sample dataset for local testing

    Args:
        full_data_path: Path to full dataset
        output_dir: Output directory for sample data
        num_subjects: Number of subjects to sample
        create_tarball: Whether to create a .tar.gz file
    """
    print("="*80)
    print("Creating Sample Dataset for Local Testing")
    print("="*80)

    # Get all subject directories
    full_path = Path(full_data_path)
    all_subjects = [d for d in full_path.iterdir() if d.is_dir()]

    print(f"\nFound {len(all_subjects)} subjects in {full_data_path}")

    # Sample random subjects
    if num_subjects > len(all_subjects):
        num_subjects = len(all_subjects)
        print(f"Warning: Only {len(all_subjects)} subjects available")

    sampled_subjects = random.sample(all_subjects, num_subjects)

    print(f"Sampling {num_subjects} subjects...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy sampled subjects
    total_size = 0

    for subject_dir in tqdm(sampled_subjects, desc="Copying subjects"):
        dest_dir = output_path / subject_dir.name

        # Copy entire subject directory
        shutil.copytree(subject_dir, dest_dir, dirs_exist_ok=True)

        # Calculate size
        subject_size = sum(f.stat().st_size for f in dest_dir.rglob('*') if f.is_file())
        total_size += subject_size

        print(f"  Copied {subject_dir.name}: {subject_size / 1e9:.2f} GB")

    print(f"\nTotal sample dataset size: {total_size / 1e9:.2f} GB")

    # Create metadata file
    metadata = {
        'num_subjects': num_subjects,
        'subjects': [s.name for s in sampled_subjects],
        'total_size_gb': total_size / 1e9,
        'source_path': str(full_data_path),
        'created_date': str(Path.ctime(output_path))
    }

    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSample dataset created in: {output_path}")

    # Create tarball for easy download
    if create_tarball:
        print("\nCreating tarball...")
        tarball_path = f"{output_dir}.tar.gz"

        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(output_path, arcname=output_path.name)

        tarball_size = os.path.getsize(tarball_path) / 1e9
        print(f"Tarball created: {tarball_path} ({tarball_size:.2f} GB)")
        print(f"\nTo download to local:")
        print(f"  scp kimbo@server:{tarball_path} ~/Downloads/")

    print("\n" + "="*80)
    print("Sample dataset creation complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Create sample dataset for local testing"
    )

    parser.add_argument(
        "--full_data_path",
        type=str,
        default="/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
        help="Path to full dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data_for_local",
        help="Output directory for sample data"
    )

    parser.add_argument(
        "--num_subjects",
        type=int,
        default=5,
        help="Number of subjects to sample"
    )

    parser.add_argument(
        "--no_tarball",
        action="store_true",
        help="Skip creating tarball"
    )

    args = parser.parse_args()

    create_sample_dataset(
        full_data_path=args.full_data_path,
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        create_tarball=not args.no_tarball
    )


if __name__ == "__main__":
    main()
