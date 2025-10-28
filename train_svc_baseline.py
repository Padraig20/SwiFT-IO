#!/usr/bin/env python3
"""
Train SVC (Support Vector Classification) baseline for emotion classification

This script trains the SVC baseline for classification task with seq_length=20
to match the classification experiment (mc3r4vhf).

Usage:
    python train_svc_baseline.py \
        --task_type classification \
        --sequence_length 20 \
        --dataset_split_seed 2 \
        --output_dir output/svc_baseline_seq20
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_svr_baseline import main

if __name__ == "__main__":
    # Override default arguments for classification
    import argparse

    # Get original parser
    original_argv = sys.argv.copy()

    # Add classification-specific defaults if not provided
    if '--task_type' not in sys.argv:
        sys.argv.extend(['--task_type', 'classification'])
    if '--sequence_length' not in sys.argv:
        sys.argv.extend(['--sequence_length', '20'])
    if '--dataset_split_seed' not in sys.argv:
        sys.argv.extend(['--dataset_split_seed', '2'])
    if '--downstream_task_type' not in sys.argv:
        sys.argv.extend(['--downstream_task_type', 'classification'])

    # Run main training
    main()
