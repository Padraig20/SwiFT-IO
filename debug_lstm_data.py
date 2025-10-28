"""
Debug script to identify LSTM baseline issues
1. Metadata matching problem
2. Test step tensor shape problem

Usage: python debug_lstm_data.py
"""
import os
import sys
import pandas as pd
import numpy as np
import torch

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

print("="*80)
print("LSTM Baseline Debug Analysis")
print("="*80)

# ============================================================================
# Part 1: Metadata Matching Debug
# ============================================================================
print("\n" + "="*80)
print("PART 1: METADATA MATCHING DEBUG")
print("="*80)

image_path = '/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img'
metadata_path = '/scratch/HBN/0_meta/HBN_metadata_240501_CJB.csv'

print(f"\n[1] Checking data directory...")
print(f"Path: {image_path}")

if not os.path.exists(image_path):
    print(f"ERROR: Data path does not exist!")
    sys.exit(1)

# Get subject directories
subjects = sorted([d for d in os.listdir(image_path)
                  if os.path.isdir(os.path.join(image_path, d))])

print(f"✓ Found {len(subjects)} subject directories")
print(f"Sample data IDs (first 10):")
for i, s in enumerate(subjects[:10], 1):
    print(f"  {i}. {s}")

print(f"\n[2] Checking metadata CSV...")
print(f"Path: {metadata_path}")

if not os.path.exists(metadata_path):
    print(f"ERROR: Metadata path does not exist!")
    sys.exit(1)

df = pd.read_csv(metadata_path)
df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)

print(f"✓ Loaded metadata with {len(df)} entries")
print(f"Sample metadata IDs (first 10):")
for i, s in enumerate(df['SUBJECT_ID'].head(10), 1):
    print(f"  {i}. {s}")

print(f"\n[3] Checking ID format matching...")

# Direct overlap
data_ids = set(subjects)
metadata_ids = set(df['SUBJECT_ID'].values)

direct_overlap = data_ids & metadata_ids
print(f"\nDirect ID overlap: {len(direct_overlap)} / {len(subjects)} subjects")

if direct_overlap:
    print(f"✓ Sample matching IDs:")
    for i, s in enumerate(list(direct_overlap)[:5], 1):
        print(f"  {i}. {s}")
else:
    print(f"✗ NO DIRECT MATCH!")

# Check format differences
print(f"\n[4] Analyzing ID format differences...")

def analyze_id_format(id_str):
    """Analyze ID format"""
    has_prefix = id_str.startswith('sub-')
    core_id = id_str.replace('sub-', '') if has_prefix else id_str
    return {
        'original': id_str,
        'has_prefix': has_prefix,
        'core_id': core_id,
        'length': len(id_str)
    }

# Analyze data IDs
data_format = [analyze_id_format(s) for s in list(data_ids)[:5]]
print(f"\nData ID format analysis:")
for fmt in data_format:
    print(f"  '{fmt['original']}' -> prefix={fmt['has_prefix']}, core='{fmt['core_id']}', len={fmt['length']}")

# Analyze metadata IDs
meta_format = [analyze_id_format(s) for s in list(metadata_ids)[:5]]
print(f"\nMetadata ID format analysis:")
for fmt in meta_format:
    print(f"  '{fmt['original']}' -> prefix={fmt['has_prefix']}, core='{fmt['core_id']}', len={fmt['length']}")

# Try normalization
print(f"\n[5] Testing ID normalization...")

def normalize_id(sid):
    """Normalize subject ID (ensure 'sub-' prefix)"""
    sid = str(sid).strip()
    if not sid.startswith('sub-'):
        return f'sub-{sid}'
    return sid

data_ids_norm = {normalize_id(s) for s in data_ids}
metadata_ids_norm = {normalize_id(s) for s in metadata_ids}

normalized_overlap = data_ids_norm & metadata_ids_norm
print(f"After normalization: {len(normalized_overlap)} / {len(subjects)} subjects matched")

if len(normalized_overlap) > len(direct_overlap):
    print(f"✓ Normalization IMPROVED matching by {len(normalized_overlap) - len(direct_overlap)} subjects")
    print(f"  Recommendation: USE ID NORMALIZATION")
else:
    print(f"⚠ Normalization did not help")

# ============================================================================
# Part 2: Test Step Shape Debug
# ============================================================================
print("\n" + "="*80)
print("PART 2: TEST STEP TENSOR SHAPE DEBUG")
print("="*80)

print(f"\n[1] Simulating LSTM output shapes...")

# LSTM baseline configuration (from logs)
batch_size = 16
seq_length = 30
num_emotions = 7

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_length}")
print(f"  Num emotions: {num_emotions}")

# Simulate LSTM output
print(f"\n[2] Creating simulated tensors...")

# LSTM logits output (from successful training)
logits = torch.randn(batch_size, seq_length, num_emotions)
print(f"\nLSTM logits shape: {logits.shape}")
print(f"  - Interpretation: (batch={batch_size}, time={seq_length}, emotions={num_emotions})")

# Target labels
target = torch.randn(batch_size, seq_length, num_emotions)
print(f"\nTarget labels shape: {target.shape}")
print(f"  - Interpretation: (batch={batch_size}, time={seq_length}, emotions={num_emotions})")

# Test the problematic code from pl_classifier.py:554
print(f"\n[3] Testing current test_step code...")
print(f"\nCurrent code (PROBLEMATIC):")
print(f"  output = [(logit.cpu().detach(), targets.cpu().item())")
print(f"            for logit, targets in zip(logits, target)]")

try:
    output = [(logit.cpu().detach(), targets.cpu().item())
              for logit, targets in zip(logits, target)]
    print(f"\n✓ Code worked (unexpected!)")
except RuntimeError as e:
    print(f"\n✗ Code FAILED with error:")
    print(f"  {e}")

    # Analyze why
    print(f"\n[4] Analyzing the error...")
    print(f"\nIn the loop:")
    print(f"  logits has shape {logits.shape}")
    print(f"  Iterating over first dimension gives:")

    for i, (logit, targets) in enumerate(zip(logits, target)):
        print(f"\n  Iteration {i}:")
        print(f"    logit shape: {logit.shape}")
        print(f"    targets shape: {targets.shape}")
        print(f"    targets.numel(): {targets.numel()} elements")

        # Try .item()
        try:
            val = targets.cpu().item()
            print(f"    targets.cpu().item(): SUCCESS (value={val})")
        except RuntimeError as e:
            print(f"    targets.cpu().item(): FAILED")
            print(f"      Error: {e}")
            print(f"      Reason: .item() requires single element, but got {targets.numel()} elements")

        if i >= 2:  # Only show first 3 iterations
            print(f"\n  ... (showing first 3 iterations)")
            break

# Test the corrected code
print(f"\n[5] Testing CORRECTED test_step code...")
print(f"\nCorrected code:")
print(f"  output = [(logit.cpu().detach(), targets.cpu().detach())")
print(f"            for logit, targets in zip(logits, target)]")

try:
    output = [(logit.cpu().detach(), targets.cpu().detach())
              for logit, targets in zip(logits, target)]
    print(f"\n✓ Corrected code WORKS!")
    print(f"  Output length: {len(output)}")
    print(f"  First element:")
    print(f"    logit shape: {output[0][0].shape}")
    print(f"    target shape: {output[0][1].shape}")
except Exception as e:
    print(f"\n✗ Corrected code FAILED (unexpected):")
    print(f"  {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print(f"\n[Issue 1] Metadata Matching:")
if len(normalized_overlap) > 0:
    print(f"  ✓ SOLUTION FOUND: Use ID normalization")
    print(f"  ✓ Matched subjects: {len(normalized_overlap)} / {len(subjects)}")
    print(f"  → Action: Add normalize_id() to data_module.py")
else:
    print(f"  ✗ ISSUE REMAINS: No overlap even with normalization")
    print(f"  → Action: Manual investigation needed")

print(f"\n[Issue 2] Test Step Shape:")
print(f"  ✗ PROBLEM: Using .item() on multi-element tensor")
print(f"  ✓ SOLUTION: Remove .item(), use .detach() only")
print(f"  → Action: Modify pl_classifier.py test_step for LSTM")

print(f"\n[Code Changes Required]:")
print(f"  1. data_module.py: Add ID normalization (~15 lines)")
print(f"  2. pl_classifier.py: Fix test_step for LSTM (~10 lines)")

print(f"\n[Expected Result]:")
print(f"  ✓ LSTM training will start successfully")
print(f"  ✓ Test phase will complete without errors")

print("\n" + "="*80)
print("Debug analysis complete!")
print("="*80)
