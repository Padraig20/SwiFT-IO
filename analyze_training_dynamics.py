#!/usr/bin/env python3
"""
Analyze training dynamics to understand why seq 30 collapsed
Compare seq 20 vs seq 30 training curves
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO")

# Load wandb exports
seq20_path = project_root / "output/moviefmri/mc3r4vhf/wandb_export_2025-10-27T11_47_13.241+09_00.csv"
seq30_path = project_root / "output/moviefmri/gajr5p1p/wandb_export_2025-10-27T11_47_13.241+09_00.csv"

df_seq20 = pd.read_csv(seq20_path)
df_seq30 = pd.read_csv(seq30_path)

print("="*80)
print("TRAINING DYNAMICS COMPARISON")
print("="*80)

# Extract key metrics
print("\n📊 Final Metrics:")
print("-"*80)

metrics_to_compare = [
    'epoch',
    'valid_AUROC_global',
    'test_AUROC_global',
    'valid_AUROC_0', 'valid_AUROC_1', 'valid_AUROC_2', 'valid_AUROC_3',
    'valid_AUROC_4', 'valid_AUROC_5', 'valid_AUROC_6',
]

print(f"\n{'Metric':<25} {'Seq 20':<15} {'Seq 30':<15} {'Difference':<15}")
print("-"*80)

for metric in metrics_to_compare:
    if metric in df_seq20.columns and metric in df_seq30.columns:
        val20 = df_seq20[metric].iloc[-1]
        val30 = df_seq30[metric].iloc[-1]

        if pd.notna(val20) and pd.notna(val30):
            diff = val20 - val30
            print(f"{metric:<25} {val20:<15.4f} {val30:<15.4f} {diff:+.4f}")

print("\n"+"="*80)
print("📈 KEY OBSERVATIONS:")
print("="*80)

# Check for early stopping
best_epoch_seq20 = "8" if "checkpt-epoch=08" in str(df_seq20.iloc[0]) else "Unknown"
best_epoch_seq30 = "3" if "checkpt-epoch=03" in str(df_seq30.iloc[0]) else "Unknown"

print(f"\n1. Training Duration:")
print(f"   - Seq 20: {df_seq20['epoch'].iloc[-1]:.0f} epochs, best at epoch {best_epoch_seq20}")
print(f"   - Seq 30: {df_seq30['epoch'].iloc[-1]:.0f} epochs, best at epoch {best_epoch_seq30}")
print(f"   → Seq 30 stopped improving very early!")

print(f"\n2. Validation AUROC:")
auroc_20 = df_seq20['valid_AUROC_global'].iloc[-1] if 'valid_AUROC_global' in df_seq20.columns else 0
auroc_30 = df_seq30['valid_AUROC_global'].iloc[-1] if 'valid_AUROC_global' in df_seq30.columns else 0
print(f"   - Seq 20: {auroc_20:.4f}")
print(f"   - Seq 30: {auroc_30:.4f}")
print(f"   → Difference: {auroc_20 - auroc_30:.4f}")

if auroc_30 < 0.65:
    print("   ⚠️  WARNING: Seq 30 AUROC < 0.65 suggests model collapse to majority class!")

print(f"\n3. Per-Emotion Analysis:")
emotion_names = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
for i, emo in enumerate(emotion_names):
    col_name = f'valid_AUROC_{i}'
    if col_name in df_seq20.columns and col_name in df_seq30.columns:
        auroc_20_emo = df_seq20[col_name].iloc[-1]
        auroc_30_emo = df_seq30[col_name].iloc[-1]

        if pd.notna(auroc_20_emo) and pd.notna(auroc_30_emo):
            diff = auroc_20_emo - auroc_30_emo
            status = "🔴 COLLAPSED" if auroc_30_emo < 0.65 else "🟡 POOR" if auroc_30_emo < 0.75 else "🟢 OK"
            print(f"   {emo:<10}: Seq20={auroc_20_emo:.3f}, Seq30={auroc_30_emo:.3f}, Δ={diff:+.3f} {status}")

print("\n"+"="*80)
print("💡 RECOMMENDATIONS:")
print("="*80)

if auroc_30 < 0.65:
    print("\n⚠️  Seq 30 shows clear signs of model collapse!")
    print("\nNext steps to validate:")
    print("1. ✅ Try different random seeds (777, 888, 999)")
    print("2. ✅ Test intermediate lengths (15, 20, 25, 30, 35)")
    print("3. ✅ Evaluate best checkpoint (epoch 3) instead of last")
    print("4. ✅ Check if class imbalance worsens with longer sequences")
    print("5. ✅ Reduce batch size (longer sequences = more memory = gradient issues)")
    print("6. ✅ Add gradient clipping analysis")
    print("7. ✅ Check if loss increased after epoch 3")
else:
    print("\nSeq 30 performance is suboptimal but not collapsed.")
    print("Consider hyperparameter tuning.")

print("\n" + "="*80)
