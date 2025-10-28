#!/usr/bin/env python3
"""
Visualize emotion labels over time for movieDM task
Generates a comprehensive PDF plot showing all 7 emotions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_emotion_labels():
    """Load emotion labels from CSV"""
    csv_path = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/data_behavior/DespicableMe_summary_codes_1.2Hz_intuitivenames_260120.csv"

    df = pd.read_csv(csv_path)

    # Select emotion columns
    emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

    # Extract frame and emotion data
    data = df[['frame'] + emotions].copy()
    data = data.dropna()
    data = data.sort_values('frame').reset_index(drop=True)

    return data, emotions

def compute_statistics(data, emotions):
    """Compute statistics for each emotion"""
    stats = {}
    for emotion in emotions:
        values = data[emotion].values
        stats[emotion] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'variance': np.var(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0  # Coefficient of variation
        }
    return stats

def visualize_emotions(data, emotions, output_path):
    """Create comprehensive emotion visualization"""

    # Compute statistics
    stats = compute_statistics(data, emotions)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.25)

    # Color palette for emotions
    colors = {
        'Anger': '#d62728',      # Red
        'Happy': '#ffd700',      # Gold
        'Fear': '#9467bd',       # Purple
        'Sad': '#1f77b4',        # Blue
        'Excited': '#ff7f0e',    # Orange
        'Positive': '#2ca02c',   # Green
        'Negative': '#8c564b'    # Brown
    }

    # Plot 1: All emotions together
    ax1 = fig.add_subplot(gs[0, :])
    for emotion in emotions:
        ax1.plot(data['frame'], data[emotion], label=emotion,
                color=colors[emotion], alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Emotion Valence')
    ax1.set_title('All Emotions Over Time (movieDM)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=7, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2-8: Individual emotions
    for i, emotion in enumerate(emotions):
        row = 1 + (i // 2)
        col = i % 2
        ax = fig.add_subplot(gs[row, col])

        # Plot time series
        ax.plot(data['frame'], data[emotion], color=colors[emotion],
               linewidth=2, alpha=0.8)
        ax.fill_between(data['frame'], data[emotion], alpha=0.3, color=colors[emotion])

        # Add horizontal line at mean
        mean_val = stats[emotion]['mean']
        ax.axhline(y=mean_val, color='red', linestyle='--',
                  linewidth=1, alpha=0.6, label=f'Mean: {mean_val:.2f}')

        # Styling
        ax.set_xlabel('Frame')
        ax.set_ylabel('Valence')
        ax.set_title(f'{emotion}', fontsize=12, fontweight='bold', color=colors[emotion])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Add statistics text box
        stat_text = (f'Mean: {stats[emotion]["mean"]:.2f}\n'
                    f'Std: {stats[emotion]["std"]:.2f}\n'
                    f'Range: [{stats[emotion]["min"]:.1f}, {stats[emotion]["max"]:.1f}]\n'
                    f'Var: {stats[emotion]["variance"]:.2f}\n'
                    f'CV: {stats[emotion]["cv"]:.2f}')
        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add overall title and info
    fig.suptitle('Emotion Label Analysis: movieDM Task (HBN Dataset)',
                fontsize=16, fontweight='bold', y=0.995)

    # Add footer with data info
    total_frames = len(data)
    duration_seconds = total_frames * 0.833  # 1.2Hz → 0.833s per frame
    duration_minutes = duration_seconds / 60

    footer_text = (f'Dataset: Despicable Me (movieDM) | '
                  f'Total Frames: {total_frames} | '
                  f'Sampling Rate: 1.2 Hz | '
                  f'Duration: {duration_minutes:.1f} minutes')
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved emotion visualization to: {output_path}")

    # Also save as PNG for quick viewing
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved PNG version to: {png_path}")

    plt.close()

    return stats

def print_statistics(stats, emotions):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("EMOTION LABEL STATISTICS")
    print("="*80)
    print(f"{'Emotion':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'Var':<8} {'CV':<8}")
    print("-"*80)
    for emotion in emotions:
        s = stats[emotion]
        print(f"{emotion:<10} {s['mean']:<8.2f} {s['std']:<8.2f} {s['min']:<8.2f} "
              f"{s['max']:<8.2f} {s['range']:<8.2f} {s['variance']:<8.2f} {s['cv']:<8.2f}")
    print("="*80)

    # Identify potential issues
    print("\n⚠️  POTENTIAL ISSUES:")
    print("-"*80)

    # Check for low variance emotions
    low_var_threshold = 1.0
    low_var_emotions = [e for e in emotions if stats[e]['variance'] < low_var_threshold]
    if low_var_emotions:
        print(f"🔴 Low variance emotions (Var < {low_var_threshold}): {', '.join(low_var_emotions)}")
        print("   → May cause R² calculation issues (denominator close to zero)")

    # Check for near-constant emotions
    low_range_threshold = 2.0
    low_range_emotions = [e for e in emotions if stats[e]['range'] < low_range_threshold]
    if low_range_emotions:
        print(f"🔴 Low range emotions (Range < {low_range_threshold}): {', '.join(low_range_emotions)}")
        print("   → Difficult to predict, may lead to inflated metrics")

    # Check coefficient of variation
    high_cv_emotions = [e for e in emotions if stats[e]['cv'] > 1.0]
    if high_cv_emotions:
        print(f"🟡 High variability emotions (CV > 1.0): {', '.join(high_cv_emotions)}")
        print("   → Highly variable, may be harder to predict")

    low_cv_emotions = [e for e in emotions if stats[e]['cv'] < 0.3]
    if low_cv_emotions:
        print(f"🟡 Low variability emotions (CV < 0.3): {', '.join(low_cv_emotions)}")
        print("   → Relatively stable, easier to predict")

    print("="*80)

def main():
    """Main function"""
    print("Starting emotion label visualization...")

    # Load data
    print("\n[1/3] Loading emotion labels...")
    data, emotions = load_emotion_labels()
    print(f"✓ Loaded {len(data)} frames with {len(emotions)} emotions")

    # Create visualization
    print("\n[2/3] Creating visualization...")
    output_path = "/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/docs/baselines/emotion_labels_movieDM.pdf"
    stats = visualize_emotions(data, emotions, output_path)

    # Print statistics
    print("\n[3/3] Computing statistics...")
    print_statistics(stats, emotions)

    print("\n✅ Done! Check the PDF and PNG files in docs/baselines/")

if __name__ == "__main__":
    main()
