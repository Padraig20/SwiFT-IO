#!/bin/bash

PROJECT_ROOT="/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO"
cd $PROJECT_ROOT

# Create docs directory structure
mkdir -p docs/{baselines,lstm,analysis,technical}

# Move baseline-related docs
for file in \
    "251014_SVR_BASELINE_RATIONALE.md" \
    "251014_SVM_vs_SVR_baseline.md" \
    "251020_SVR_ROI_BASELINE_RESULTS.md" \
    "251022_SVR_PCA_Optimization.md" \
    "251023_SVR_Baseline_Methods_Comparison.md" \
    "251025_svr_baseline_comparison_methodology.md" \
    "BASELINE_SETUP.md" \
    "SVR_BASELINE_READY.md" \
    "baseline_performance_table.md"
do
    if [ -f "$file" ]; then
        mv "$file" docs/baselines/
        echo "✓ Moved $file"
    fi
done

# Move LSTM-related docs
for file in \
    "251014_LSTM_BASELINE.md" \
    "LSTM_BASELINE_ERROR_ANALYSIS.md" \
    "LSTM_FIX_SUMMARY.md"
do
    if [ -f "$file" ]; then
        mv "$file" docs/lstm/
        echo "✓ Moved $file"
    fi
done

# Move analysis docs
for file in \
    "251025_emotion_specific_performance_analysis.md" \
    "251022_Baseline_Progress_Summary.md" \
    "251021_TRAINED_MODELS_SUMMARY.md"
do
    if [ -f "$file" ]; then
        mv "$file" docs/analysis/
        echo "✓ Moved $file"
    fi
done

# Move technical docs
for file in \
    "BUS_ERROR_FIX.md" \
    "ROI_IMPLEMENTATION.md"
do
    if [ -f "$file" ]; then
        mv "$file" docs/technical/
        echo "✓ Moved $file"
    fi
done

echo ""
echo "✅ Documentation organized!"
echo ""
echo "Summary:"
echo "  Baselines: $(ls docs/baselines/*.md 2>/dev/null | wc -l) files"
echo "  LSTM: $(ls docs/lstm/*.md 2>/dev/null | wc -l) files"
echo "  Analysis: $(ls docs/analysis/*.md 2>/dev/null | wc -l) files"
echo "  Technical: $(ls docs/technical/*.md 2>/dev/null | wc -l) files"
