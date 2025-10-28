#!/bin/bash
# Classification IG Map Generation Pipeline
# Model: mc3r4vhf (seq20 classification model)

set -e  # Exit on error

PROJECT_ROOT="/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO"
RUN_ID="mc3r4vhf"

cd $PROJECT_ROOT

echo "=========================================="
echo "Classification IG Map Pipeline"
echo "Run ID: $RUN_ID"
echo "Start Time: $(date)"
echo "=========================================="

# Step 1: Evaluate all test subjects (classification task)
echo ""
echo "Step 1/4: Evaluating test subjects..."
echo "=========================================="
python evaluate_clf_test_subjects.py --run_id $RUN_ID

# Step 2: Analyze emotion-specific performance
echo ""
echo "Step 2/4: Analyzing emotion-specific performance..."
echo "=========================================="
python analyze_clf_emotion_performance.py

# Step 3: Select peak sequences (high confidence predictions)
echo ""
echo "Step 3/4: Selecting peak sequences..."
echo "=========================================="
python select_clf_peak_sequences.py --run_id $RUN_ID --top_k 10

# Step 4: Generate SLURM scripts for IG map generation
echo ""
echo "Step 4/4: Generating SLURM scripts..."
echo "=========================================="
python igmap/generate_clf_emotion_top5_ig_scripts.py

echo ""
echo "=========================================="
echo "✅ Pipeline setup complete!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the generated analysis:"
echo "     - analysis/4_IGmap/subject_performance/${RUN_ID}_clf_test_subject_performance.json"
echo "     - analysis/4_IGmap/subject_performance/${RUN_ID}_clf_emotion_specific_performance.json"
echo "     - analysis/4_IGmap/clf_peak_sequences/${RUN_ID}_clf_peak_sequences_top10.json"
echo ""
echo "  2. Submit IG map jobs:"
echo "     cd igmap/clf_emotion_top5_ig_jobs"
echo "     for script in run_igc_*.slurm; do sbatch \$script && sleep 0.5; done"
echo ""
echo "  3. Monitor jobs:"
echo "     squeue -u \$USER"
echo ""
