#!/bin/bash
# Submit IG map jobs for top 5 subjects (as separate jobs on different nodes)

RUN_ID="opr6oq97"
TOP5_FILE="analysis/4_IGmap/subject_performance/${RUN_ID}_top5_subjects.txt"

echo "========================================================================"
echo "Submitting IG Map Jobs for Top 5 Subjects"
echo "========================================================================"

# Check if top 5 file exists
if [ ! -f "$TOP5_FILE" ]; then
    echo "❌ Error: Top 5 subjects file not found: $TOP5_FILE"
    echo "Please run MSE evaluation first:"
    echo "  sbatch run_evaluate_test_subjects.slurm"
    exit 1
fi

# Read subjects
mapfile -t SUBJECTS < "$TOP5_FILE"

echo "Top 5 Subjects:"
for i in "${!SUBJECTS[@]}"; do
    echo "  $((i+1)). ${SUBJECTS[$i]}"
done
echo ""

# Available nodes (adjust as needed)
NODES=("node1" "node2" "node3" "node4" "node5")

# Submit jobs
JOB_IDS=()
for i in "${!SUBJECTS[@]}"; do
    SUBJECT="${SUBJECTS[$i]}"
    NODE="${NODES[$i]}"

    echo "Submitting job $((i+1))/5: $SUBJECT on $NODE"

    # Create individual job script
    JOBFILE="igmap/job_${SUBJECT}.slurm"

    cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=ig_${SUBJECT: -8}
#SBATCH --output=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_${SUBJECT}-%j.o
#SBATCH --error=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/ig_${SUBJECT}-%j.o
#SBATCH --partition=debug
#SBATCH --nodelist=${NODE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0

echo "=========================================="
echo "IG Map - Subject: ${SUBJECT}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "=========================================="

export PYTHONPATH=/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO:\$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

/home/connectome/kimbo/.conda/envs/swiftio/bin/python igmap/igmap_selective_sequences.py \\
    --run_id ${RUN_ID} \\
    --subjects ${SUBJECT} \\
    --top_k_seqs 5 \\
    --n_steps 20 \\
    --project_root /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

echo "=========================================="
echo "End Time: \$(date)"
echo "=========================================="
EOF

    # Submit job
    JOB_ID=$(sbatch "$JOBFILE" | awk '{print $4}')
    JOB_IDS+=("$JOB_ID")

    echo "  ✅ Submitted: Job ID $JOB_ID"
    echo ""
done

echo "========================================================================"
echo "Summary:"
echo "========================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check outputs:"
for SUBJECT in "${SUBJECTS[@]}"; do
    echo "  tail -f logs/ig_${SUBJECT}-*.o"
done
echo "========================================================================"
