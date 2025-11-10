#!/bin/bash
#SBATCH --job-name fxgvztr4_nonzero_metrics
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH -o /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o
#SBATCH -e /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/logs/%A-%x.o


echo "=========================================="
echo "Calculate Non-Zero Metrics for fxgvztr4"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swiftio

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO"

# Verify environment
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "=========================================="
echo "Experiment: fxgvztr4 (Ver 9 Regression)"
echo "Checkpoint: output/moviefmri/fxgvztr4/checkpt-epoch=06-valid_mse=0.06.ckpt"
echo "Output: docs/analysis/2025-11-10-fxgvztr4_nonzero_metrics.csv"
echo "=========================================="
echo ""

python -u calculate_nonzero_metrics_fxgvztr4.py \
    --checkpoint output/moviefmri/fxgvztr4/checkpt-epoch=06-valid_mse=0.06.ckpt \
    --output docs/analysis/2025-11-10-fxgvztr4_nonzero_metrics.csv \
    --batch_size 16 \
    --num_workers 4

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Non-zero metrics calculation completed!"
    echo ""
    echo "📊 Output file: docs/analysis/2025-11-10-fxgvztr4_nonzero_metrics.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Check the CSV file for detailed metrics"
    echo "  2. Compare with Ver 11 regression metrics"
    echo "  3. Analyze why Ver 9 regression metrics look good but plots are flat"
else
    echo "❌ Calculation failed with exit code $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "End Time: $(date)"
echo "=========================================="
