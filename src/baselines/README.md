# Baseline Models for SwiFT-IO

This directory contains baseline models for comparison with SwiFT-IO.

## Available Baselines

### 1. SVR (Support Vector Regression) - **Primary Baseline**

**Fair comparison baseline that uses identical input as SwiFT-IO.**

**Features:**
- Uses **same 4D fMRI input** as SwiFT-IO (96×96×96×30 sequences)
- Uses **same train/val/test splits** as SwiFT-IO
- Flattens spatial dimensions and trains one SVR per emotion
- Provides fair model comparison (SVR vs Transformer)
- Predicts 7 emotions: Anger, Happy, Fear, Sad, Excited, Positive, Negative

**Architecture:**
```
4D fMRI (96×96×96×30) → Flatten → SVR (per emotion) → Emotion Predictions
```

### 2. GLM (General Linear Model) - **Dimension-Reduced Baseline**

Traditional GLM-based approach for emotion prediction from fMRI data.

**Features:**
- Uses pre-extracted ROI timeseries from `/scratch/HBN/9.2.movieDM_ROI_timeseries/`
- Selects emotion-related ROIs using `EmotionROISelector`
- Fits Ridge regression (with cross-validation for alpha selection)
- Uses the same train/val/test splits as SwiFT-IO
- Predicts 7 emotions: Anger, Happy, Fear, Sad, Excited, Positive, Negative

**Architecture:**
```
ROI Timeseries (34 emotion ROIs) → Ridge Regression → Emotion Predictions
```

## Quick Start

### Training SVR Baseline (Recommended)

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run with default settings
bash run_svr_baseline.sh

# Or customize parameters
python src/train_svr_baseline.py \
    --image_path /scratch/HBN/9.2.movieDM_SwiFT \
    --downstream_task emotions \
    --input_type movieDM \
    --dataset_split_seed 777 \
    --sequence_length 30 \
    --kernel rbf \
    --C 1.0 \
    --output_dir output/svr_baseline
```

### Training GLM Baseline

```bash
cd /scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO

# Run with default settings
bash run_glm_baseline.sh

# Or customize parameters
python src/train_glm_baseline.py \
    --image_path /scratch/HBN/9.2.movieDM_SwiFT \
    --downstream_task emotions \
    --input_type movieDM \
    --dataset_split_seed 777 \
    --sequence_length 20 \
    --use_emotion_rois_only \
    --output_dir output/glm_baseline
```

### Key Arguments

- `--downstream_task`: Task to perform (`emotions`, `contents`, `features`)
- `--input_type`: Movie type (`movieDM` or `movieTP`)
- `--dataset_split_seed`: Random seed for data split (same as SwiFT-IO)
- `--sequence_length`: Length of fMRI sequence
- `--use_emotion_rois_only`: Use only emotion-related ROIs (34 ROIs)
- `--use_cross_validation`: Use cross-validation for Ridge alpha selection
- `--adjust_hrf`: Use HRF-adjusted emotion labels

### Output

The script saves:
- `glm_metrics.json`: Training, validation, and test metrics
- `glm_model.pkl`: Trained GLM models (one per emotion)
- `glm_config.json`: Configuration and metadata

## Components

### 1. `roi_selector.py`

Selects emotion-related ROIs based on neuroimaging literature.

**Selected ROIs (34 total):**
- Amygdala (bilateral)
- Hippocampus (bilateral)
- Accumbens (bilateral)
- Cingulate cortex (anterior, posterior, isthmus)
- Orbitofrontal cortex
- Insula
- Temporal regions (pole, superior, inferior, fusiform)
- Prefrontal regions (superior, middle frontal)

**Usage:**
```python
from baselines import EmotionROISelector

selector = EmotionROISelector()
df_selected = selector.select_rois(df_roi_timeseries)
```

### 2. `glm_baseline.py`

Main GLM baseline implementation.

**Key Methods:**
- `fit(train_subject_dict, emotion_labels)`: Train GLM on training subjects
- `evaluate(subject_dict, emotion_labels, mode)`: Evaluate on val/test sets
- `save(path)` / `load(path)`: Save/load trained models

**Usage:**
```python
from baselines import GLMBaseline

# Initialize
glm = GLMBaseline(
    num_emotions=7,
    sequence_length=20,
    use_cross_validation=True,
    use_emotion_rois_only=True
)

# Train
train_metrics = glm.fit(train_dict, emotion_labels)

# Evaluate
val_metrics = glm.evaluate(val_dict, emotion_labels, mode='valid')
test_metrics = glm.evaluate(test_dict, emotion_labels, mode='test')

# Save
glm.save('output/glm_model.pkl')
```

## Implementation Details

### Data Flow

```
1. Load train/val/test splits from SwiFT-IO data_module
   ↓
2. For each subject:
   - Load ROI timeseries CSV (750 TRs × 110 ROIs)
   - Select emotion-related ROIs (34 ROIs)
   ↓
3. Create training samples:
   - Extract sequences of length 20
   - Each timepoint = one training sample
   - Features: ROI timeseries values
   - Targets: Emotion labels (7 emotions)
   ↓
4. Train Ridge regression (one per emotion):
   - Input: (num_samples, 34 ROIs)
   - Output: (num_samples,) for each emotion
   - Cross-validation for alpha selection
   ↓
5. Evaluate on val/test sets
```

### Metrics

For each emotion (0-6) and overall:
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error
- `r2_score`: R² Score
- `corrcoef`: Pearson Correlation Coefficient

## Comparison with SwiFT-IO

| Aspect | SVR Baseline | GLM Baseline | SwiFT-IO |
|--------|--------------|--------------|----------|
| Input | Full 4D fMRI (96×96×96×30) | ROI timeseries (34 ROIs) | Full 4D fMRI (96×96×96×30) |
| Input Dimensions | ~884k voxels × 30 TRs | 34 ROIs × 30 TRs | 96×96×96×30 |
| Model | SVR (per emotion) | Ridge Regression | Swin4D Transformer + Perceiver |
| Features | Flattened voxels | Hand-crafted (ROI averages) | Learned (hierarchical) |
| Parameters | ~7M per emotion | ~238 per emotion | ~5M total |
| Training Time | Hours | Minutes | Hours |
| Comparison Fairness | ✅ Same input | ⚠️ Reduced input | - |

## Expected Performance

Based on similar studies, GLM baseline typically achieves:
- Test MSE: ~0.3-0.5 (normalized emotions)
- Test Correlation: ~0.3-0.5 per emotion

This provides a strong baseline for comparison with SwiFT-IO's deep learning approach.

## Troubleshooting

### Issue: Subject ROI timeseries not found

**Solution:** Check that subject ID matches the CSV filename format:
```
sub-{SUBJECT_ID}_movieDM_roi_temporal_activity.csv
```

### Issue: Different train/test split than SwiFT-IO

**Solution:** Make sure to use the same `--dataset_split_seed` as your SwiFT-IO experiments (default: 777).

### Issue: Import errors

**Solution:** Run from the `src/` directory:
```bash
cd src/
python train_glm_baseline.py ...
```

## Citation

If you use this baseline in your research, please cite SwiFT-IO and acknowledge the use of GLM baseline for comparison.
