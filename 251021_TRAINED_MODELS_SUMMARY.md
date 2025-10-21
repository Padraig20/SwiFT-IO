# Trained Models Summary - 2025-10-21

## Overview

Analysis of models in `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri` to identify models trained for more than 20 epochs.

## Models with Epoch > 20

Total: **4 models**

### Detailed Information

| Model ID | Final Epoch | Sequence Length | Input Offset | Stride Between Seq | Stride Within Seq | Batch Size | Task | Input Type |
|----------|-------------|-----------------|--------------|-------------------|-------------------|------------|------|------------|
| **bycj3a1y** | **63** | 30 | **3** | 1 | 1 | 2 | emotions | movieDM |
| **j6yu5erp** | **55** | **20** | **5** | 1 | 1 | 2 | emotions | movieDM |
| **8utbcxtg** | **45** | 30 | **5** | 1 | 1 | 2 | emotions | movieDM |
| **3hv1pnzf** | **41** | 30 | **3** | 1 | 1 | 2 | emotions | movieDM |

### Individual Model Details

#### 1. bycj3a1y (Best - 63 epochs)
- **Final Epoch**: 63
- **Sequence Length**: 30
- **Input Offset**: 3
- **Stride Between Seq**: 1
- **Stride Within Seq**: 1
- **Batch Size**: 2
- **Downstream Task**: emotions
- **Input Type**: movieDM

#### 2. j6yu5erp (55 epochs)
- **Final Epoch**: 55
- **Sequence Length**: 20 ⚠️ (Different from others)
- **Input Offset**: 5 ⚠️ (Different from offset=3 models)
- **Stride Between Seq**: 1
- **Stride Within Seq**: 1
- **Batch Size**: 2
- **Downstream Task**: emotions
- **Input Type**: movieDM

#### 3. 8utbcxtg (45 epochs)
- **Final Epoch**: 45
- **Sequence Length**: 30
- **Input Offset**: 5
- **Stride Between Seq**: 1
- **Stride Within Seq**: 1
- **Batch Size**: 2
- **Downstream Task**: emotions
- **Input Type**: movieDM

#### 4. 3hv1pnzf (41 epochs)
- **Final Epoch**: 41
- **Sequence Length**: 30
- **Input Offset**: 3
- **Stride Between Seq**: 1
- **Stride Within Seq**: 1
- **Batch Size**: 2
- **Downstream Task**: emotions
- **Input Type**: movieDM

## Key Findings

### Common Hyperparameters
- **Task**: All models trained on `emotions` task
- **Input Type**: All use `movieDM`
- **Batch Size**: All use batch size of 2
- **Stride Settings**: All use stride_between_seq=1 and stride_within_seq=1

### Input Offset Analysis
- **Offset = 3**: 2 models (bycj3a1y, 3hv1pnzf)
- **Offset = 5**: 2 models (j6yu5erp, 8utbcxtg)
- **Pattern**: No clear correlation between offset and training success

### Sequence Length Analysis
- **Most Common**: 30 (3 out of 4 models)
- **Alternative**: 20 (1 model: j6yu5erp)
- **Note**: j6yu5erp (seq_len=20, offset=5) vs others with seq_len=30

### Training Duration
- **Longest Training**: bycj3a1y with 63 epochs (seq_len=30, offset=3)
- **Shortest (in this group)**: 3hv1pnzf with 41 epochs (seq_len=30, offset=3)
- **Average Epochs**: 51 epochs

## Models NOT Included (Epoch ≤ 20)

| Model ID | Epochs | Status |
|----------|--------|--------|
| 45t8fcnp | 17 | Incomplete |
| fd865zrm | 17 | Incomplete |
| rwlzwhcz | 15 | Incomplete |
| 5iuyfc40 | 8 | Early stopped |
| k4aeqipr | 7 | Early stopped |
| kxy8bvb8 | 1 | Failed/Early stopped |

## Recommendations

1. **Primary Model for Analysis**: `bycj3a1y` (most epochs, standard seq_len=30, offset=3)
2. **Comparison Studies**:
   - **Sequence Length Impact**: Compare `j6yu5erp` (seq_len=20) vs others (seq_len=30)
   - **Input Offset Impact**: Compare offset=3 models (bycj3a1y, 3hv1pnzf) vs offset=5 models (j6yu5erp, 8utbcxtg)
3. **Ensemble Potential**: Top 3 models (bycj3a1y, j6yu5erp, 8utbcxtg) could be used for ensemble predictions
4. **Hyperparameter Exploration**:
   - Test offset=3 with seq_len=20
   - Test offset=5 with different seq_len values

## File Paths

### Checkpoints Location
```
/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/{model_id}/
├── last.ckpt
└── checkpt-epoch={N}-valid_mse={X}.ckpt
```

### Example Access
```python
# Load best model
checkpoint = torch.load('/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri/bycj3a1y/last.ckpt')
```

---

**Analysis Date**: 2025-10-21
**Total Models Analyzed**: 10
**Models Meeting Criteria**: 4
**Directory**: `/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO/output/moviefmri`
