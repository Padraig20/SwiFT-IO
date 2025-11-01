# Ver11 Downstream Task Integration

**Date**: 2025-11-01
**Author**: Claude (with kimbo)
**Status**: Testing in progress

## Overview

Ver11 (RoPE4DSwinTransformer) 모델을 downstream task (regression/classification)에 사용할 수 있도록 통합하는 작업을 진행했습니다.

## Background

### Ver9 vs Ver11 주요 차이점

**Ver9 (기존 downstream 모델)**:
- Unified 4D window attention
- Learnable position embeddings
- 단일 attention mechanism

**Ver11 (새로운 masked autoencoder)**:
- **Factorized attention**: Temporal + Spatial attention 분리
- **Rotary Position Embedding (RoPE)**: 상대적 위치 정보 encoding
- **LayerScale**: Training stability 향상
- 원래 masked autoencoding을 위해 설계됨 (mask parameter 필요)

### 목표

Ver11의 개선된 architecture (RoPE, Factorized attention, LayerScale)를 활용하면서, downstream task에서 scratch부터 학습할 수 있도록 수정.

## Technical Challenges & Solutions

### Challenge 1: Masked Autoencoder Architecture

**문제**: Ver11은 `forward(x, mask)` signature를 요구하며, masking mechanism이 내장되어 있음.

**해결**:
- `swin4d_transformer_ver11_downstream.py` 생성
- Masking mechanism 완전 제거:
  - `mask_token` 제거
  - `forward(x)` signature로 변경 (mask parameter 불필요)
  - `forward_encoder`에서 모든 patch 사용 (masking 없이)

```python
class RoPE4DSwinTransformer_Downstream(nn.Module):
    """
    Ver11 기반이지만 downstream task용으로 수정:
    - Masking mechanism 제거
    - Decoder 제거
    - Direct forward pass
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B, C, T, D, H, W
        if self.to_float:
            x = x.float()
        z = self.forward_encoder(x)   # B, C, D*H*W*T
        return z
```

### Challenge 2: Window Size Configuration

**문제**: Ver11은 Ver9와 다른 window size order와 strict divisibility requirements 사용.

**Ver9**: `window_size = [T, D, H, W]` (temporal first)
**Ver11**: `window_size = [D, H, W, T]` (spatial first)

**Divisibility requirements**:
```python
# Ver11에서 각 layer의 resolution이 window size로 나누어떨어져야 함
assert spatial_resolution % spatial_window_size == 0
assert temporal_resolution % temporal_window_size == 0
```

**Resolution progression** (patch_size=[4,4,4,1]):
- Initial: [96, 96, 96, 20]
- After patch embedding: [24, 24, 24, 20]
- Layer 1: [24, 24, 24, 20]
- Layer 2: [12, 12, 12, 20]
- Layer 3: [6, 6, 6, 20]
- Layer 4: [3, 3, 3, 20]

**해결**: `window_size = [6, 6, 6, 4]` 선택
- Spatial: 6 divides [24, 12, 6]
- Temporal: 4 divides 20
- Format: [D, H, W, T] (Ver11 convention)

### Challenge 3: Perceiver IO Decoder Dimension Matching

**핵심 이슈**: Encoder output과 decoder input 간의 dimension 매칭이 필요.

#### Perceiver IO Architecture 이해

```
fMRI Input (B, C, D, H, W, T)
     ↓
[Encoder] → Latent Features
     ↓
[Perceiver IO Decoder]
     ├─ Learnable Query: (num_queries, query_channels)
     └─ Cross-Attention: Query × Latent → Output
     ↓
Output (B, T, num_targets)
```

**Learnable Query의 역할**:
- 고정된 차원의 학습 가능한 query vector
- Cross-attention으로 encoder latent에서 필요한 정보 선택적 추출
- Query 개수 = 출력 timepoint 개수 (e.g., 20)

#### Dimension Mismatch Problem

**초기 문제**:
```python
# Ver11 encoder output: [B, C=288, D=3, H=3, W=3, T=20]
# Decoder expects: [B, num_latents, num_latent_channels]

# load_model.py 설정:
SeriesDecoder(
    num_latents=embed_dim,        # 288
    num_latent_channels=dims,     # D*H*W*T = 540
)

# Error: LayerNorm(540) received tensor with last dim = 288
RuntimeError: Given normalized_shape=[540], expected input with shape [*, 540],
             but got input of size[2, 288, 3, 3, 3, 20]
```

**혼란의 원인**:
- Parameter 이름이 직관적이지 않음
- `num_latents=288`이지만 실제로는 channel dimension
- `num_latent_channels=540`이지만 실제로는 spatial-temporal positions

**실제 의미**:
```python
# Encoder output: (B, 288, 540)
#                    ↑    ↑
#                    |    └─ 540 spatial-temporal positions (D×H×W×T)
#                    └────── 288 feature channels

# Decoder interprets as:
# (B, num_latents=288, num_latent_channels=540)
#      ↑              ↑
#      |              └─ 각 latent의 feature 차원
#      └────────────── latent sequence 길이
```

#### 왜 540이 중요한가?

**Seq-to-seq prediction을 위한 spatial-temporal resolution 유지**:
- `540 = D×H×W×T = 3×3×3×20`
- 각 timepoint의 emotion prediction을 위해 spatial-temporal context 필요
- Cross-attention에서 query가 540개 position의 정보를 활용

**해결**:
```python
def forward_encoder(self, x: torch.Tensor):
    """
    Returns: (B, C=288, D*H*W*T=540) matching Ver9 format
    """
    # ... swin blocks ...
    x = self.norm(x)  # B, L_final, C_final

    # Reshape to (B, C, D, H, W, T)
    b, _, c = x.shape
    t, d, h, w = self.last_layer_resolution
    x = x.permute(0, 2, 1).reshape(b, c, d, h, w, t)

    # Flatten spatial-temporal dims to match Ver9
    x = x.flatten(start_dim=2)  # B, C=288, L=540

    return x
```

### Challenge 4: Time Limit Issues

**문제**: 초기 test에서 30분 time limit으로 TIME OUT 발생.

**원인**: Ver11 모델 초기화와 데이터 로딩이 Ver9보다 오래 걸림 (RoPE, LayerScale 등 추가 components).

**해결**: Time limit을 1시간으로 증가.

### Challenge 5: Invalid Arguments

**문제**: Classification test에서 `--adjust_thresh 0.5` unrecognized argument error.

**해결**: Classification script에서 해당 argument 제거 (regression-specific parameter).

## Implementation Details

### Files Created/Modified

1. **Created**: `src/module/models/encoder/swin4d_transformer_ver11_downstream.py`
   - Downstream-only version of Ver11
   - No masking mechanism
   - Output format matches Ver9: `[B, C, D*H*W*T]`

2. **Modified**: `src/module/models/load_model.py`
   - Import Ver11 downstream version
   - Register as "swin4d_ver11"

3. **Created**: `sample_scripts/250602_seq20_ver11/test_regression_ver11.sh`
   - Quick test with fast_dev_run=10
   - Time limit: 1 hour
   - Window size: [6, 6, 6, 4]

4. **Created**: `sample_scripts/250602_seq20_ver11/test_classification_ver11.sh`
   - Quick test with fast_dev_run=10
   - Time limit: 1 hour
   - Removed `--adjust_thresh` argument

5. **Created**: `sample_scripts/250602_seq20_ver11/regression_seq20_ver11_stratified.sh`
   - Full 40-epoch training script

6. **Created**: `sample_scripts/250602_seq20_ver11/classification_seq20_ver11_stratified.sh`
   - Full 40-epoch training script

### Configuration

```bash
# Model configuration
--model swin4d_ver11
--depth 2 2 6 2
--embed_dim 36
--sequence_length 20
--img_size 96 96 96 20
--patch_size 4 4 4 1
--window_size 6 6 6 4
--first_window_size 6 6 6 4

# Decoder configuration
--decoder series_decoder
--num_targets 7  # 7 emotions

# Task-specific
--downstream_task_type regression  # or classification
--num_classes 1  # regression
--num_classes 2  # classification
```

## Current Status

### Testing (2025-11-01)

**Jobs submitted**:
- `63894`: Regression quick test (fast_dev_run=10)
- `63895`: Classification quick test (fast_dev_run=10)

**Status**: Running (13+ minutes elapsed)
- Both jobs passed distributed initialization
- Currently in model initialization/data loading phase
- Logs show successful WandB setup and GPU allocation

**Expected timeline**:
- Model initialization: ~15-20 minutes
- Data loading: ~5-10 minutes
- Fast dev run (10 batches): ~5 minutes
- Total: ~30-35 minutes

### Next Steps

1. **If tests succeed**:
   - Launch full 40-epoch training for both regression and classification
   - Monitor training curves on WandB
   - Compare with Ver9 baseline performance

2. **If tests fail**:
   - Analyze error logs
   - Debug dimension mismatches or architectural issues
   - Iterate on fixes

## Key Learnings

### 1. Perceiver IO Decoder Architecture

Perceiver IO는 dimension matching의 "마법"을 제공:
- **Learnable Query**: 고정된 크기의 학습 가능한 query vector
- **Cross-Attention**: Query가 가변 길이의 encoder output에서 정보 추출
- **Flexible Output**: Query 개수로 출력 sequence 길이 결정

이를 통해 복잡한 spatial-temporal latent (540 dimensions)를 원하는 출력 format (20 timepoints × 7 emotions)으로 변환.

### 2. Parameter Naming Confusion

`num_latents`와 `num_latent_channels`는 직관적이지 않은 naming:
- `num_latents`: 실제로는 channel dimension (feature channels)
- `num_latent_channels`: 실제로는 sequence length (spatial-temporal positions)

이로 인해 초기에 dimension mismatch를 잘못 해석함.

### 3. Ver11 Window Size Convention

Ver11은 Ver9와 다른 convention 사용:
- Ver9: Temporal-first `[T, D, H, W]`
- Ver11: Spatial-first `[D, H, W, T]`

코드에서 이를 확인하지 않으면 subtle bug 발생 가능.

### 4. Seq-to-Seq Prediction Requirement

540 = D×H×W×T를 유지하는 이유:
- 각 timepoint의 emotion을 독립적으로 예측하기 위해 full spatial-temporal context 필요
- Cross-attention에서 query가 이 context를 활용하여 정보 추출

## References

### Model Architecture
- Ver9: `src/module/models/encoder/swin4d_transformer_ver9.py`
- Ver11 (original): `src/module/models/encoder/simmim_swin4d_transformer_ver11.py`
- Ver11 (downstream): `src/module/models/encoder/swin4d_transformer_ver11_downstream.py`

### Decoder
- SeriesDecoder: `src/module/models/decoder/series_decoder.py`
- PerceiverDecoder: `src/module/models/decoder/backend/modules.py`
- Adapters: `src/module/models/decoder/backend/adapter.py`

### Training
- Main script: `src/main.py`
- Lightning module: `src/module/pl_classifier.py`

## Appendix: Error History

### Error 1: Window Size Divisibility
```
AssertionError: Spatial dims must be divisible by window size.
```
**Fix**: Changed window_size from [4,4,4,4] to [6,6,6,4]

### Error 2: Missing mask Parameter
```
TypeError: forward() missing 1 required positional argument: 'mask'
```
**Fix**: Created downstream version without mask requirement

### Error 3: Dimension Mismatch
```
RuntimeError: Given normalized_shape=[540], expected input with shape [*, 540],
             but got input of size[2, 288, 3, 3, 3, 20]
```
**Fix**: Added `flatten(start_dim=2)` to encoder output

### Error 4: Time Limit
```
CANCELLED AT 2025-11-01T21:19:11 DUE TO TIME LIMIT
```
**Fix**: Increased time limit from 30 min to 1 hour

### Error 5: Invalid Argument
```
main.py: error: unrecognized arguments: --adjust_thresh 0.5
```
**Fix**: Removed `--adjust_thresh` from classification script

---

**Last Updated**: 2025-11-01 22:55 KST
**Test Jobs**: 63894 (regression), 63895 (classification) - In Progress
