# SwiFT-IO Ver9 vs Ver11 아키텍처 비교

**날짜**: 2025-11-26
**작성자**: Claude Code 자동화 분석
**코드 기반**: `src/module/models/encoder/`

---

## 요약

SwiFT-IO 프로젝트에는 두 가지 주요 인코더 버전이 존재합니다:
- **Ver9**: 표준 4D Swin Transformer (시공간 통합 attention)
- **Ver11**: 분해된 (1+3)D Swin Transformer (시간/공간 분리 attention + RoPE)

### 핵심 차이점 요약표

| 특성 | Ver9 | Ver11 |
|------|------|-------|
| **Attention 방식** | 4D 통합 윈도우 attention | 분해된 1D(시간) + 3D(공간) attention |
| **Position Embedding** | 학습 가능한 Positional Embedding | RoPE (Rotary Position Embedding) |
| **윈도우 크기** | [6,6,6,4] (공간×시간) | [4,4,4,20] (공간×시간) |
| **시간 범위** | 4 timepoints | 20 timepoints (전체 시퀀스) |
| **공간 범위** | 6×6×6 voxels | 4×4×4 voxels |
| **입력 차원 순서** | (B, C, D, H, W, T) | (B, C, T, D, H, W) |
| **학습 안정화** | 표준 | LayerScale 추가 |

---

## 1. Ver9 아키텍처 상세 분석

### 1.1 파일 위치
```
src/module/models/encoder/swin4d_transformer_ver9.py
```

### 1.2 핵심 구조

```
입력 (B, C, D, H, W, T)
    ↓
PatchEmbed (패치 임베딩)
    ↓
PositionalEmbedding (공간+시간 분리)
    ↓
┌─────────────────────────────────┐
│ BasicLayer (각 스테이지 반복)      │
│   ├─ SwinTransformerBlock4D ×N   │
│   │   ├─ WindowAttention4D       │ ← 4D 통합 attention
│   │   └─ MLP                     │
│   └─ PatchMergingV2 (다운샘플링)  │
└─────────────────────────────────┘
    ↓
LayerNorm
    ↓
출력 (B, C, L)  [L = D×H×W×T 평탄화]
```

### 1.3 핵심 컴포넌트

#### WindowAttention4D (4D 윈도우 기반 Self-Attention)
```python
# 파일: swin4d_transformer_ver9.py:153-227

class WindowAttention4D(nn.Module):
    """4차원 윈도우 기반 multi-head self attention"""

    def __init__(self, dim, num_heads, window_size, ...):
        # 4D 윈도우: [D_w, H_w, W_w, T_w]
        self.window_size = window_size  # 예: [6, 6, 6, 4]
        # 스케일링
        self.scale = head_dim ** -0.5
        # QKV 프로젝션
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
```

**작동 방식**:
- 입력을 4D 윈도우로 분할: `(B, D, H, W, T, C)` → `(B×num_windows, window_size^4, C)`
- 각 윈도우 내에서 **시공간이 통합된** self-attention 수행
- 공간(D,H,W)과 시간(T)이 **동시에** 상호작용

#### PositionalEmbedding (위치 임베딩)
```python
# 파일: swin4d_transformer_ver9.py:658-703

class PositionalEmbedding(nn.Module):
    """절대 위치 임베딩"""

    def __init__(self, dim, patch_num, emb_type='spatio_temporal'):
        # 공간 위치 임베딩 (학습 가능)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
        # 시간 위치 임베딩 (학습 가능 또는 sincos)
        self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))
```

**특징**:
- 공간과 시간 임베딩을 **분리**하여 더함
- 학습 가능한 파라미터 또는 sinusoidal 인코딩 선택 가능

#### PatchMergingV2 (패치 병합/다운샘플링)
```python
# 파일: swin4d_transformer_ver9.py:360-406

class PatchMergingV2(nn.Module):
    """공간 차원만 2×2×2로 병합, 시간 차원 유지"""

    def forward(self, x):
        # 공간 차원 병합: D×H×W → D/2 × H/2 × W/2
        # 시간 차원: 유지
        x = torch.cat([x[:, i::2, j::2, k::2, :, :]
                       for i,j,k in product(range(2),range(2),range(2))], -1)
        # 8배 채널 → c_multiplier배로 축소
        x = self.reduction(self.norm(x))  # Linear(8*dim, 2*dim)
```

### 1.4 윈도우 시프팅 메커니즘

Ver9는 Swin Transformer의 표준 **shifted window attention**을 사용:

```python
# BasicLayer의 블록 구성
for i in range(depth):
    SwinTransformerBlock4D(
        shift_size = self.no_shift if (i % 2 == 0) else self.shift_size
        # 짝수 블록: 시프트 없음
        # 홀수 블록: window_size // 2 만큼 시프트
    )
```

**효과**: 인접 윈도우 간 정보 교환 가능

---

## 2. Ver11 아키텍처 상세 분석

### 2.1 파일 위치
```
src/module/models/encoder/swin4d_transformer_ver11_downstream.py
```

### 2.2 핵심 구조

```
입력 (B, C, D, H, W, T)
    ↓
차원 변환: (B, C, D, H, W, T) → (B, C, T, D, H, W)
    ↓
PatchEmbed (패치 임베딩)
    ↓
┌────────────────────────────────────────┐
│ RoPE4DBasicLayer (각 스테이지 반복)       │
│   ├─ FactorizedSwinTransformerBlock ×N  │
│   │   ├─ TemporalRoPEWindowAttention    │ ← 시간 attention + 1D RoPE
│   │   │   └─ LayerScale                 │
│   │   ├─ SpatialRoPEWindowAttention     │ ← 공간 attention + 3D RoPE
│   │   │   └─ LayerScale                 │
│   │   └─ MLP + LayerScale               │
│   └─ PatchMergingV2 (다운샘플링)          │
└────────────────────────────────────────┘
    ↓
LayerNorm
    ↓
출력 (B, C, L)  [L = D×H×W×T 평탄화]
```

### 2.3 핵심 컴포넌트

#### FactorizedSwinTransformerBlock (분해된 트랜스포머 블록)
```python
# 파일: swin4d_transformer_ver11_downstream.py:583-801

class FactorizedSwinTransformerBlock(nn.Module):
    """시간과 공간 attention을 분리하여 순차적으로 적용"""

    def forward(self, x):
        # 1단계: 시간 Attention
        x = shortcut_t + self.drop_path_t(self.ls_t(temporal_attn_output))

        # 2단계: 공간 Attention
        x = shortcut_s + self.drop_path_s(self.ls_s(spatial_attn_output))

        # 3단계: MLP
        x = x + self.drop_path_mlp(self.ls_mlp(self.mlp(self.norm2(x))))
```

**핵심 설계 철학**:
- 4D attention을 1D(시간) + 3D(공간)으로 **분해**
- 계산 복잡도 감소: O(T×D×H×W)² → O(T²) + O((D×H×W)²)
- 시간적/공간적 특성을 **독립적으로** 학습

#### TemporalRoPEWindowAttention (시간 RoPE Attention)
```python
# 파일: swin4d_transformer_ver11_downstream.py:409-489

class TemporalRoPEWindowAttention(nn.Module):
    """시간 차원에 대한 윈도우 attention + 1D RoPE"""

    def __init__(self, dim, temporal_window_size, num_heads, ...):
        # 시간 윈도우 크기: 예) 20 (전체 시퀀스)
        self.temporal_window_size = temporal_window_size

        # 1D RoPE 초기화
        rope_positions_t = init_1d_positions(temporal_window_size)
        rope_freqs_t = init_1d_rope_freqs(head_dim, num_heads, rope_theta)
        cis_t = compute_1d_rope_cis(rope_freqs_t, rope_positions_t, head_dim)

    def forward(self, x, mask=None):
        # QKV 계산
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # RoPE 회전 적용
        q = apply_rope_rotation(q, device_cis_t)
        k = apply_rope_rotation(k, device_cis_t)

        # Attention 계산
        attn = (q @ k.transpose(-2, -1)) * self.scale
```

**RoPE (Rotary Position Embedding) 작동 원리**:
```python
def apply_rope_rotation(x, freqs_cis):
    """복소수 회전을 통한 위치 인코딩"""
    x_ = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    x_rotated = x_complex * freqs_cis  # 복소수 곱셈 = 회전
    return torch.view_as_real(x_rotated).flatten(3)
```

**RoPE의 장점**:
- **상대적 위치 인코딩**: 절대 위치가 아닌 토큰 간 거리 학습
- **길이 일반화**: 학습 시보다 긴 시퀀스에도 적용 가능
- **효율성**: 추가 파라미터 없이 attention 계산에 통합

#### SpatialRoPEWindowAttention (공간 RoPE Attention)
```python
# 파일: swin4d_transformer_ver11_downstream.py:494-580

class SpatialRoPEWindowAttention(nn.Module):
    """3D 공간(D,H,W)에 대한 윈도우 attention + Joint 3D RoPE"""

    def __init__(self, dim, spatial_window_size, num_heads, ...):
        # 공간 윈도우 크기: 예) (4, 4, 4)
        self.spatial_window_size = spatial_window_size

        # 3D 위치 초기화
        rope_positions_d, rope_positions_h, rope_positions_w = init_3d_positions(
            spatial_window_size[0], spatial_window_size[1], spatial_window_size[2]
        )

        # Joint 3D RoPE 계산
        cis_spatial = compute_joint_3d_rope_cis(
            base_freqs, pos_d, pos_h, pos_w, head_dim
        )
```

**Joint 3D RoPE 설계**:
```python
def compute_joint_3d_rope_cis(base_freqs, pos_d, pos_h, pos_w, head_dim):
    """D, H, W 각 축의 위치를 합산하여 3D 위치 인코딩"""
    angles_d = torch.outer(pos_d, base_freqs)
    angles_h = torch.outer(pos_h, base_freqs)
    angles_w = torch.outer(pos_w, base_freqs)

    # 각 축 각도 합산
    total_angles = angles_d + angles_h + angles_w
    return torch.polar(torch.ones_like(total_angles), total_angles)
```

#### LayerScale (학습 안정화)
```python
# 파일: swin4d_transformer_ver11_downstream.py:191-204

class LayerScale(nn.Module):
    """잔차 연결 전 스케일링으로 학습 안정화"""

    def __init__(self, dim, init_values=1e-5):
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma
```

**LayerScale 효과**:
- 초기값을 매우 작게 설정 (1e-5)
- 잔차 경로의 기여를 점진적으로 증가
- 깊은 네트워크의 학습 안정성 향상

### 2.4 윈도우 분할 방식

```python
# 시간 윈도우 분할
def window_partition_temporal(x, temporal_window_size):
    # x: (B*D*H*W, T_global, C)
    # → (B*D*H*W * T_global//Tw, Tw, C)

# 공간 윈도우 분할
def window_partition_spatial3d(x, spatial_window_size):
    # x: (B*T, D_global, H_global, W_global, C)
    # → (B*T * num_spatial_windows, Dw*Hw*Ww, C)
```

---

## 3. 아키텍처 비교 상세

### 3.1 Attention 메커니즘 비교

#### Ver9: 4D 통합 Attention
```
입력: (B, D, H, W, T, C)
    ↓
4D 윈도우 분할: [6, 6, 6, 4]
    ↓
모든 시공간 위치가 동시에 상호작용
    ↓
Attention 토큰 수: 6×6×6×4 = 864개
```

**장점**:
- 시공간 상호작용을 직접 모델링
- 공간적 패턴과 시간적 변화의 결합 학습

**단점**:
- 높은 계산 복잡도: O(864²) = O(746,496)
- 시간 범위가 제한적 (4 timepoints)

#### Ver11: 분해된 (1+3)D Attention
```
입력: (B, T, D, H, W, C)
    ↓
[1단계] 시간 Attention
├─ (B*D*H*W, T, C) → 시간 윈도우 [20]
└─ Attention 토큰 수: 20개
    ↓
[2단계] 공간 Attention
├─ (B*T, D, H, W, C) → 공간 윈도우 [4,4,4]
└─ Attention 토큰 수: 64개
```

**장점**:
- 낮은 계산 복잡도: O(20²) + O(64²) = O(400 + 4,096) = O(4,496)
- 긴 시간 범위 (20 timepoints = 전체 시퀀스)
- 각 차원의 특성을 독립적으로 학습

**단점**:
- 시공간 상호작용이 간접적
- 분해로 인한 정보 손실 가능성

### 3.2 Position Embedding 비교

| 측면 | Ver9 (학습 가능한 PE) | Ver11 (RoPE) |
|------|----------------------|--------------|
| **유형** | 절대 위치 | 상대 위치 |
| **파라미터** | 학습 가능 | 고정 (sin/cos) |
| **길이 일반화** | 제한적 | 우수 |
| **구현** | 덧셈 | 복소수 회전 |
| **초기화** | 랜덤/truncated normal | 주파수 기반 |

### 3.3 계산 복잡도 비교

**Ver9** (window size = [6,6,6,4]):
- 윈도우 내 토큰 수: 6×6×6×4 = 864
- Self-attention 복잡도: O(864²) ≈ O(746K)

**Ver11** (temporal=20, spatial=[4,4,4]):
- 시간 attention 토큰 수: 20
- 공간 attention 토큰 수: 4×4×4 = 64
- 총 복잡도: O(20²) + O(64²) ≈ O(4.5K)

**복잡도 비율**: Ver9 / Ver11 ≈ **166배** 더 많은 연산

### 3.4 수용 영역 (Receptive Field) 비교

| 차원 | Ver9 | Ver11 | 비교 |
|------|------|-------|------|
| **공간 (D×H×W)** | 6×6×6 = 216 voxels | 4×4×4 = 64 voxels | Ver9 **3.4배** 넓음 |
| **시간 (T)** | 4 timepoints | 20 timepoints | Ver11 **5배** 넓음 |
| **시공간 결합** | 직접 상호작용 | 간접 (순차적) | Ver9 더 강함 |

---

## 4. 가설과 예상 효과

### 4.1 Ver9의 설계 가설

> "fMRI 뇌 신호 분석에서 공간적 맥락이 중요하다"

**예상 효과**:
- 넓은 공간 수용 영역(6×6×6)으로 인접 뇌 영역 간 관계 포착
- 시공간 통합 attention으로 "어디서 무엇이 변하는지" 직접 학습
- 감정의 공간적 분포 패턴에 민감

### 4.2 Ver11의 설계 가설

> "시간적 dynamics가 fMRI 신호 해석에 핵심이다"

**예상 효과**:
- 긴 시간 범위(20 timepoints)로 전체 시퀀스의 temporal pattern 포착
- 분해된 attention으로 시간적/공간적 특성 독립 학습
- RoPE로 다양한 시퀀스 길이에 일반화
- LayerScale로 안정적인 학습

### 4.3 과제별 적합성 가설

| 과제 | 예상 적합 버전 | 이유 |
|------|---------------|------|
| **감정 회귀** | Ver9 | 공간 패턴 중요, 시공간 결합 필요 |
| **성별 분류** | Ver11 | 전체 시퀀스 통합, 안정적 학습 |
| **나이 회귀** | 불확실 | 두 모델 모두 가능 |

---

## 5. 실험적 검증 결과 요약

> 상세 성능 분석: `docs/analysis/2025-11-18-ver9-vs-ver11-performance-comparison-ko.md`

### 5.1 감정 회귀 결과

| 메트릭 | Ver9 | Ver11 | 차이 |
|--------|------|-------|------|
| **평균 Nonzero MAE** | **0.0743** | 0.1561 | Ver9 **52.4% 우수** |
| **승리 감정 수** | **6/7** | 1/7 | Ver9 압도적 우위 |

**분석**:
- Ver9가 감정 **크기 예측**에서 압도적 우위
- Ver11은 **탐지(AUROC)**에서 우수하나 크기 예측 실패
- 가설 검증: 감정 회귀는 공간적 맥락이 더 중요함을 시사

### 5.2 성별 분류 결과

| 메트릭 | Ver9 | Ver11 | 차이 |
|--------|------|-------|------|
| **AUROC** | 0.7980 | **0.8419** | Ver11 **5.5% 우수** |
| **Balanced Accuracy** | **0.7509** | 0.5812 | Ver9 **22.6% 우수** |

**분석**:
- Ver11이 주요 메트릭(AUROC)에서 근소 우위
- Ver11의 학습 안정성이 우수 (100% 성공률)
- 가설 검증: 분류는 시간적 통합이 더 도움됨을 시사

### 5.3 결과 종합

```
┌─────────────────────────────────────────────────────┐
│                  과제별 권장 모델                      │
├─────────────────────────────────────────────────────┤
│  감정 회귀 (Emotion Regression)    →    Ver9 권장    │
│    - 공간적 맥락이 핵심                              │
│    - 크기 예측 정확도 중요                           │
├─────────────────────────────────────────────────────┤
│  성별 분류 (Sex Classification)    →    Ver11 권장   │
│    - 시간적 통합이 유리                              │
│    - AUROC 기준 평가                                │
├─────────────────────────────────────────────────────┤
│  나이 회귀 (Age Regression)        →    추가 실험 필요 │
│    - 데이터 부족                                    │
└─────────────────────────────────────────────────────┘
```

---

## 6. 코드 사용 가이드

### 6.1 Ver9 사용법

```python
from src.module.models.encoder.swin4d_transformer_ver9 import SwinTransformer4D

model = SwinTransformer4D(
    img_size=(96, 96, 96, 20),    # D, H, W, T
    in_chans=1,
    embed_dim=24,
    window_size=[4, 4, 4, 4],     # 공간×시간 윈도우
    first_window_size=[6, 6, 6, 4],
    patch_size=[4, 4, 4, 1],
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    use_flashattn=True,           # Flash Attention 사용
)

# 입력: (B, C, D, H, W, T)
x = torch.randn(2, 1, 96, 96, 96, 20)
output = model(x)  # (B, C, L)
```

### 6.2 Ver11 사용법

```python
from src.module.models.encoder.swin4d_transformer_ver11_downstream import RoPE4DSwinTransformer_Downstream

model = RoPE4DSwinTransformer_Downstream(
    img_size=(96, 96, 96, 20),    # D, H, W, T (내부에서 T,D,H,W로 변환)
    in_chans=1,
    embed_dim=24,
    window_size=[4, 4, 4, 20],    # 공간 [4,4,4], 시간 20
    patch_size=[4, 4, 4, 1],
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    ls_t_init_values=1.0,         # 시간 LayerScale 초기값
    ls_s_init_values=1.0,         # 공간 LayerScale 초기값
    ls_mlp_init_values=1.0,       # MLP LayerScale 초기값
    rope_theta=10000.0,           # RoPE theta 파라미터
    use_flashattn=True,
)

# 입력: (B, C, D, H, W, T) - Ver9와 동일
x = torch.randn(2, 1, 96, 96, 96, 20)
output = model(x)  # (B, C, L)
```

---

## 7. 결론

### 7.1 아키텍처 선택 가이드

| 상황 | 권장 버전 | 근거 |
|------|----------|------|
| 감정 크기 예측 | **Ver9** | 넓은 공간 수용영역, 시공간 결합 |
| 이진 분류 | **Ver11** | 안정적 학습, 시간 통합 |
| 메모리 제약 | **Ver11** | 166배 낮은 attention 복잡도 |
| 긴 시퀀스 | **Ver11** | RoPE 길이 일반화, 넓은 시간 범위 |
| 공간 패턴 중요 | **Ver9** | 6×6×6 수용영역 |

### 7.2 향후 연구 방향

1. **하이브리드 아키텍처**: Ver9의 공간 강점 + Ver11의 시간 강점 결합
2. **적응형 윈도우 크기**: 과제에 따라 동적 조절
3. **Ver11 크기 예측 개선**: 손실 함수 연구, attention 메커니즘 수정
4. **효율성-성능 트레이드오프**: 최적 윈도우 크기 탐색

---

## 참고 자료

- `docs/analysis/2025-11-18-ver9-vs-ver11-performance-comparison-ko.md` - 상세 성능 비교
- `docs/analysis/2025-11-10-perceiver-io-architecture-analysis.md` - Perceiver IO 분석
- `src/module/models/encoder/swin4d_transformer_ver9.py` - Ver9 소스 코드
- `src/module/models/encoder/swin4d_transformer_ver11_downstream.py` - Ver11 소스 코드

---

**문서 끝**
