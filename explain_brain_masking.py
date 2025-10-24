"""
Brain Masking in SwiFT-IO: What happens vs. What could happen

CURRENT IMPLEMENTATION (모델이 background를 구분하지 않음):
"""

# 1. Input data
# Brain voxels: 다양한 z-score 값 (-2.0, 0.5, 1.2, ...)
# Background voxels: 모두 -2.9316

# 2. Swin Transformer attention
# Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
#
# Background voxel도 Q, K, V 계산에 포함됨!
# 예시:
#   Brain voxel A attends to: [Brain B: 0.3, Brain C: 0.4, Background D: 0.2, Background E: 0.1]
#   → Background voxel이 attention weight 0.3을 받음
#
# 문제:
#   - Background는 정보가 없는데 attention을 받음
#   - 71.4%의 voxel이 불필요하게 attention 계산에 참여
#   - 계산 비용 증가

# 3. Why does the model do this?
#
# 가능한 이유:
# a) 구현의 단순성
#    - Swin Transformer는 window 기반이라서 irregular masking이 복잡함
#
# b) Background가 모두 같은 값이므로 영향이 적다고 판단
#    - 모든 background voxel = -2.9316
#    - Self-attention에서 같은 값들은 서로 비슷한 attention을 받음
#    - 따라서 실제 학습에 미치는 영향이 제한적일 수 있음
#
# c) Padding을 위해 어차피 필요
#    - (81,95,81) → (96,96,96) padding 필요
#    - Padding도 어차피 background value 사용

"""
ALTERNATIVE IMPLEMENTATION (background를 제외할 수 있다면):

방법 1: Attention mask 사용
  - Background voxel의 attention weight를 강제로 0으로 만듦
  - QK^T에 -inf를 더해서 softmax 후 0이 되도록

방법 2: Sparse attention
  - Brain voxel끼리만 attention 계산
  - 계산량: O(N^2) → O(brain_voxels^2)
  - 71.4% 감소 → 약 5배 빠름!

방법 3: Preprocessing에서 brain voxel만 추출
  - 3D volume 대신 1D list of brain voxels
  - Transformer에 직접 입력
  - 공간 정보 손실 가능성
"""

print(__doc__)

# 현재 구현 시뮬레이션
import torch
import torch.nn.functional as F

# 예시 데이터
batch_size = 1
d, h, w, t = 8, 8, 8, 1  # 작은 예시
embed_dim = 64

# Brain mask (71.4% background)
brain_mask = torch.rand(batch_size, d, h, w, t) > 0.714

print("\n" + "="*80)
print("SIMULATION")
print("="*80)
print(f"Total voxels: {d*h*w*t}")
print(f"Brain voxels: {brain_mask.sum().item()}")
print(f"Background voxels: {(~brain_mask).sum().item()}")
print(f"Background ratio: {(~brain_mask).sum().item() / (d*h*w*t) * 100:.1f}%")

# 현재 방식: 모든 voxel 처리
print(f"\nCurrent: Attention on all {d*h*w*t} voxels")
print(f"  Computation: O({d*h*w*t}^2) = O({(d*h*w*t)**2})")

# 개선 방식: Brain voxel만 처리
brain_voxel_count = brain_mask.sum().item()
print(f"\nImproved: Attention on only {brain_voxel_count} brain voxels")
print(f"  Computation: O({brain_voxel_count}^2) = O({brain_voxel_count**2})")
print(f"  Speedup: {((d*h*w*t)**2) / (brain_voxel_count**2 + 1):.1f}x faster")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("Brain masking 정보는 데이터에 '유지'되지만,")
print("모델이 이를 '활용'하지 않아서 계산 비용이 불필요하게 높습니다.")
print("="*80)
