#!/usr/bin/env python3
"""
IG Baseline 문제: 왜 background에 높은 IG 값이 나올까?

문제 코드 (run_igmap_manual_by_subject_group.py:104):
    baseline = torch.zeros_like(input_ts)  # ❌ 문제!

IG (Integrated Gradients)는:
    IG = (input - baseline) * ∫[0→1] ∇f(baseline + α(input - baseline)) dα

즉, baseline에서 input까지의 변화량과 gradient를 곱한 값입니다.
"""

import torch
import numpy as np

print("="*80)
print("WHY BACKGROUND HAS HIGH IG VALUES")
print("="*80)

# 실제 데이터 예시
background_value = -2.9316  # 전처리에서 설정된 background
brain_values = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5, 2.5])  # Brain voxel 예시

# 현재 IG 설정
baseline_current = 0.0

print("\n[현재 설정: baseline = 0]")
print(f"Background voxel: {background_value:.4f}")
print(f"  → Baseline과의 차이: {background_value - baseline_current:.4f}")
print(f"  → 변화량: {abs(background_value - baseline_current):.4f}")

print(f"\nBrain voxels 예시:")
for bv in brain_values:
    diff = bv - baseline_current
    print(f"  {bv:.2f} → baseline과의 차이: {diff:+.2f}, 변화량: {abs(diff):.2f}")

print("\n" + "="*80)
print("PROBLEM ANALYSIS")
print("="*80)

print("\n1. Background voxel의 변화량:")
print(f"   |{background_value:.4f} - {baseline_current:.1f}| = {abs(background_value - baseline_current):.4f}")

print("\n2. Brain voxel의 평균 변화량:")
avg_brain_change = torch.abs(brain_values - baseline_current).mean()
print(f"   평균 = {avg_brain_change:.4f}")

print("\n3. 결과:")
if abs(background_value - baseline_current) > avg_brain_change:
    print(f"   ❌ Background 변화량({abs(background_value - baseline_current):.4f}) > Brain 변화량({avg_brain_change:.4f})")
    print("   → Background가 더 큰 IG 값을 받을 수 있음!")

print("\n" + "="*80)
print("BETTER BASELINE OPTIONS")
print("="*80)

print("\n옵션 1: Background value를 baseline으로 사용")
baseline_option1 = background_value
print(f"  baseline = {baseline_option1:.4f}")
print(f"  Background 변화량: {abs(background_value - baseline_option1):.4f} ✓")
print(f"  Brain 변화량 예시: {abs(brain_values[3] - baseline_option1):.4f}")
print("  → Background의 IG = 0이 됨!")

print("\n옵션 2: 각 voxel의 평균값을 baseline으로 사용")
print("  baseline = subject의 각 voxel 평균 (temporal average)")
print("  → Background와 Brain 모두 평균에서의 변화만 측정")

print("\n옵션 3: Brain voxel 평균을 baseline으로 사용")
brain_mean = brain_values.mean()
print(f"  baseline = {brain_mean:.4f} (brain voxel 평균)")
print(f"  Background 변화량: {abs(background_value - brain_mean):.4f}")
print(f"  Brain 변화량 평균: {torch.abs(brain_values - brain_mean).mean():.4f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\n✅ 가장 좋은 방법: baseline = background_value (-2.9316)")
print("   이유:")
print("   1. Background의 IG가 자동으로 0이 됨")
print("   2. Brain voxel의 변화만 측정")
print("   3. Brain masking의 의도에 부합")
print("\n코드 수정:")
print("   # 현재")
print("   baseline = torch.zeros_like(input_ts)  # ❌")
print("\n   # 수정")
print("   background_value = input_ts.flatten()[0]  # or -2.9316")
print("   baseline = torch.full_like(input_ts, background_value)  # ✅")
print("="*80)
