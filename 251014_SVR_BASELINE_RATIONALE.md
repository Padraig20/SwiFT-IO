# SVR Baseline의 타당성: 뇌 디코딩 맥락에서

## 왜 SVR인가?

### 1. 뇌 디코딩에서 SVR의 위치

**뇌 디코딩(Brain Decoding)**은 뇌 활동 패턴으로부터 인지 상태, 감정, 행동 등을 예측하는 작업입니다. 이 맥락에서 SVR(Support Vector Regression)은:

- **고전적 표준(Gold Standard)**: fMRI 디코딩 연구에서 가장 널리 사용되는 머신러닝 방법 중 하나
- **선형 해석 가능성**: 어떤 뇌 영역이 예측에 기여하는지 weight를 통해 해석 가능
- **Small-sample 강건성**: 뇌 영상 데이터는 보통 피험자 수가 제한적인데, SVR은 작은 샘플에서도 비교적 안정적
- **정규화(Regularization)**: C 파라미터를 통한 overfitting 방지

### 2. 왜 Baseline으로 적절한가?

SwiFT-IO와 같은 **시공간 트랜스포머 모델**의 기여를 평가하려면, 시간 정보를 어떻게 활용하는지를 비교할 수 있는 baseline이 필요합니다.

SVR baseline은:
- **특징 공학(Feature Engineering) 기반**: 명시적으로 설계된 feature reduction 사용
- **얕은 모델(Shallow Model)**: 복잡한 비선형 temporal dynamics를 학습하지 않음
- **해석 가능성**: SwiFT-IO가 무엇을 더 잘 포착하는지 명확히 비교 가능

---

## 세 가지 Reduction 방법의 의미

우리는 동일한 SVR을 사용하되, **시간 정보를 다루는 방식**을 달리한 세 가지 버전을 구성했습니다.

### 1. Time-Averaged SVR (시간 평균)

```
입력: (96, 96, 96, 30) → 시간 축 평균 → (96, 96, 96) → flatten → 884,736 features
```

**특징**:
- 시간 정보를 **완전히 제거**
- 30개 프레임을 평균내어 단일 정적 뇌 패턴(static brain pattern)만 사용
- 가장 단순한 baseline

**의미**:
- **시간 정보가 얼마나 중요한지**를 측정하는 하한선(lower bound)
- 만약 SwiFT-IO가 이것보다 크게 좋다면 → 시간적 dynamics를 포착하는 능력이 핵심

**디코딩 관점**:
- 전통적인 "평균 활성화" 접근법 (많은 고전 fMRI 연구에서 사용)
- 질문: "특정 감정 상태와 연관된 평균적인 뇌 활성화 패턴이 있는가?"

---

### 2. PCA-based SVR (차원 축소 후 시간 유지)

```
입력: (96, 96, 96, 30)
각 timepoint에 PCA 적용 → (100, 30)
flatten → 3,000 features (100 components × 30 timepoints)
```

**특징**:
- 시간 정보를 **보존**하되, 공간 차원을 압축
- 각 시점의 주요 활성화 패턴(principal components)을 시간 순서대로 연결
- 중간 수준의 복잡도

**의미**:
- **차원의 저주(curse of dimensionality)**를 완화하면서 시간 정보 유지
- PCA가 뇌 활동의 주요 변동성을 잘 포착한다면, 시간적 순서가 중요할 수 있음
- SwiFT-IO와 비교 → 단순 선형 projection vs learned spatiotemporal representation

**디코딩 관점**:
- "감정은 뇌 활성화의 주요 모드(principal modes)가 시간에 따라 어떻게 변하는가?"
- 시간적 변화는 보지만, 공간 구조의 복잡한 패턴은 선형 축소로만 다룸

---

### 3. ROI-based SVR (관심 영역 기반 시간 유지)

```
입력: ROI timeseries (750 TRs, 109 ROIs)
30-frame sequence 추출 → (30, 109)
flatten → 3,270 features (30 timepoints × 109 ROIs)
```

**특징**:
- 시간 정보를 **보존**, 공간 정보를 **해부학적 ROI**로 축소
- FreeSurfer 뇌 parcellation 기반 (해부학적으로 의미 있는 영역)
- 신경과학적으로 해석 가능

**의미**:
- **해부학적 사전 지식(anatomical prior)** 활용
- ROI = 뇌 영역이 기능적 단위라는 가정
- SwiFT-IO와 비교 → 사전 정의된 영역 vs 데이터 기반 attention

**디코딩 관점**:
- "어떤 뇌 영역들의 시간적 활동 패턴이 감정을 예측하는가?"
- 가장 신경과학적 해석이 용이한 방법
- 많은 fMRI 연구의 표준 접근법

---

## 세 방법의 비교 매트릭스

| 방법          | Feature 차원 | 시간 정보 | 공간 정보 | 해석 가능성 | 신경과학적 타당성 |
|---------------|--------------|-----------|-----------|-------------|-------------------|
| **Time-Avg**  | 884,736      | ✗ 제거    | ✓ 전체    | 중간        | 낮음              |
| **PCA**       | 3,000        | ✓ 보존    | 선형 축소 | 낮음        | 중간              |
| **ROI**       | 3,270        | ✓ 보존    | 해부학적  | 높음        | 높음              |

---

## SwiFT-IO와의 비교 지점

### 1. 시간 정보 활용
- **Time-Avg**: 시간 정보 없음
- **PCA/ROI**: 시간 정보 있지만 순서만 보존 (독립적으로 처리)
- **SwiFT-IO**: Temporal attention으로 timepoint 간 복잡한 관계 학습

### 2. 공간 정보 활용
- **Time-Avg**: 모든 voxel 독립적으로 평균
- **PCA**: 선형 조합으로 주성분 추출
- **ROI**: 해부학적 영역 단위로 평균
- **SwiFT-IO**: Spatial attention으로 voxel 간 관계 학습

### 3. 시공간 상호작용
- **All SVR baselines**: 시간과 공간을 독립적으로 처리
- **SwiFT-IO**: Spatiotemporal transformer로 시공간 joint representation 학습

---

## 예상되는 결과 패턴과 해석

### Case 1: Time-Avg < PCA ≈ ROI < SwiFT-IO
**해석**:
- 시간 정보가 중요함 (Time-Avg가 가장 낮음)
- 단순히 시간 순서를 보존하는 것만으로는 부족 (PCA/ROI 유사)
- SwiFT-IO의 시공간 attention이 핵심 (가장 높음)

### Case 2: Time-Avg < PCA < ROI < SwiFT-IO
**해석**:
- 해부학적 사전 지식이 도움됨 (ROI > PCA)
- SwiFT-IO는 데이터에서 이를 자동으로 학습

### Case 3: Time-Avg ≈ PCA ≈ ROI ≈ SwiFT-IO
**해석**:
- 시간 정보가 이 task에 중요하지 않음
- 정적 패턴만으로도 충분
- (이 경우 SwiFT-IO의 복잡도가 불필요할 수 있음)

---

## 결론

이 세 가지 SVR baseline은:

1. **시간 정보의 중요성** 검증 (Time-Avg vs PCA/ROI)
2. **공간 표현의 방법** 비교 (PCA vs ROI)
3. **SwiFT-IO의 기여** 정량화 (learned spatiotemporal representation의 가치)

를 체계적으로 평가할 수 있게 합니다.

### 왜 이것이 fair comparison인가?

- **동일한 데이터**: 모든 방법이 같은 train/val/test split 사용
- **동일한 모델 클래스**: 모두 SVR (kernel, C, epsilon 동일)
- **통제된 차이점**: Feature engineering 방법만 다름
- **명확한 해석**: 성능 차이가 시간/공간 정보 활용 방식의 차이에서 비롯됨을 명확히 귀속 가능

이를 통해 SwiFT-IO의 복잡한 아키텍처가 단순한 feature engineering보다 **얼마나, 왜** 더 나은지를 설명할 수 있습니다.
