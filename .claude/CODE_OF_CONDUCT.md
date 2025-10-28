# SwiFT-IO Project Rules & Context

## 🎯 프로젝트 개요

**SwiFT-IO (Spatiotemporal Windowed Fourier Transform for fMRI with Interpretability and Optimization)**
- fMRI 데이터로 영화 시청 중 감정 예측
- Transformer 기반 딥러닝 모델 + 전통적 baseline 비교
- 데이터셋: HBN (Healthy Brain Network) movieDM task
- 주요 task: 7가지 감정 regression 또는 classification

## 💬 의사소통 규칙

### ✅ 필수 규칙
1. **한글로 답변**: 사용자와 대화는 항상 한글로 (코드와 기술 용어는 영어 유지)
2. **간결하고 친근하게**: 존댓말 사용, 불필요한 설명 최소화
3. **코드는 영어**: 변수명, 함수명, 주석 모두 영어로 작성

### 예시
```python
# ✅ Good
def train_svc_baseline(X_train, Y_train):
    """Train SVC classifier for emotion prediction."""
    pass

# ❌ Bad
def svc_베이스라인_학습(훈련_데이터, 라벨):
    """SVC 분류기를 학습합니다."""
    pass
```

## 🖥️ 컴퓨팅 환경

### 서버 노드
- **node1**: 6x RTX GPU (딥러닝 학습용)
- **node2**: CPU only (PCA, 데이터 전처리)
- **node3**: CPU only
- **node4**: CPU only (SVC 베이스라인)

### Conda 환경
- **환경명**: `swiftio` (❌ 절대 `swiftv9` 아님!)
- **활성화**: `source /usr/anaconda3/etc/profile.d/conda.sh && conda activate swiftio`

### SLURM 작업 제출 규칙
1. **SVC/SVR 베이스라인**: node2 또는 node4 (CPU만 사용)
2. **LSTM/SwiFT-IO 학습**: node1 (GPU 필요)
3. **시간 제한**: 최소 24시간, 보통 48시간 설정 (중간에 끊기지 않도록)
4. **num_workers**: 0~4 사이 (8 이상은 shared memory 에러 발생)

## 📊 데이터 경로

### fMRI 데이터
```bash
# 현재 사용 중인 전처리 데이터
/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/

# ROI timeseries (FreeSurfer parcellation)
/scratch/HBN/9.2.movieDM_ROI_timeseries/

# SwiFT pretrain weights
/scratch/HBN/9.2.movieDM_SwiFT/
```

### 레이블 파일
- Emotion labels: 데이터 모듈에서 자동 로드
- 7 emotions: Anger, Happy, Fear, Sad, Excited, Positive, Negative

## 🧪 실험 설정

### Classification (현재 작업 중)
- **Sequence length**: 20 TRs
- **Dataset split seed**: 2
- **Reference run**: `mc3r4vhf` (WandB)
- **Task type**: `classification` (7 binary classifiers)

### Regression (기존 완료)
- **Sequence length**: 30 TRs
- **Dataset split seed**: 777
- **Task type**: `regression`
- **성능**: `docs/baselines/baseline_performance_table.md` 참고

## 🛠️ 베이스라인 모델

### 1. SVC/SVR Full Voxels
- **파일**: `src/train_svr_baseline.py`
- **특징**: 전체 복셀 사용 (96³ × seq_length)
- **노드**: node4
- **예상 시간**: 12-16h

### 2. SVC/SVR with PCA
- **파일**: `src/train_svr_with_reduction.py`
- **특징**: PCA 100 components/TR → 2,000 features (seq=20)
- **노드**: node2
- **예상 시간**: 7-9h

### 3. SVC/SVR with ROI
- **파일**: `src/train_svr_with_reduction.py`
- **특징**: FreeSurfer ROI 95개 → ~1,900 features
- **노드**: node4
- **예상 시간**: 5-7h

### 4. LSTM Baseline
- **파일**: `src/train_lstm_baseline.py`
- **특징**: LSTM encoder + classification/regression head
- **노드**: node1 (GPU)
- **예상 시간**: 24-48h

## 📝 작업 흐름 (Workflow)

### 새로운 실험 시작 시
1. **TODO 작성**: 복잡한 작업은 반드시 TodoWrite 사용
2. **코드 검증**: 변경 후 import 테스트 실행
3. **Job 제출**: 한 번에 하나씩, 로그 확인 후 다음 제출
4. **모니터링**: `squeue -u kimbo`, `tail -f logs/xxx.out`

### 에러 발생 시
1. **로그 먼저 확인**: `logs/` 디렉토리
2. **작은 단계로 쪼개기**: 한 번에 하나씩 수정
3. **검증 후 재제출**: 동일한 에러 반복 방지

### Git Commit 규칙
- **언제 커밋**: 사용자가 명시적으로 요청할 때만
- **메시지 스타일**: 간결하고 명확하게 (영어)
- **Co-author 추가**:
  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

## 🚨 주의사항

### ❌ 절대 하지 말 것
1. WandB API key 설정 없이 `--use_wandb` 사용
2. SVC job을 GPU 노드(node1)에 제출
3. `num_workers > 4` 설정 (shared memory 에러)
4. Conda 환경 이름 오타 (`swiftv9` → `swiftio`)
5. 시간 제한 24시간 미만 설정

### ⚠️ 자주 발생하는 에러
1. **Bus error / Shared memory**: → `num_workers=0` 설정
2. **WandB login error**: → `--use_wandb` 플래그 제거
3. **DataLoader worker crash**: → `num_workers` 감소
4. **unrecognized arguments**: → argparse에 `--task_type` 추가 확인

## 📚 문서 위치

- **베이스라인 성능 표**: `docs/baselines/baseline_performance_table.md`
- **연구 노트**: `docs/251027_Research_Journey_Sequential_Investigation.md`
- **Classification README**: `docs/baselines/CLASSIFICATION_BASELINES_README.md`
- **Job 실행 가이드**: `CLASSIFICATION_BASELINE_JOBS_SUMMARY.md`

## 🎓 팁

1. **파일 읽기 전에 Glob 사용**: 특정 파일 찾을 때 효율적
2. **병렬 Tool 호출**: 독립적인 작업은 동시에 실행
3. **로그 실시간 확인**: `tail -f` 대신 주기적으로 `tail -30` 사용
4. **Context 절약**: 큰 파일은 필요한 부분만 읽기

---

**마지막 업데이트**: 2025-10-28
**현재 작업**: Classification baselines (SVC, ROI-SVC, PCA-SVC, LSTM)
**진행 중인 Job**: 63506 (SVC Full), 63507 (ROI-SVC), 63508 (PCA-SVC)
