# LSTM Baseline 수정 완료 보고서

**날짜**: 2025-10-24
**작업자**: Claude Code + 사용자
**상태**: ✅ 수정 완료

---

## 🔍 발견된 문제

### 문제 1: Metadata 매칭 (예상 문제였으나 실제로는 정상!)
**초기 예상**: Subject ID 형식 불일치로 매칭 실패
**실제 상황**: **정상 작동 중**
- Data subjects: 677개
- Metadata entries: 1,494개
- **매칭 성공**: 590개 (87%)
- **불일치 87개**: 해당 subjects가 metadata에 없음 (정상)

**결론**: Metadata 매칭은 수정 불필요! ✅

---

### 문제 2: Test Step Tensor Shape (실제 문제!)
**문제**: `RuntimeError: a Tensor with 210 elements cannot be converted to Scalar`

**원인 분석**:

```python
# LSTM decoder 이름: 'lstm_series_regression_head'
# 하지만 조건문에서:
if self.hparams.decoder in ['series_decoder', 'lstm_regression_head']:  # ← 누락!
    output = [(logit.cpu().detach(), targets.cpu()) ...]
else:
    output = [(logit.cpu().detach(), targets.cpu().item()) ...]  # ← 여기로 감!
```

**왜 에러가 발생했는가**:
1. LSTM의 decoder 이름: `lstm_series_regression_head`
2. 조건문: `'lstm_regression_head'` 만 체크 (series 빠짐)
3. 결과: `else` 브랜치 실행 → `.item()` 호출
4. LSTM output shape: `(batch=16, time=30, emotions=7)`
5. Loop 후: `targets.shape = (30, 7)` → **210개 원소**
6. `.item()`은 1개 원소만 가능 → **에러!**

---

## ✅ 수정 내용

### 수정 파일: `src/module/pl_classifier.py:551`

**Before**:
```python
if self.hparams.decoder in ['series_decoder', 'lstm_regression_head']:
    output = [(logit.cpu().detach(), targets.cpu()) ...]
```

**After**:
```python
if self.hparams.decoder in ['series_decoder', 'lstm_regression_head', 'lstm_series_regression_head']:
    output = [(logit.cpu().detach(), targets.cpu().detach()) ...]
```

**변경사항**:
1. ✅ `'lstm_series_regression_head'` 추가
2. ✅ `targets.cpu()` → `targets.cpu().detach()` (일관성)

---

## 🎯 문제 해결 상세 설명

### LSTM Output Shape 분석

```python
# LSTM forward pass
input: (batch, channels, H, W, D, T) = (16, 1, 96, 96, 96, 30)
  ↓ Spatial pooling (96³ → 16³)
pooled: (16, 16, 16, 16, 30)
  ↓ Reshape for LSTM
lstm_input: (16, 30, 4096)  # 4096 = 16³
  ↓ LSTM encoder
hidden: (16, 30, 256)  # hidden_dim=256
  ↓ LSTM series regression head
logits: (16, 30, 7)  # 7 emotions
```

### Test Step 처리

```python
# logits.shape = (16, 30, 7)
# target.shape = (16, 30, 7)

for logit, targets in zip(logits, target):
    # logit.shape = (30, 7)    ← 16개 중 1개
    # targets.shape = (30, 7)   ← 16개 중 1개

    # ❌ BEFORE: targets.cpu().item()
    #    → 210개 원소 (30×7)를 scalar로 변환 시도 → 에러!

    # ✅ AFTER: targets.cpu().detach()
    #    → tensor 그대로 유지 → 정상!
```

---

## 📊 테스트 결과 (debug_lstm_data.py)

### Part 1: Metadata Matching
```
✓ Found 677 subject directories
✓ Loaded metadata with 1,494 entries
✓ Direct ID overlap: 590 / 677 subjects (87%)
✓ ID format: Both use 'sub-' prefix
→ 결론: 정상 작동
```

### Part 2: Test Step Simulation
```
Configuration:
  Batch size: 16
  Sequence length: 30
  Num emotions: 7

LSTM logits shape: torch.Size([16, 30, 7])
Target labels shape: torch.Size([16, 30, 7])

Current code (PROBLEMATIC):
  ✗ Code FAILED with error: a Tensor with 210 elements cannot be converted to Scalar

Corrected code:
  ✓ Corrected code WORKS!
```

---

## 🚀 다음 단계

### 1. LSTM Baseline 재실행
이제 다음 명령으로 LSTM을 성공적으로 학습할 수 있습니다:

```bash
sbatch sample_scripts/run_lstm_baseline.slurm
```

**예상 결과**:
- ✅ Metadata 로딩 성공
- ✅ Training 완료 (3 epochs)
- ✅ Validation 통과
- ✅ Test 완료 (에러 없음)

### 2. 성능 예측
Job 62789의 training 결과 기준:
```
Train MSE (by emotion):
  - Emotion 0 (Anger): 1.010
  - Emotion 1 (Happy): 0.348
  - Emotion 2 (Fear): 1.100
  - Emotion 3 (Sad): 2.940
  - Emotion 4 (Excited): 0.0473
  - Emotion 5 (Positive): 0.471
  - Emotion 6 (Negative): 0.875

Valid MSE: 3.199 (improving)
```

**기대 성능**:
- Test MSE: ~3.2 (Valid MSE와 유사)
- SVR-ROI (2.074)보다는 높을 것으로 예상
- 하지만 temporal modeling 효과 입증 가능

---

## 📝 변경 파일 목록

### 1. 수정된 파일
- ✅ `src/module/pl_classifier.py` (1줄 수정)

### 2. 새로 생성된 파일
- ✅ `debug_lstm_data.py` (디버깅 스크립트)
- ✅ `LSTM_BASELINE_ERROR_ANALYSIS.md` (상세 분석)
- ✅ `LSTM_FIX_SUMMARY.md` (이 문서)

---

## 🎓 학습한 내용

### 1. PyTorch Tensor 처리
- `.item()`: **1개 원소**만 있는 tensor → Python scalar 변환
- `.detach()`: Computational graph에서 분리 (gradient 계산 안 함)
- LSTM처럼 **multi-element tensor**는 `.item()` 사용 불가!

### 2. Decoder Type 매칭
- Decoder 이름이 정확히 일치해야 함
- `lstm_regression_head` ≠ `lstm_series_regression_head`
- 유사한 이름도 **명시적으로 모두 추가** 필요

### 3. 디버깅 전략
- ✅ 문제 재현 (시뮬레이션)
- ✅ 각 단계별 shape 확인
- ✅ 조건문 분기 추적
- ✅ 로그에서 실제 값 확인

---

## ✅ 완료 체크리스트

- [x] 문제 원인 파악
- [x] Debug 스크립트 작성 및 실행
- [x] 코드 수정 (pl_classifier.py)
- [x] Test step shape 문제 해결
- [x] Metadata 매칭 확인 (정상)
- [x] 문서화 완료
- [ ] LSTM 재학습 (다음 단계)
- [ ] 결과 검증 및 baseline table 업데이트

---

## 💬 요약

**핵심 문제**:
- LSTM decoder 이름 `lstm_series_regression_head`가 조건문에 없어서 test step에서 `.item()` 호출 → 210개 원소 tensor → 에러

**해결**:
- 조건문에 `'lstm_series_regression_head'` 추가
- `.cpu()` → `.cpu().detach()` 일관성 개선

**Impact**:
- 1줄 수정으로 LSTM baseline 완전히 작동 가능! 🎉
- 33번의 실패 끝에 성공!

---

**작성 완료**: 2025-10-24
**상태**: 수정 완료, 테스트 준비 완료
