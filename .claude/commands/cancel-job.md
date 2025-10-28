---
description: SLURM job을 취소 (예: /cancel-job 63506)
---

Job ID {{ args }}를 취소하고 확인해주세요.

1. `scancel {{ args }}` 실행
2. `squeue -u kimbo`로 취소 확인
3. 취소된 job의 로그 파일 마지막 부분 확인

한글로 간단히 보고해주세요.
