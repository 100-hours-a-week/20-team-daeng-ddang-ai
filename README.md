# 댕동여지도 AI Repo

댕동여지도 AI 서비스 모노레포입니다.

## 구성
- `ai-orchestrator`: Backend가 호출하는 통합 API 게이트웨이
- `face-service`: 반려견 표정 분석 마이크로서비스
- `healthcare-service`: 보행 분석/헬스케어 마이크로서비스
- `chatbot-service`: 수의사 상담 챗봇 마이크로서비스
- `scripts`: 성능 측정/점검 스크립트
- `training`, `metrics`: 학습 코드/평가 결과 관리

## 빠른 점검 스크립트
- 비동기 리팩터링 반영 여부 점검:
  - `python3 scripts/verify_async_refactor_smoke.py --skip-smoke`
- 오케스트레이터 스모크 테스트:
  - `python3 scripts/verify_async_refactor_smoke.py --base-url http://localhost:8000 --timeout-seconds 240 --allow-429`
- HF 리비전 상태 확인:
  - `python3 scripts/check_hf_revision_status.py`
