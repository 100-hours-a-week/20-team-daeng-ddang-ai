# 댕동여지도 AI Repo

댕동여지도 AI 서비스 모노레포입니다.

## 구성
- `ai-orchestrator`: Backend가 호출하는 통합 API 게이트웨이
- `face-service`: 반려견 표정 분석 마이크로서비스
- `healthcare-service`: 보행 분석/헬스케어 마이크로서비스
- `chatbot-service`: 수의사 상담 챗봇 마이크로서비스
- `vllm-service`: 챗봇 생성 전용 vLLM 추론 서비스
- `scripts`: 성능 측정/점검 스크립트
- `training`, `metrics`: 학습 코드/평가 결과 관리

## 챗봇 배포 구조

- 기본 진입점은 `ai-orchestrator`
- 챗봇 요청은 `ai-orchestrator -> chatbot-service -> vllm-service` 순서로 처리 가능
- `chatbot-service`는 RAG 검색, 리랭크, 프롬프트 조합 담당
- `vllm-service`는 Qwen 계열 모델 생성 담당
- 운영 시에는 `chatbot-service`와 `vllm-service`를 같은 GPU 서버에 두는 구성이 기본 권장안

`chatbot-service`는 환경 변수 `LLM_BACKEND`로 두 가지 모드를 지원합니다.

- `LLM_BACKEND=local`: `chatbot-service`가 직접 Hugging Face 모델 추론
- `LLM_BACKEND=vllm`: 생성만 `vllm-service`에 위임

## 빠른 점검 스크립트
- 비동기 리팩터링 반영 여부 점검:
  - `python3 scripts/verify_async_refactor_smoke.py --skip-smoke`
- 오케스트레이터 스모크 테스트:
  - `python3 scripts/verify_async_refactor_smoke.py --base-url http://localhost:8000 --timeout-seconds 240 --allow-429`
- HF 리비전 상태 확인:
  - `python3 scripts/check_hf_revision_status.py`

## 챗봇 벤치마크

`scripts/performance_test.py`는 결과를 기본적으로 `scripts/bench_results/` 아래에 자동 저장합니다.

예시:

```bash
python3 scripts/performance_test.py -e chatbot --count 20 --label chatbot_local_sync
python3 scripts/performance_test.py -e chatbot --count 20 --async --concurrency 4 --label chatbot_vllm_async
```
