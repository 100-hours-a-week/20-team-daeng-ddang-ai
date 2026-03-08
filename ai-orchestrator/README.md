# AI Orchestrator (FastAPI)

`ai-orchestrator`는 Backend와 AI 마이크로서비스 사이의 통합 진입점입니다.

## 엔드포인트
- `POST /api/missions/judge`: 돌발미션 판정 (Gemini)
- `POST /api/face/analyze`: 표정 분석 위임
- `POST /api/healthcare/analyze`: 보행 분석 위임
- `POST /api/vet/chat`: 수의사 챗봇 위임
- `GET /health`: 헬스 체크

## 핵심 운영 설정
`.env.example` 참고

- 모드/URL
  - `FACE_MODE`, `FACE_SERVICE_URL`
  - `HEALTHCARE_MODE`, `HEALTHCARE_SERVICE_URL`
  - `CHATBOT_MODE`, `CHATBOT_SERVICE_URL`
- 타임아웃
  - `FACE_HTTP_TIMEOUT_SECONDS`
  - `HEALTHCARE_HTTP_TIMEOUT_SECONDS`
  - `CHATBOT_HTTP_TIMEOUT_SECONDS`
- 백프레셔(과부하 제어)
  - `*_MAX_CONCURRENCY`: 기능별 동시 처리 슬롯 수
  - `*_QUEUE_WAIT_SECONDS`: 슬롯 대기 시간(초), 초과 시 429 반환

## 실행
```bash
cd ai-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## 벤치마크
레포 루트에서:
```bash
python3 scripts/performance_test.py -e face healthcare chatbot --count 10 --timeout-seconds 240 -o scripts/bench_results/sync_baseline.json
python3 scripts/performance_test.py -e face healthcare chatbot --count 10 --async --concurrency 4 --timeout-seconds 240 -o scripts/bench_results/async_c4_fair.json
python3 scripts/performance_test.py -e face healthcare chatbot --count 10 --async --concurrency 8 --mix-endpoints --timeout-seconds 240 -o scripts/bench_results/async_c8_mixed.json
```

## 점검 스크립트
레포 루트에서:
```bash
# 최근 async 리팩터링 코드 반영 여부 점검
python3 scripts/verify_async_refactor_smoke.py --skip-smoke

# 오케스트레이터 스모크 테스트
python3 scripts/verify_async_refactor_smoke.py --base-url http://localhost:8000 --timeout-seconds 240 --allow-429
```
