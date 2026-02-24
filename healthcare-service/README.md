# 🐕 Healthcare Analysis Service (FastAPI)

강아지 보행 영상을 분석해 건강 지표를 산출하고, 오버레이 영상을 S3에 업로드하는 마이크로서비스입니다. `analyze_health.py` 파이프라인을 래핑하여 `ai-orchestrator`와 분리된 서버로 실행할 수 있습니다.

## 기능
- `POST /analyze`: 영상 URL을 받아 보행 분석 수행
- `GET /health`: 헬스 체크
- 분석 결과에 오버레이 영상 URL(S3) 포함, `DEBUG_MODE=true` 시 처리 메타데이터/디버그 정보 포함

## 환경 변수 (.env 예시)
```
PORT=8200
DEBUG_MODE=false

# 모델 다운로드 (HF에서 best.pt 다운로드 시)
HEALTH_MODEL_ID=20-team-daeng-ddang-ai/dog-pose-estimation
HEALTH_MODEL_FILENAME=best.pt
MODEL_CACHE_DIR=models

# Hugging Face (private 모델일 경우 필요)
HF_TOKEN=your_hf_token

# S3 업로드
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-bucket
S3_PREFIX=healthcare
```

## 실행
```bash
cd healthcare-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py  # 기본 포트 8200
```

## 엔드포인트 예시
```bash
curl -X POST "http://localhost:8200/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/dog.mp4",
    "dog_id": 123
  }'
```

## 구조
```
healthcare-service/
├── run.py
├── app/
│   ├── main.py
│   ├── core/config.py
│   ├── schemas/health_schema.py
│   └── services/health_analyzer.py
├── scripts/analyze_health.py   # 핵심 분석 로직 (모델/지표 계산)
├── models/                     # best.pt 다운로드 위치
└── requirements.txt
```
