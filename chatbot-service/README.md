# 🐶 Chatbot Service

수의사 AI 챗봇 추론 서버입니다. RAG(Vector DB) + LoRA-Fine-tuned Qwen 7B 모델을 사용해 반려견 상담 답변을 생성합니다.

## 서비스 역할

- `ai-orchestrator`로부터 내부 요청을 받아 챗봇 추론 수행
- 클라이언트가 직접 이 서버를 호출하지 않음 (Backend → Orchestrator → Chatbot-Service)
- RAG 기반 수의학 지식 검색 + LoRA 어댑터 적용 Qwen 7B 답변 생성

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/vet/chat` | 수의사 상담 답변 생성 |
| `GET`  | `/health` | 헬스 체크 |

## 실행 방법

### 1. 환경 설정

```bash
cd chatbot-service
chmod +x setup.sh && ./setup.sh
```

### 2. 모델 준비

아래 Hugging Face 저장소에서 모델과 Vector DB를 다운로드해 `models/` 디렉토리에 배치:
- **HF Repo**: `huggingface.co/20-team-daeng-ddang-ai/vet-chat`
  - LoRA Adapter → `models/lora-qwen-7b-final/`
  - Vector DB → `models/chroma_db/`

### 3. 서버 실행

```bash
source .venv/bin/activate
python run.py
# 기본 포트: 8300
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace 베이스 모델 ID |
| `ADAPTER_PATH`  | `models/lora-qwen-7b-final`  | LoRA 어댑터 경로 |
| `CHROMA_DB_DIR` | `models/chroma_db`           | Vector DB 디렉토리 |
| `PORT`          | `8300`                       | 서버 포트 |
| `DEBUG`         | `false`                      | 디버그 모드 |

## 폴더 구조

```
chatbot-service/
├── run.py                    # uvicorn 실행 진입점
├── Dockerfile
├── requirements.txt
├── setup.sh                  # 환경 설정 스크립트
├── app/
│   ├── core/
│   │   └── config.py         # 환경 변수 설정
│   ├── schemas/
│   │   └── chat_schema.py    # Pydantic 요청/응답 모델
│   ├── services/
│   │   └── chat_service.py   # VetChatbotCore 싱글턴 래퍼
│   └── main.py               # FastAPI 앱 & 라우트
├── scripts/
│   └── chatbot_core.py       # VetChatbotCore 핵심 로직
└── models/                   # 모델 파일 (별도 다운로드)
    ├── lora-qwen-7b-final/
    └── chroma_db/
```
