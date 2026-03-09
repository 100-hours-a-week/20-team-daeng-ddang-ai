# 🐶 Chatbot Service

수의사 AI 챗봇 추론 서버입니다. RAG(Vector DB) + Qwen 계열 모델을 사용해 반려견 상담 답변을 생성합니다. 생성 백엔드는 로컬 Hugging Face 또는 별도 `vLLM` 서버를 사용할 수 있습니다.

## 서비스 역할

- `ai-orchestrator`로부터 내부 요청을 받아 챗봇 추론 수행
- 클라이언트가 직접 이 서버를 호출하지 않음 (Backend → Orchestrator → Chatbot-Service)
- RAG 기반 수의학 지식 검색 + 생성 백엔드(local/vLLM) 호출

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

아래 Hugging Face 저장소에서 모델과 Vector DB를 다운로드해 `models/` 디렉토리에 배치하거나, 서버 시작 시 자동 다운로드를 사용합니다.
- **HF Repo**: `huggingface.co/20-team-daeng-ddang-ai/vet-chat`
  - LoRA Adapter → `models/Qwen2.5-7B/7B-LoRA/`
  - Vector DB → `models/chroma_db/`

`LLM_BACKEND=vllm`일 때는 `chatbot-service`가 `vLLM`으로 생성 요청을 보내므로, 이 서비스는 최소한 `Vector DB`만 로컬에 있으면 됩니다.

### 3. 서버 실행

```bash
source .venv/bin/activate
python run.py
# 기본 포트: 8300
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM_BACKEND` | `local` | 생성 백엔드 (`local` 또는 `vllm`) |
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace 베이스 모델 ID |
| `ADAPTER_PATH`  | `models/Qwen2.5-7B/7B-LoRA`  | LoRA 어댑터 경로 |
| `CHROMA_DB_DIR` | `models/chroma_db`           | Vector DB 디렉토리 |
| `VLLM_BASE_URL` | `http://localhost:8400` | vLLM OpenAI-compatible 서버 주소 |
| `VLLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | vLLM에 요청할 모델 이름 |
| `VLLM_API_KEY` | 없음 | vLLM 인증 토큰 (선택) |
| `VLLM_HTTP_TIMEOUT_SECONDS` | `120` | vLLM 요청 타임아웃 |
| `EMBEDDING_MODEL_ID` | `jhgan/ko-sroberta-multitask` | 질의 임베딩 모델 ID |
| `EMBEDDING_NORMALIZE` | `true` | 임베딩 L2 정규화 여부 |
| `RAG_RETRIEVAL_K` | `5` | 1차 검색 문서 수 |
| `RAG_FINAL_TOP_K` | `3` | 리랭크 후 최종 컨텍스트 문서 수 |
| `RAG_RERANK_ENABLED` | `true` | CrossEncoder 리랭크 사용 여부 |
| `RERANKER_MODEL_ID` | `BAAI/bge-reranker-v2-m3` | 리랭커 모델 ID |
| `GEN_TEMPERATURE` | `0.1` | 생성 다양성 |
| `GEN_TOP_P` | `0.9` | nucleus sampling 비율 |
| `GEN_MAX_NEW_TOKENS` | `384` | 최대 생성 토큰 |
| `GEN_REPETITION_PENALTY` | `1.08` | 반복 억제 페널티 |
| `PORT`          | `8300`                       | 서버 포트 |
| `DEBUG`         | `false`                      | 디버그 모드 |
| `CHECK_MODEL_UPDATE_ON_START` | `true` | HF 최신 리비전(sha) 비교 활성화 (시작/백그라운드 체크 공통) |
| `FORCE_REFRESH_MODELS` | `false` | `true`면 체크 주기마다 자산 재다운로드 강제 |
| `MODEL_REVISION_FILE` | `models/.vet_chat_revision` | 마지막 적용 HF 리비전 sha 저장 파일 |
| `MODEL_UPDATE_CHECK_INTERVAL_SECONDS` | `86400` | 백그라운드 모델 업데이트 체크 주기(초, `<=0`이면 비활성화) |
| `HUGGING_FACE_HUB_TOKEN` | 없음 | HF 자산 다운로드/리비전 조회 토큰 |

`EMBEDDING_MODEL_ID` 또는 `EMBEDDING_NORMALIZE`를 변경하면 기존 `models/chroma_db` 인덱스와 벡터 공간이 달라질 수 있습니다. 이 경우 인덱스를 동일 설정으로 재구축하는 것을 권장합니다.

## 모델 업데이트 동작
- 시작 시 자산 존재 여부와 HF revision을 확인해 필요한 경우만 다운로드합니다.
- `FORCE_REFRESH_MODELS=true`면 주기마다 강제 갱신합니다.
- 첫 실행 시 `sentence-transformers`, reranker 모델이 별도 캐시에 다운로드될 수 있습니다.
- `LLM_BACKEND=vllm`일 때는 챗봇 서비스가 LoRA 가중치를 직접 로드하지 않습니다.

## vLLM 연동

- `chatbot-service`는 RAG 검색, 리랭크, 프롬프트 조합을 담당합니다.
- 실제 텍스트 생성은 `vllm-service`가 담당할 수 있습니다.
- 운영 시에는 `chatbot-service`와 `vllm-service`를 같은 GPU 서버에 두고 내부 주소로 연결하는 구성이 가장 단순합니다.

## Docker 운영 메모
- 컨테이너는 non-root 사용자로 실행됩니다.
- `HEALTHCHECK`가 `/health`를 주기적으로 확인합니다.

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
│   │   ├── chat_service.py   # VetChatbotCore 싱글턴 래퍼
│   │   └── generation/       # local/vLLM 생성 백엔드
│   └── main.py               # FastAPI 앱 & 라우트
├── scripts/
│   └── chatbot_core.py       # RAG + generation backend 연동 로직
└── models/                   # 모델 파일 (별도 다운로드)
    ├── Qwen2.5-7B/7B-LoRA/
    └── chroma_db/
```
