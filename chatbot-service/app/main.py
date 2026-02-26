# app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.core.config import DEBUG
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import initialize_engine, generate_chat_response

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("chatbot_server")


# 앱 생명주기 관리 (Startup/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chatbot Service 시작: VetChatbotCore 로딩 중...")
    try:
        initialize_engine()
    except Exception as e:
        logger.error(f"엔진 초기화 실패: {e}")
        raise e
    yield
    logger.info("Chatbot Service 종료.")


app = FastAPI(
    title="Vet Chatbot Service",
    description="수의사 AI 챗봇 모델 서버 (RAG + LoRA Qwen 7B)",
    version="1.0.0",
    lifespan=lifespan,
)


# 챗봇 추론 엔드포인트
@app.post("/api/vet/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """사용자 질문(text)과 선택 image_url을 받아 수의사 상담 답변 생성"""
    if not req.message or not req.message.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_INPUT",
                    "message": "질문 내용이 비어 있습니다.",
                    "details": {"field": "message"},
                }
            },
        )
    try:
        return generate_chat_response(req)
    except Exception as e:
        logger.exception(f"Chat failed for conversation_id={req.conversation_id}")
        raise HTTPException(status_code=500, detail=str(e))


# 헬스 체크
@app.get("/health")
def health():
    from app.services.chat_service import get_engine
    return {"status": "ok", "engine_loaded": get_engine() is not None}
