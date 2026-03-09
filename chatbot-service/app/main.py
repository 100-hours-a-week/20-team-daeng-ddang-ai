# app/main.py
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import DEBUG, MODEL_UPDATE_CHECK_INTERVAL_SECONDS
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import (
    initialize_engine,
    generate_chat_response,
    reload_engine_if_model_updated,
)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("chatbot_server")


async def _background_model_update_checker(stop_event: asyncio.Event):
    interval = MODEL_UPDATE_CHECK_INTERVAL_SECONDS
    logger.info(f"백그라운드 모델 업데이트 체크 시작 (interval={interval}s)")
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        try:
            updated = await run_in_threadpool(reload_engine_if_model_updated)
            if updated:
                logger.info("모델 업데이트가 감지되어 엔진 리로드를 완료했습니다.")
        except Exception:
            logger.exception("백그라운드 모델 업데이트 체크 중 오류 발생")


# 앱 생명주기 관리 (Startup/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    checker_task = None

    logger.info("Chatbot Service 시작: VetChatbotCore 로딩 중...")
    try:
        initialize_engine()
    except Exception as e:
        logger.error(f"엔진 초기화 실패: {e}")
        raise e

    if MODEL_UPDATE_CHECK_INTERVAL_SECONDS > 0:
        checker_task = asyncio.create_task(_background_model_update_checker(stop_event))
    else:
        logger.info("백그라운드 모델 업데이트 체크 비활성화 (interval<=0)")

    yield

    if checker_task:
        stop_event.set()
        await checker_task
    logger.info("Chatbot Service 종료.")


app = FastAPI(
    title="Vet Chatbot Service",
    description="수의사 AI 챗봇 모델 서버 (RAG + LoRA Qwen 7B)",
    version="1.0.0",
    lifespan=lifespan,
)


# 챗봇 추론 엔드포인트
@app.post("/api/vet/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """사용자 질문(text)과 선택 image_url을 받아 수의사 상담 답변 생성"""
    if not req.message or not req.message.content.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_INPUT",
                    "message": "질문 내용이 비어 있습니다.",
                    "details": {"field": "message.content"},
                }
            },
        )
    try:
        return await run_in_threadpool(generate_chat_response, req)
    except Exception as e:
        logger.exception(f"Chat failed for conversation_id={req.conversation_id}")
        raise HTTPException(status_code=500, detail=str(e))


# 헬스 체크
@app.get("/health")
def health():
    from app.services.chat_service import get_engine
    return {"status": "ok", "engine_loaded": get_engine() is not None}
