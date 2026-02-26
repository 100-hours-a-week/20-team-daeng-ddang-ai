# app/services/chat_service.py
from __future__ import annotations

import datetime
import time
import logging
from typing import Optional

from app.core.config import BASE_MODEL_ID, ADAPTER_PATH, CHROMA_DB_DIR, DEBUG
from app.schemas.chat_schema import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)

# 싱글턴 - 앱 시작 시 한 번만 초기화
_chatbot_engine = None


def initialize_engine():
    """FastAPI lifespan에서 호출. VetChatbotCore를 메모리에 로드."""
    global _chatbot_engine
    logger.info(f"VetChatbotCore 초기화 시작 (base={BASE_MODEL_ID}, adapter={ADAPTER_PATH})")
    try:
        from scripts.chatbot_core import VetChatbotCore
        _chatbot_engine = VetChatbotCore(
            base_model_id=BASE_MODEL_ID,
            adapter_path=ADAPTER_PATH,
            chroma_db_dir=CHROMA_DB_DIR,
        )
        logger.info("✅ VetChatbotCore 초기화 완료")
    except Exception as e:
        logger.error(f"❌ VetChatbotCore 초기화 실패: {e}")
        raise e


def get_engine():
    return _chatbot_engine


def generate_chat_response(req: ChatRequest) -> ChatResponse:
    """
    VetChatbotCore.generate_answer()를 호출하고 ChatResponse 형태로 반환.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("ChatbotEngine이 초기화되지 않았습니다.")

    start_ms = time.time() * 1000

    # user_context dict 변환
    user_context_dict = {}
    if req.user_context:
        user_context_dict = {
            "dog_age_years": req.user_context.dog_age_years,
            "dog_weight_kg": req.user_context.dog_weight_kg,
            "breed":         req.user_context.breed,
        }

    # history list[dict] 변환
    history_list = [{"role": m.role, "content": m.content} for m in req.history]

    result = engine.generate_answer(
        message=req.message.content,
        user_context=user_context_dict,
        history=history_list,
    )

    elapsed_ms = int(time.time() * 1000 - start_ms)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    citations = [
        Citation(
            doc_id=c.get("doc_id", "unknown"),
            title=c.get("title", "수의학 지식"),
            score=float(c.get("score", 1.0)),
            snippet=c.get("snippet", ""),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        dog_id=req.dog_id,
        conversation_id=req.conversation_id,
        answered_at=now,
        answer=result["answer"],
        citations=citations,
        processing={
            "latency_ms": elapsed_ms,
            "model_used": BASE_MODEL_ID,
        },
    )
