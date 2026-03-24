# app/routers/vetchat_router.py
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse
from app.services.vetchat_service import chat_async

router = APIRouter(prefix="/api/vet", tags=["vetchat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=VetChatResponse, response_model_exclude_none=True)
async def vet_chat(req: VetChatRequest, request: Request) -> VetChatResponse:
    """
    Backend → Orchestrator 내부 챗봇 상담 엔드포인트 (비동기).
    Orchestrator는 chatbot-service에 요청을 위임하고 결과를 반환합니다.
    """
    logger.info(
        "[VETCHAT_RECEIVED] request_id=%s conversation_id=%s dog_id=%s has_image=%s history_len=%s",
        getattr(request.state, "request_id", "-"),
        req.conversation_id,
        req.dog_id,
        bool(req.image_url),
        len(req.history or []),
    )
    return await chat_async(req)
