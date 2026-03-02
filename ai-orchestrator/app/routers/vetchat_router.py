# app/routers/vetchat_router.py
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse
from app.services.vetchat_service import chat_sync, chat_async

router = APIRouter(prefix="/api/vet", tags=["vetchat"])


@router.post("/chat", response_model=VetChatResponse, response_model_exclude_none=True)
async def vet_chat(req: VetChatRequest) -> VetChatResponse:
    """
    Backend → Orchestrator 내부 챗봇 상담 엔드포인트 (비동기).
    Orchestrator는 chatbot-service에 요청을 위임하고 결과를 반환합니다.
    """
    return await chat_async(req)
