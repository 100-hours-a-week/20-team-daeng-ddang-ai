# app/routers/vetchat_router.py
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse
from app.services.vetchat_service import chat_sync

router = APIRouter(prefix="/internal/ai/vet", tags=["vetchat"])


@router.post("/chat", response_model=VetChatResponse)
def vet_chat(req: VetChatRequest) -> VetChatResponse:
    """
    Backend → Orchestrator 내부 챗봇 상담 엔드포인트 (동기).
    Orchestrator는 chatbot-service에 요청을 위임하고 결과를 반환합니다.
    """
    return chat_sync(req)
