# app/services/vetchat_service.py
from __future__ import annotations

import uuid
from fastapi import HTTPException

from app.core.config import CHATBOT_MODE
from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse
from app.services.adapters.vetchat_adapter import VetChatAdapter
from app.services.adapters.vetchat_http_adapter import VetChatHttpAdapter
from app.services.adapters.vetchat_mock_adapter import VetChatMockAdapter

_adapter_instance: VetChatAdapter | None = None


def _select_adapter() -> VetChatAdapter:
    """CHATBOT_MODE 설정에 따라 HTTP 또는 Mock 어댑터를 선택 (Singleton 패턴)"""
    global _adapter_instance
    if _adapter_instance:
        return _adapter_instance

    if CHATBOT_MODE == "http":
        _adapter_instance = VetChatHttpAdapter()
    else:
        # 기본값: Mock 어댑터 (개발/테스트 환경)
        _adapter_instance = VetChatMockAdapter()

    return _adapter_instance


def chat_sync(req: VetChatRequest) -> VetChatResponse:
    """챗봇 상담 메인 진입점 (Sync) — 어댑터를 선택하고 챗봇 서비스에 요청 위임"""
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

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    return adapter.chat(request_id, req)
