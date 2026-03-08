# app/services/vetchat_service.py
from __future__ import annotations

import asyncio
import uuid
import requests
import httpx
from fastapi import HTTPException

from app.core.config import CHATBOT_MODE, CHATBOT_MAX_CONCURRENCY, CHATBOT_QUEUE_WAIT_SECONDS
from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse
from app.services.adapters.vetchat_adapter import VetChatAdapter
from app.services.adapters.vetchat_http_adapter import VetChatHttpAdapter
from app.services.adapters.vetchat_mock_adapter import VetChatMockAdapter

_adapter_instance: VetChatAdapter | None = None
_chatbot_sem = asyncio.Semaphore(max(1, CHATBOT_MAX_CONCURRENCY))


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


async def chat_async(req: VetChatRequest) -> VetChatResponse:
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

    acquired = False
    try:
        await asyncio.wait_for(_chatbot_sem.acquire(), timeout=CHATBOT_QUEUE_WAIT_SECONDS)
        acquired = True
    except TimeoutError:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "CHATBOT_OVERLOADED",
                    "message": "Chatbot is overloaded. Please retry shortly.",
                }
            },
        )

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    try:
        return await adapter.chat_async(request_id, req)
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=detail)
    except requests.HTTPError as e:
        resp = getattr(e, "response", None)
        detail = resp.text if resp is not None else str(e)
        raise HTTPException(status_code=resp.status_code if resp is not None else 502, detail=detail)
    except (httpx.RequestError, requests.RequestException) as e:
        raise HTTPException(status_code=502, detail=f"Chatbot upstream request failed: {e}")
    finally:
        if acquired:
            _chatbot_sem.release()
