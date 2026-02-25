# app/services/adapters/vetchat_http_adapter.py
from __future__ import annotations

import requests

from app.core.config import CHATBOT_SERVICE_URL, CHATBOT_HTTP_TIMEOUT_SECONDS
from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse, VetCitation
from app.services.adapters.vetchat_adapter import VetChatAdapter


class VetChatHttpAdapter(VetChatAdapter):
    """chatbot-service의 POST /api/vet/chat 를 호출하는 HTTP 어댑터"""

    def __init__(self) -> None:
        self.base_url = CHATBOT_SERVICE_URL

    def chat(self, request_id: str, req: VetChatRequest) -> VetChatResponse:
        url = f"{self.base_url}/api/vet/chat"

        # ai-orchestrator → chatbot-service 페이로드 변환
        # (orchestrator 내부 스키마 → chatbot-service 스키마)
        payload = {
            "conversation_id": req.conversation_id,
            "message": req.message.content,
            "image_url": req.image_url,
            "history": [{"role": m.role, "content": m.content} for m in req.history],
            "user_context": req.user_context.model_dump() if req.user_context else None,
        }

        r = requests.post(url, json=payload, timeout=CHATBOT_HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()

        # chatbot-service 응답을 VetChatResponse로 직접 파싱 시도
        try:
            return VetChatResponse(**data)
        except Exception:
            pass

        # Fallback: 필드 직접 매핑
        citations = [
            VetCitation(
                doc_id=c.get("doc_id", "unknown"),
                chunk_id=c.get("chunk_id"),
                title=c.get("title"),
                score=float(c.get("score", 1.0)),
                snippet=c.get("snippet", ""),
            )
            for c in data.get("citations", [])
        ]

        return VetChatResponse(
            conversation_id=data.get("conversation_id", req.conversation_id),
            answered_at=data.get("answered_at", ""),
            answer=data.get("answer", ""),
            citations=citations,
            processing=data.get("processing") or {"note": "http_adapter_fallback"},
            error_code=data.get("error_code"),
        )
