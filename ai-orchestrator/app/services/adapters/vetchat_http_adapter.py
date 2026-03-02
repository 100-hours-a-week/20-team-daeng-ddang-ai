# app/services/adapters/vetchat_http_adapter.py
from __future__ import annotations

import requests
import httpx
import asyncio

from app.core.config import CHATBOT_SERVICE_URL, CHATBOT_HTTP_TIMEOUT_SECONDS, DEBUG
from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse, VetCitation
from app.services.adapters.vetchat_adapter import VetChatAdapter


class VetChatHttpAdapter(VetChatAdapter):
    """chatbot-service의 POST /api/vet/chat 를 호출하는 HTTP 어댑터"""

    def __init__(self) -> None:
        self.base_url = CHATBOT_SERVICE_URL

    def chat(self, request_id: str, req: VetChatRequest) -> VetChatResponse:
        return asyncio.get_event_loop().run_until_complete(self.chat_async(request_id, req))

    async def chat_async(self, request_id: str, req: VetChatRequest) -> VetChatResponse:
        url = f"{self.base_url}/api/vet/chat"

        # ai-orchestrator → chatbot-service 페이로드 변환
        payload = {
            "dog_id": req.dog_id,
            "conversation_id": req.conversation_id,
            "message": {"role": req.message.role, "content": req.message.content},
            "image_url": req.image_url,
            "history": [{"role": m.role, "content": m.content} for m in req.history],
            "user_context": req.user_context.model_dump() if req.user_context else None,
        }

        try:
            async with httpx.AsyncClient(timeout=CHATBOT_HTTP_TIMEOUT_SECONDS) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()

            # chatbot-service 응답을 VetChatResponse로 직접 파싱 시도
            try:
                if "dog_id" not in data:
                    data["dog_id"] = req.dog_id
                
                # DEBUG 모드가 아니면 processing 제거 (Pydantic 객체 생성 전 data에서 가공)
                if not DEBUG:
                    data.pop("processing", None)
                
                return VetChatResponse(**data)
            except Exception:
                pass

            # Fallback: 필드 직접 매핑
            citations = [
                VetCitation(
                    doc_id=c.get("doc_id", "unknown"),
                    title=c.get("title"),
                    score=float(c.get("score", 1.0)),
                    snippet=c.get("snippet", ""),
                )
                for c in data.get("citations", [])
            ]

            return VetChatResponse(
                dog_id=data.get("dog_id", req.dog_id),
                conversation_id=data.get("conversation_id", req.conversation_id),
                answered_at=data.get("answered_at"),
                answer=data.get("answer"),
                citations=citations,
                processing=data.get("processing") if DEBUG else None,
                error_code=data.get("error_code"),
            )
        except Exception as e:
            # 에러 발생 시 상세 정보 포함 (특히 422 Unprocessable Entity 대응)
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_msg += f" | Details: {e.response.text}"
                except:
                    pass
            
            return VetChatResponse(
                dog_id=req.dog_id,
                error_code=f"HTTP_ADAPTER_ERROR: {err_msg}"
            )
