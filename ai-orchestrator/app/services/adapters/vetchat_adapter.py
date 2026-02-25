# app/services/adapters/vetchat_adapter.py
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse


# 챗봇 어댑터 인터페이스 (Strategy Pattern, face/healthcare_adapter와 동일 구조)
class VetChatAdapter(ABC):
    @abstractmethod
    def chat(self, request_id: str, req: VetChatRequest) -> VetChatResponse:
        # 동기 방식으로 chatbot-service 호출 후 응답 반환
        raise NotImplementedError
