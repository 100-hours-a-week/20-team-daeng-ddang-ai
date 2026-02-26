# app/services/adapters/vetchat_mock_adapter.py
from __future__ import annotations

import datetime
import uuid

from app.schemas.vetchat_schema import VetChatRequest, VetChatResponse, VetCitation
from app.services.adapters.vetchat_adapter import VetChatAdapter


class VetChatMockAdapter(VetChatAdapter):
    """개발/테스트용 Mock 어댑터 — chatbot-service 없이 더미 응답 반환"""

    def chat(self, request_id: str, req: VetChatRequest) -> VetChatResponse:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        question = req.message.content

        # 반려견 정보 기반 간단한 Mock 답변 생성
        breed = "알수없음"
        age = "알수없음"
        if req.user_context:
            breed = req.user_context.breed or breed
            age = str(req.user_context.dog_age_years or age)

        mock_answer = (
            f"{breed} ({age}살) 보호자님, 안녕하세요! "
            f"'{question}'에 대한 답변입니다. "
            "해당 증상은 여러 원인이 있을 수 있으므로, 정확한 진단을 위해 "
            "가까운 동물병원에 내원하시기를 권장합니다. "
            "(※ 이 답변은 Mock 응답입니다. 실제 서비스에서는 AI 모델이 답변합니다.)"
        )

        mock_citations = [
            VetCitation(
                doc_id="mock_doc_001",
                chunk_id="chunk_001",
                title="수의학 일반 지식 (Mock)",
                score=0.95,
                snippet="반려견 증상에 따른 일반적인 권고사항... (Mock)",
            ),
        ]

        return VetChatResponse(
            conversation_id=req.conversation_id,
            answered_at=now,
            answer=mock_answer,
            citations=mock_citations,
            processing={
                "latency_ms": 42,
                "model_used": "mock_adapter",
                "note": "Mock Adapter — chatbot-service 미연결 상태",
            },
        )
