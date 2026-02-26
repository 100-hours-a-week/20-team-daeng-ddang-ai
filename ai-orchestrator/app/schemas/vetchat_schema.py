# app/schemas/vetchat_schema.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 사용자 메시지 단일 턴 (Backend ↔ Orchestrator 내부 계약)
class VetChatMessage(BaseModel):
    role: str    = Field(..., description="'user' 또는 'assistant'")
    content: str = Field(..., description="메시지 내용")


# 반려견 기본 정보
class VetUserContext(BaseModel):
    dog_age_years: Optional[float] = Field(None, description="반려견 나이(년)")
    dog_weight_kg: Optional[float] = Field(None, description="반려견 체중(kg)")
    breed:         Optional[str]   = Field(None, description="견종")


# 추론 옵션
class VetChatOptions(BaseModel):
    use_rag:     bool  = Field(True,  description="RAG 검색 사용 여부")
    max_tokens:  int   = Field(512,   description="최대 생성 토큰 수")
    temperature: float = Field(0.4,   description="생성 온도")


# POST /internal/ai/vet/chat 요청 스키마 (chatbot.md 3-3절 기준)
class VetChatRequest(BaseModel):
    conversation_id: str                   = Field(...,   description="대화 세션 식별자")
    message:         VetChatMessage        = Field(...,   description="이번 턴 사용자 메시지")
    image_url:       Optional[str]         = Field(None,  description="첨부 이미지 URL (선택)")
    history:         List[VetChatMessage]  = Field(default_factory=list, description="이전 대화 기록")
    user_context:    Optional[VetUserContext] = Field(None, description="반려견 기본 정보")
    options:         Optional[VetChatOptions] = Field(None, description="추론 옵션")


# RAG 인용 문서
class VetCitation(BaseModel):
    doc_id:   str   = Field(..., description="문서 ID")
    chunk_id: Optional[str] = Field(None, description="청크 ID")
    title:    Optional[str] = Field(None, description="문서 제목")
    score:    float = Field(1.0, description="유사도 점수")
    snippet:  str   = Field(..., description="관련 발췌문")


# POST /internal/ai/vet/chat 응답 스키마 (chatbot.md 3-3절 기준)
class VetChatResponse(BaseModel):
    conversation_id: str                  = Field(...,  description="대화 세션 식별자")
    answered_at:     str                  = Field(...,  description="응답 생성 시각 (ISO 8601)")
    answer:          str                  = Field(...,  description="챗봇 답변 본문")
    citations:       List[VetCitation]    = Field(default_factory=list, description="RAG 인용 문서 목록")
    processing:      Optional[Dict[str, Any]] = Field(None, description="처리 메타데이터")
    error_code:      Optional[str]        = Field(None, description="에러 코드 (실패 시에만 값 있음)")
