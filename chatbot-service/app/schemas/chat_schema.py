# app/schemas/chat_schema.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 사용자 메시지 단일 턴
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' 또는 'assistant'")
    content: str = Field(..., description="메시지 내용")


# 반려견 기본 정보
class UserContext(BaseModel):
    dog_age_years: Optional[float] = Field(None, description="반려견 나이(년)")
    dog_weight_kg: Optional[float] = Field(None, description="반려견 체중(kg)")
    breed: Optional[str]           = Field(None, description="견종")


# POST /api/vet/chat 요청 스키마
class ChatRequest(BaseModel):
    dog_id: Optional[int]           = Field(None,  description="반려견 ID")
    conversation_id: str             = Field(...,   description="대화 세션 식별자")
    message: str                     = Field(...,   description="사용자 질문 텍스트")
    image_url: Optional[str]         = Field(None,  description="첨부 이미지 URL (선택)")
    user_context: Optional[UserContext] = Field(None, description="반려견 기본 정보")
    history: List[ChatMessage]       = Field(default_factory=list, description="이전 대화 기록")


# RAG 인용 문서
class Citation(BaseModel):
    doc_id:  str            = Field(...,  description="문서 ID")
    title:   str            = Field(...,  description="문서 제목")
    score:   float          = Field(1.0,  description="유사도 점수")
    snippet: str            = Field(...,  description="관련 발췌문")


# POST /api/vet/chat 응답 스키마
class ChatResponse(BaseModel):
    dog_id:          Optional[int]    = Field(None, description="반려견 ID")
    conversation_id: str              = Field(...,  description="대화 세션 식별자")
    answered_at:     str              = Field(...,  description="응답 생성 시각 (ISO 8601)")
    answer:          str              = Field(...,  description="챗봇 답변 본문")
    citations:       List[Citation]   = Field(default_factory=list, description="RAG 인용 문서 목록")
    processing:      Optional[Dict[str, Any]] = Field(None, description="처리 메타데이터 (지연 시간, 모델명 등)")
    error_code:      Optional[str]    = Field(None, description="에러 코드 (실패 시에만 값 있음)")
