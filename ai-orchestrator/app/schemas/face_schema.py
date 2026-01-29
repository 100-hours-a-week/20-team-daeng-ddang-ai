# app/schemas/face_schema.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# 얼굴 분석 요청 스키마
class FaceAnalyzeRequest(BaseModel):
    analysis_id: str = Field(..., description="분석 요청 식별자 (UUID 등)")
    video_url: Optional[str] = Field(None, description="분석할 동영상 URL")
    options: Dict[str, Any] = Field(default_factory=dict, description="추가 옵션 (예: 디버그 모드)")

# 얼굴 분석 결과 응답 스키마 (Unified)
class FaceAnalyzeResponse(BaseModel):
    analysis_id: str = Field(..., description="분석 요청 식별자")
    # 성공 시 필수, 실패 시 None
    predicted_emotion: Optional[str] = Field(None, description="예측된 최종 감정 (happy, sad, angry, relaxed)")
    confidence: Optional[float] = Field(None, description="최종 감정의 신뢰도 (0.0 ~ 1.0)")
    summary: Optional[str] = Field(None, description="감정 분석 결과에 대한 한 줄 요약/나레이션")
    emotion_probabilities: Optional[Dict[str, float]] = Field(None, description="각 감정별 확률 분포")
    video_url: Optional[str] = Field(None, description="분석된 동영상 URL")
    processing: Optional[Dict[str, Any]] = Field(None, description="처리 시간, 프레임 수 등 메타데이터")
    
    # 실패 시 값 있음 (예: "FACE_NOT_DETECTED"), 성공 시 None
    error_code: Optional[str] = Field(None, description="에러 코드 (실패 시에만 값 있음)")


# 얼굴 분석 실패 시 반환되는 에러 스키마
class FaceErrorResponse(BaseModel):
    request_id: str = Field(..., description="요청 식별자")
    error_code: str = Field(..., description="에러 코드 (예: FACE_NOT_DETECTED)")
    message: str = Field(..., description="에러 상세 메시지")
    # debug: Optional[Dict[str, Any]] = Field(None, description="디버그용 추가 정보")