# app/schemas/healthcare_schema.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, HttpUrl


class HealthcareAnalyzeRequest(BaseModel):
    analysis_id: Optional[str] = Field(None, description="분석 요청 식별자 (UUID 등)")
    dog_id: Optional[int] = Field(123, description="반려견 ID (없으면 기본값 123)")
    video_url: HttpUrl = Field(..., description="분석할 동영상 URL")


class HealthcareAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis_id: str
    dog_id: int
    analyze_at: str
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
