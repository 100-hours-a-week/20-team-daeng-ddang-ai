from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


class HealthAnalyzeRequest(BaseModel):
    analysis_id: Optional[str] = Field(None, description="요청이 제공한 분석 ID (없으면 서버가 생성)")
    dog_id: Optional[int] = Field(123, description="반려견 ID (없으면 기본값 123)")
    video_url: HttpUrl = Field(..., description="분석할 영상 URL")


class MetricPayload(BaseModel):
    level: Optional[str] = None
    score: Optional[int] = None
    description: Optional[str] = None


class ArtifactsPayload(BaseModel):
    keypoint_overlay_video_url: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class HealthAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis_id: str
    dog_id: int
    analyze_at: str
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, MetricPayload]] = None
    artifacts: Optional[ArtifactsPayload] = None
    processing: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
