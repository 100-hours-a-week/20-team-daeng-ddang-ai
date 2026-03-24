from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class MultimodalAnalyzeRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="분석할 이미지 URL")


class ClassScore(BaseModel):
    label: str = Field(..., description="클래스 이름")
    score: float = Field(..., description="0~1 확률 점수")


class RouteResult(BaseModel):
    label: str = Field(..., description="eye_closeup | skin_closeup | other")
    raw_label: Optional[str] = Field(None, description="threshold 적용 전 최상위 라벨")
    confidence: float = Field(..., description="선택된 route의 confidence")
    raw_confidence: Optional[float] = Field(None, description="threshold 적용 전 confidence")
    scores: List[ClassScore] = Field(default_factory=list, description="route score 목록")


class EyeDiseaseResult(BaseModel):
    predicted_label: str = Field(..., description="예측 질환명")
    confidence: float = Field(..., description="0~1 확률 점수")
    is_normal: bool = Field(..., description="정상 클래스 여부")
    scores: List[ClassScore] = Field(default_factory=list, description="질환 상위 score 목록")


class SkinDiseaseResult(BaseModel):
    predicted_label: str = Field(..., description="예측 질환명")
    confidence: float = Field(..., description="0~1 확률 점수")
    is_normal: bool = Field(..., description="정상 클래스 여부")
    scores: List[ClassScore] = Field(default_factory=list, description="질환 상위 score 목록")


class MultimodalAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_url: str
    analyze_at: str
    route: RouteResult
    eye_disease: Optional[EyeDiseaseResult] = None
    skin_disease: Optional[SkinDiseaseResult] = None
    processing: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
