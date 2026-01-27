# app/schemas/face_schema.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class FaceAnalyzeRequest(BaseModel):
    analysis_id: str
    video_url: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

class FaceAnalyzeResponse(BaseModel):
    analysis_id: str
    predicted_emotion: str
    confidence: float
    summary: str
    emotion_probabilities: Dict[str, float]

    # Optional debug info (not required by spec but useful)
    processing: Optional[Dict[str, Any]] = None


class FaceErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    debug: Optional[Dict[str, Any]] = None