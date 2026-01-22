# app/schemas/face_schema.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class FaceAnalyzeRequest(BaseModel):
    video_url: Optional[str] = None

    options: Dict[str, Any] = Field(default_factory=dict)

class FaceAnalyzeResponse(BaseModel):
    request_id: str

    predicted_emotion: str = "unknown"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    emotion_probabilities: Dict[str, float] = Field(default_factory=dict)

    debug: Optional[Dict[str, Any]] = None

class FaceErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    debug: Optional[Dict[str, Any]] = None