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
    request_id: str # Keep for internal tracking? User asked for analysis_id in response. Let's keep both or alias. User spec: analysis_id, predicted_emotion, confidence, summary, emotion_probabilities.
    # User's response structure:
    # { analysis_id, predicted_emotion, confidence, summary, emotion_probabilities }
    
    predicted_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    emotion_probabilities: Dict[str, float]
    
    debug: Optional[Dict[str, Any]] = None

class FaceErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    debug: Optional[Dict[str, Any]] = None