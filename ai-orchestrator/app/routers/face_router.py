# app/routers/face_router.py
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.face_service import analyze_face_sync

router = APIRouter(prefix="/api/face", tags=["face"])

@router.post("/analyze", response_model=FaceAnalyzeResponse)
def analyze(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    return analyze_face_sync(req)
