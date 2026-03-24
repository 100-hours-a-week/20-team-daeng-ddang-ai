# app/routers/face_router.py
from __future__ import annotations

import logging
from fastapi import APIRouter

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.face_service import analyze_face_async

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/face", tags=["face"])

@router.post("/analyze", response_model=FaceAnalyzeResponse)
async def analyze(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    logger.info(f"[/api/face/analyze] 요청 수신: {req.model_dump()}")
    return await analyze_face_async(req)
