# app/routers/face_router.py
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.face_service import analyze_face_async

router = APIRouter(prefix="/api/face", tags=["face"])
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=FaceAnalyzeResponse)
async def analyze(req: FaceAnalyzeRequest, request: Request) -> FaceAnalyzeResponse:
    # 얼굴 분석 요청 처리: 비동기 방식으로 서비스 호출
    logger.info(
        "[FACE_RECEIVED] request_id=%s analysis_id=%s video_url=%s",
        getattr(request.state, "request_id", "-"),
        req.analysis_id,
        req.video_url,
    )
    return await analyze_face_async(req)
