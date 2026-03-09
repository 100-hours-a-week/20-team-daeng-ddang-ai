# app/routers/face_router.py
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.face_service import analyze_face_async

router = APIRouter(prefix="/api/face", tags=["face"])

@router.post("/analyze", response_model=FaceAnalyzeResponse)
async def analyze(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    # 얼굴 분석 요청 처리: 비동기 방식으로 서비스 호출
    return await analyze_face_async(req)
