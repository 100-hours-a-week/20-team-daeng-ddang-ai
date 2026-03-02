# app/services/face_service.py
from __future__ import annotations

import uuid
from fastapi import HTTPException

from app.core.config import FACE_MODE
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_mock_adapter import FaceMockAdapter
from app.services.adapters.face_http_adapter import FaceHttpAdapter

from app.services.adapters.face_adapter import FaceAdapter

_adapter_instance = None

# 설정(FACE_MODE)에 따라 적절한 어댑터 인스턴스를 반환 (Singleton 패턴)
# http: 외부 Face Service 서버로 요청 전송
# mock: 테스트용 가짜 응답 반환 (개발 단계용)
def _select_adapter() -> FaceAdapter:
    global _adapter_instance
    if _adapter_instance:
        return _adapter_instance

    if FACE_MODE == "http":
        _adapter_instance = FaceHttpAdapter()

    else:
        # 기본값: Mock 어댑터 (테스트용)
        _adapter_instance = FaceMockAdapter()
    
    return _adapter_instance

# 얼굴 분석 메인 진입점 (Sync)
# 어댑터를 선택하고 분석 요청을 위임함
def analyze_face_sync(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    """기존 동기 호출(호환용)."""
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    response = adapter.analyze(request_id, req)
    
    if not response.video_url:
        response.video_url = req.video_url
    return response


async def analyze_face_async(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    """비동기 호출 표준 경로."""
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    response = await adapter.analyze_async(request_id, req)

    if not response.video_url:
        response.video_url = req.video_url
    return response
