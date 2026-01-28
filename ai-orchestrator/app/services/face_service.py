# app/services/face_service.py
from __future__ import annotations

import uuid
from fastapi import HTTPException

from app.core.config import FACE_MODE
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_mock_adapter import FaceMockAdapter
from app.services.adapters.face_http_adapter import FaceHttpAdapter
from app.services.adapters.face_local_adapter import FaceLocalAdapter
from app.services.adapters.face_adapter import FaceAdapter

_adapter_instance = None

# 설정(FACE_MODE)에 따라 적절한 어댑터 인스턴스를 반환 (Singleton 패턴)
def _select_adapter() -> FaceAdapter:
    global _adapter_instance
    if _adapter_instance:
        return _adapter_instance

    if FACE_MODE == "http":
        _adapter_instance = FaceHttpAdapter()
    elif FACE_MODE == "local":
        _adapter_instance = FaceLocalAdapter()
    else:
        # 기본값: Mock 어댑터 (테스트용)
        _adapter_instance = FaceMockAdapter()
    
    return _adapter_instance

# 얼굴 분석 메인 진입점 (Sync)
# 어댑터를 선택하고 분석 요청을 위임함
def analyze_face_sync(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    return adapter.analyze(request_id, req)
