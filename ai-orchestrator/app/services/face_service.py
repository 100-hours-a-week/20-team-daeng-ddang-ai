# app/services/face_service.py
from __future__ import annotations

import uuid
from fastapi import HTTPException

from app.core.config import FACE_MODE
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_mock_adapter import FaceMockAdapter
from app.services.adapters.face_http_adapter import FaceHttpAdapter
from app.services.adapters.face_adapter import FaceAdapter

def _select_adapter() -> FaceAdapter:
    if FACE_MODE == "http":
        return FaceHttpAdapter()
    return FaceMockAdapter()

def analyze_face_sync(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = str(uuid.uuid4())
    adapter = _select_adapter()
    return adapter.analyze(request_id, req)
