# app/services/healthcare_service.py
from __future__ import annotations

import uuid
from fastapi import HTTPException

from app.core.config import HEALTHCARE_MODE
from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.adapters.healthcare_adapter import HealthcareAdapter
from app.services.adapters.healthcare_http_adapter import HealthcareHttpAdapter
from app.services.adapters.healthcare_mock_adapter import HealthcareMockAdapter

_adapter_instance: HealthcareAdapter | None = None


def _select_adapter() -> HealthcareAdapter:
    global _adapter_instance
    if _adapter_instance:
        return _adapter_instance

    if HEALTHCARE_MODE == "http":
        _adapter_instance = HealthcareHttpAdapter()
    else:
        _adapter_instance = HealthcareMockAdapter()

    return _adapter_instance


def analyze_healthcare_sync(req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = req.analysis_id or str(uuid.uuid4())
    adapter = _select_adapter()
    response = adapter.analyze(request_id, req)

    return response


async def analyze_healthcare_async(req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
    if not req.video_url:
        raise HTTPException(status_code=422, detail="video_url is required")

    request_id = req.analysis_id or str(uuid.uuid4())
    adapter = _select_adapter()
    return await adapter.analyze_async(request_id, req)
