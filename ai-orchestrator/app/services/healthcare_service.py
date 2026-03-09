# app/services/healthcare_service.py
from __future__ import annotations

import asyncio
import uuid
from fastapi import HTTPException

from app.core.config import (
    HEALTHCARE_MODE,
    HEALTHCARE_MAX_CONCURRENCY,
    HEALTHCARE_QUEUE_WAIT_SECONDS,
)
from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.adapters.healthcare_adapter import HealthcareAdapter
from app.services.adapters.healthcare_http_adapter import HealthcareHttpAdapter
from app.services.adapters.healthcare_mock_adapter import HealthcareMockAdapter

_adapter_instance: HealthcareAdapter | None = None
_healthcare_sem = asyncio.Semaphore(max(1, HEALTHCARE_MAX_CONCURRENCY))


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

    acquired = False
    try:
        await asyncio.wait_for(_healthcare_sem.acquire(), timeout=HEALTHCARE_QUEUE_WAIT_SECONDS)
        acquired = True
    except TimeoutError:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "HEALTHCARE_OVERLOADED",
                    "message": "Healthcare analysis is overloaded. Please retry shortly.",
                }
            },
        )

    request_id = req.analysis_id or str(uuid.uuid4())
    adapter = _select_adapter()
    try:
        return await adapter.analyze_async(request_id, req)
    finally:
        if acquired:
            _healthcare_sem.release()
