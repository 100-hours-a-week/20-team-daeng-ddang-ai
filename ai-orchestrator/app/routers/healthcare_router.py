from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.healthcare_service import analyze_healthcare_async

router = APIRouter(prefix="/api/healthcare", tags=["healthcare"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=HealthcareAnalyzeResponse)
async def analyze(req: HealthcareAnalyzeRequest, request: Request) -> HealthcareAnalyzeResponse:
    logger.info(
        "[HEALTHCARE_RECEIVED] request_id=%s analysis_id=%s dog_id=%s video_url=%s",
        getattr(request.state, "request_id", "-"),
        req.analysis_id,
        req.dog_id,
        req.video_url,
    )
    return await analyze_healthcare_async(req)
