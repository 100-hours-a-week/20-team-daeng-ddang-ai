from __future__ import annotations

import logging
from fastapi import APIRouter

from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.healthcare_service import analyze_healthcare_async

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/healthcare", tags=["healthcare"])


@router.post("/analyze", response_model=HealthcareAnalyzeResponse)
async def analyze(req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
    logger.info(f"[/api/healthcare/analyze] 요청 수신: {req.model_dump()}")
    return await analyze_healthcare_async(req)
