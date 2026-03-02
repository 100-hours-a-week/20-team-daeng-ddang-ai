from __future__ import annotations

from fastapi import APIRouter

from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.healthcare_service import analyze_healthcare_sync, analyze_healthcare_async

router = APIRouter(prefix="/api/healthcare", tags=["healthcare"])


@router.post("/analyze", response_model=HealthcareAnalyzeResponse)
async def analyze(req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
    return await analyze_healthcare_async(req)
