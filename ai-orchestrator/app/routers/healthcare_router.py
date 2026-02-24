from __future__ import annotations

from fastapi import APIRouter

from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.healthcare_service import analyze_healthcare_sync

router = APIRouter(prefix="/api/healthcare", tags=["healthcare"])


@router.post("/analyze", response_model=HealthcareAnalyzeResponse)
def analyze(req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
    return analyze_healthcare_sync(req)
