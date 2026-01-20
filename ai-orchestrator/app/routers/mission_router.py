from __future__ import annotations

from fastapi import APIRouter

from app.schemas.mission_schema import MissionAnalysisData, MissionAnalysisRequest
from app.services.mission_service import analyze_sync, now_iso

router = APIRouter(prefix = "/api/missions", tags = ["mission"])


@router.post("/judge", response_model=MissionAnalysisData)
def analyze_missions_judge(
    req: MissionAnalysisRequest,
):
    results = analyze_sync(req.missions)
    
    return MissionAnalysisData(
        analysis_id = req.analysis_id,
        walk_id = req.walk_id,
        analyzed_at = now_iso(),
        missions = results,
    )
