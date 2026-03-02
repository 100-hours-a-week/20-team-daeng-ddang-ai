# app/routers/mission_router.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.schemas.mission_schema import (
    MissionAnalysisData, 
    MissionAnalysisRequest, 
    MissionErrorResponse, 
    MissionErrorDetail
)
from app.services.mission_service import analyze_sync, analyze_async, now_iso

router = APIRouter(prefix = "/api/missions", tags = ["mission"])

# 미션 판정 엔드포인트 – 비동기 버전 (기본)
@router.post(
    "/judge", 
    response_model=MissionAnalysisData,
    responses={
        400: {"model": MissionErrorResponse, "description": "Invalid Request"},
        500: {"model": MissionErrorResponse, "description": "Server Error"}
    }
)
async def analyze_missions_judge(
    req: MissionAnalysisRequest,
):
    try:
        # 비동기 방식으로 모든 미션을 병렬 처리
        results = await analyze_async(req.missions)
        
        return MissionAnalysisData(
            analysis_id = req.analysis_id,
            walk_id = req.walk_id,
            analyzed_at = now_iso(),
            missions = results,
        )
    except Exception as e:
        error_detail = MissionErrorDetail(
            code="ANALYSIS_REQUEST_FAILED",
            message="돌발미션 분석을 수행할 수 없습니다." + (f" ({str(e)})" if str(e) else "")
        )
        raise HTTPException(
            status_code=500,
            detail={
                "analysis_id": req.analysis_id,
                "status": "failed",
                "error": error_detail.model_dump()
            }
        )

# --- 비동기 확장용 (Async Expansion) ---
# from app.services.mission_service import analyze_async
#
# @router.post("/judge/async", response_model=MissionAnalysisData)
# async def analyze_missions_judge_async(
#     req: MissionAnalysisRequest,
# ):
#     results = await analyze_async(req.missions)
#     
#     return MissionAnalysisData(
#         analysis_id = req.analysis_id,
#         walk_id = req.walk_id,
#         analyzed_at = now_iso(),
#         missions = results,
#     )
# -------------------------------------