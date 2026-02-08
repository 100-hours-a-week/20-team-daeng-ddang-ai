# app/routers/mission_router.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.schemas.mission_schema import (
    MissionAnalysisData, 
    MissionAnalysisRequest, 
    MissionErrorResponse, 
    MissionErrorDetail
)
from app.services.mission_service import analyze_sync, now_iso

router = APIRouter(prefix = "/api/missions", tags = ["mission"])

# 미션 판정 엔드포인트
# 백엔드 서버가 판정을 요청하면 Gemini를 통해 분석 후 결과를 반환합니다.
@router.post(
    "/judge", 
    response_model=MissionAnalysisData,
    responses={
        400: {"model": MissionErrorResponse, "description": "Invalid Request"},
        500: {"model": MissionErrorResponse, "description": "Server Error"}
    }
)
def analyze_missions_judge(
    req: MissionAnalysisRequest, # 요청 본문 데이터 스키마
):
    try:
        # 동기 방식으로 비디오 분석 수행 (실제 AI 호출 발생)
        results = analyze_sync(req.missions)
        
        # 분석 결과를 스키마에 맞춰 반환
        return MissionAnalysisData(
            analysis_id = req.analysis_id,
            walk_id = req.walk_id,
            analyzed_at = now_iso(), # 분석 완료 시각
            missions = results,
        )
    # --- 429 Error Handling Example ---
    # except HTTPException as he:
    #     # 서비스 계층에서 발생시킨 HTTP 예외(429 등)는 그대로 통과시켜 클라이언트(API Gateway/Backend)가 알 수 있게 함
    #     raise he
    # --------------------------------__
    except Exception as e:
        # 전역 예외 처리 -> 500 Error Spec 반환
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