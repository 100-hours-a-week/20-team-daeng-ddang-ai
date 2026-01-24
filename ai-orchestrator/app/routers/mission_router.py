from __future__ import annotations
from fastapi import APIRouter
from app.schemas.mission_schema import MissionAnalysisData, MissionAnalysisRequest
from app.services.mission_service import analyze_sync, now_iso

router = APIRouter(prefix = "/api/missions", tags = ["mission"])

# 미션 판정 엔드포인트
# 백엔드 서버가 판정을 요청하면 Gemini를 통해 분석 후 결과를 반환
@router.post("/judge", response_model=MissionAnalysisData)
def analyze_missions_judge(
    req: MissionAnalysisRequest, # 요청 본문 데이터 스키마
):
    # 동기 방식으로 비디오 분석 수행 (실제 AI 호출 발생)
    results = analyze_sync(req.missions)
    
    # 분석 결과를 스키마에 맞춰 반환
    return MissionAnalysisData(
        analysis_id = req.analysis_id,
        walk_id = req.walk_id,
        analyzed_at = now_iso(), # 분석 완료 시각
        missions = results,
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