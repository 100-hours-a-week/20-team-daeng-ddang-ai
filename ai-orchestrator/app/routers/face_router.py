# app/routers/face_router.py
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.face_service import analyze_face_sync

router = APIRouter(prefix="/api/face", tags=["face"])

@router.post("/analyze", response_model=FaceAnalyzeResponse)
def analyze(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    # 얼굴 분석 요청 처리: 동기(Sync) 방식으로 서비스 호출
    # 에러 발생 시 try-except 블록에서 처리하여 적절한 JSON 응답 반환
    try:
        return analyze_face_sync(req)
    except ValueError as e:
        # Map ValueError (e.g., FACE_NOT_DETECTED) to 422
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail={
            "analysis_id": req.analysis_id,
            "status": "failed",
            "error": {
                "code": "FACE_NOT_DETECTED" if "FACE_NOT_DETECTED" in str(e) else "ANALYSIS_FAILED",
                "message": str(e)
            }
        })
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
