# app/services/adapters/healthcare_mock_adapter.py
from __future__ import annotations

import datetime
import uuid

from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.adapters.healthcare_adapter import HealthcareAdapter


class HealthcareMockAdapter(HealthcareAdapter):
    def analyze(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        analysis_id = req.analysis_id or request_id or str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()

        # Lightweight dummy response mimicking analyze_health output
        return HealthcareAnalyzeResponse(
            analysis_id=analysis_id,
            dog_id=int(req.dog_id) if req.dog_id is not None else 123,
            analyze_at=now,
            result={
                "overall_score": 80,
                "overall_risk_level": "low",
                "summary": "모의 응답: 보행이 안정적입니다."
            },
            metrics={
                "gait_rhythm": {"level": "consistent", "score": 85, "description": "리듬이 일정합니다."},
                "gait_balance": {"level": "good", "score": 82, "description": "균형이 좋습니다."},
                "knee_mobility": {"level": "normal", "score": 78, "description": "관절 가동성 정상."},
                "gait_stability": {"level": "stable", "score": 80, "description": "상체 흔들림이 적습니다."},
                "patella_risk_signal": {"level": "low", "score": 90, "description": "슬개골 위험 낮음."}
            },
            artifacts={"keypoint_overlay_video_url": "https://example.com/mock_overlay.mp4"},
            processing={"note": "mock_adapter"},
            error_code=None
        )

    async def analyze_async(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        return self.analyze(request_id, req)
