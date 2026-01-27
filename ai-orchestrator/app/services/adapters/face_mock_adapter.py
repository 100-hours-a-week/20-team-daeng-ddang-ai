# app/services/adapters/face_mock_adpater.py
from __future__ import annotations
import random
import datetime

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_adapter import FaceAdapter

class FaceMockAdapter(FaceAdapter):
    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        forced = (req.options or {}).get("force_emotion")
        label = forced or "relaxed"

        probs = {
            "happy": 0.10,
            "sad": 0.05,
            "relaxed": 0.80,
            "angry": 0.05,
        }
        if forced and forced in probs:
            for k in probs:
                probs[k] = 0.05
            probs[forced] = 0.85

        summary_options = {
            "angry": [
                "강아지가 현재 불만이 있거나 화가 난 상태로 보입니다. (Mock)",
                "지금은 강아지가 예민해 보여요. 주의가 필요합니다. (Mock)",
                "으르렁거리거나 화가 난 표정이 감지되었습니다. (Mock)"
            ],
            "happy": [
                "강아지가 즐겁고 행복해 보입니다! (Mock)",
                "산책이 정말 즐거운가 봐요! 표정이 아주 밝습니다. (Mock)",
                "강아지가 신나 있어요! 꼬리를 흔들고 있을지도 몰라요. (Mock)"
            ],
            "sad": [
                "강아지가 다소 우울하거나 겁을 먹은 것 같아요. (Mock)",
                "혹시 무서운 게 있었나요? 강아지가 위축되어 보입니다. (Mock)",
                "표정이 조금 슬퍼 보입니다. 컨디션을 확인해 주세요. (Mock)"
            ],
            "relaxed": [
                "강아지가 편안하고 평온한 상태입니다. (Mock)",
                "아주 여유로운 표정이네요. 산책을 즐기고 있어요. (Mock)",
                "긴장하지 않고 편안하게 쉬거나 걷고 있는 모습입니다. (Mock)"
            ]
        }
        options = summary_options.get(label, ["알 수 없음"])
        summary = random.choice(options)

        import time
        start_time = time.time()
        
        return FaceAnalyzeResponse(
            analysis_id=req.analysis_id or request_id,
            predicted_emotion=label,
            confidence=max(probs.get(label, 0.0), 0.0),
            summary=summary,
            emotion_probabilities=probs,
            processing={
                "analysis_time_ms": int((time.time() - start_time) * 1000),
                "frames_extracted": 8,
                "frames_face_detected": 6,
                "note": "Mock Adapter"
            }
        )