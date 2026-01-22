# app/services/adapters/face_mock_adpater.py
from __future__ import annotations

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

        return FaceAnalyzeResponse(
            request_id=request_id,
            predicted_emotion=label,
            confidence=max(probs.get(label, 0.0), 0.0),
            emotion_probabilities=probs,
            debug={
                "mode": "mock",
                "note": "mock response (no model)",
                "input_type": "video" if req.video_url else ("image" if req.image_url else "none"),
                "options": req.options,
            },
        )