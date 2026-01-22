# app/services/adapters/face_http_adpater.py
from __future__ import annotations

import requests

from app.core.config import FACE_SERVICE_URL, FACE_HTTP_TIMEOUT_SECONDS
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_adapter import FaceAdapter

class FaceHttpAdapter(FaceAdapter):
    def __init__(self) -> None:
        self.base_url = FACE_SERVICE_URL

    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        url = f"{self.base_url}/analyze"

        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()

        payload["request_id"] = request_id

        r = requests.post(url, json=payload, timeout=FACE_HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()

        predicted = data.get("predicted_emotion") or data.get("emotion") or "unknown"
        conf = data.get("confidence") or data.get("score") or 0.0
        probs = data.get("emotion_probabilities") or data.get("probs") or {}
        debug = data.get("debug")

        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)

        return FaceAnalyzeResponse(
            request_id=request_id,
            predicted_emotion=str(predicted),
            confidence=conf,
            emotion_probabilities={str(k): float(v) for k, v in probs.items()} if isinstance(probs, dict) else {},
            debug={"mode": "http", "upstream_url": url, "upstream_debug": debug},
        )