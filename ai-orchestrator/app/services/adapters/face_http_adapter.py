# app/services/adapters/face_http_adapter.py
from __future__ import annotations

import requests
import datetime

from app.core.config import FACE_SERVICE_URL, FACE_HTTP_TIMEOUT_SECONDS
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_adapter import FaceAdapter

# 외부 HTTP API를 호출하여 얼굴 분석을 수행하는 어댑터
class FaceHttpAdapter(FaceAdapter):
    def __init__(self) -> None:
        self.base_url = FACE_SERVICE_URL

    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        url = f"{self.base_url}/analyze"

        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        payload["request_id"] = request_id

        # 외부 API 호출 (Timeout 설정 포함)
        r = requests.post(url, json=payload, timeout=FACE_HTTP_TIMEOUT_SECONDS)

        r.raise_for_status()
        data = r.json()

        # 외부 API 응답 처리:
        # 1. 스키마가 완벽히 일치하면 바로 변환 (Happy Path)
        # 2. 불일치하면 필드 매핑 시도 (Fallback) - 외부 스키마 변경 대응
        
        # Try to parse directly if upstream matches spec
        try:
            return FaceAnalyzeResponse(**data)
        except Exception:
            pass

        # Fallback mapping if upstream returns inconsistent flat or nested data
        # (과거 스키마나 다른 형태의 응답을 최대한 현재 스키마로 변환)
        predicted = data.get("predicted_emotion") or data.get("emotion") or "unknown"
        conf = data.get("confidence") or data.get("score") or 0.0
        
        # Handle probability map (감정별 확률)
        probs = data.get("emotion_probabilities") or data.get("probs")
        if not probs:
            # Check if nested in 'emotion' dict (old schema style)
            if isinstance(data.get("emotion"), dict):
                probs = data.get("emotion", {}).get("emotion_probabilities")
        probs = probs or {}

        summary = data.get("summary") or "External Analysis"
        debug = data.get("debug")
        
        # Safe float conversion (형변환 안전 장치)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)

        # 최종 응답 객체 생성
        return FaceAnalyzeResponse(
            analysis_id=req.analysis_id or request_id,
            predicted_emotion=str(predicted),
            confidence=conf,
            summary=str(summary),
            emotion_probabilities={str(k): float(v) for k, v in probs.items()} if isinstance(probs, dict) else {},
            processing={
                "note": "http_adapter_fallback",
                "debug": debug
            }
        )