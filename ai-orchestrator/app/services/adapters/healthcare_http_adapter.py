# app/services/adapters/healthcare_http_adapter.py
from __future__ import annotations

import requests
import json
import httpx
from app.core.config import HEALTHCARE_SERVICE_URL, HEALTHCARE_HTTP_TIMEOUT_SECONDS
from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse
from app.services.adapters.healthcare_adapter import HealthcareAdapter


class HealthcareHttpAdapter(HealthcareAdapter):
    def __init__(self) -> None:
        self.base_url = HEALTHCARE_SERVICE_URL

    def analyze(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        url = f"{self.base_url}/analyze"
        payload = req.model_dump(mode="json") if hasattr(req, "model_dump") else json.loads(req.json())
        payload["request_id"] = request_id

        r = requests.post(url, json=payload, timeout=HEALTHCARE_HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        return self._build_response(request_id, req, data)

    async def analyze_async(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        url = f"{self.base_url}/analyze"
        payload = req.model_dump(mode="json") if hasattr(req, "model_dump") else json.loads(req.json())
        payload["request_id"] = request_id

        async with httpx.AsyncClient(timeout=HEALTHCARE_HTTP_TIMEOUT_SECONDS) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        return self._build_response(request_id, req, data)

    def _build_response(
        self, request_id: str, req: HealthcareAnalyzeRequest, data: dict
    ) -> HealthcareAnalyzeResponse:
        # Try direct parse first
        try:
            return HealthcareAnalyzeResponse(**data)
        except Exception:
            pass

        # Fallback mapping for unexpected upstream schema variants
        analysis_id = data.get("analysis_id") or req.analysis_id or request_id
        dog_id = data.get("dog_id") or req.dog_id or 123
        analyze_at = data.get("analyze_at") or data.get("analyzed_at") or ""
        result = data.get("result") or {}
        metrics = data.get("metrics") or {}
        artifacts = data.get("artifacts") or {}
        processing = data.get("processing") or data.get("debug") or {}
        error_code = data.get("error_code")

        return HealthcareAnalyzeResponse(
            analysis_id=str(analysis_id),
            dog_id=int(dog_id),
            analyze_at=str(analyze_at),
            result=result,
            metrics=metrics,
            artifacts=artifacts,
            processing=processing,
            error_code=error_code,
        )
