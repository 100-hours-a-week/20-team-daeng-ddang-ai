# app/services/adapters/healthcare_adapter.py
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schemas.healthcare_schema import HealthcareAnalyzeRequest, HealthcareAnalyzeResponse


class HealthcareAdapter(ABC):
    @abstractmethod
    def analyze(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        raise NotImplementedError

    @abstractmethod
    async def analyze_async(self, request_id: str, req: HealthcareAnalyzeRequest) -> HealthcareAnalyzeResponse:
        raise NotImplementedError
