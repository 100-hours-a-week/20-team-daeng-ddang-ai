# app/services/adapters/face_adpater.py
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse

# 얼굴 분석 어댑터 인터페이스 (Strategy Pattern)
class FaceAdapter(ABC):
    @abstractmethod
    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        # 동기 분석: 즉시 결과 반환 (Mock, Local, Http 등 구현체에서 처리)
        raise NotImplementedError

    @abstractmethod
    async def analyze_async(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        # 비동기 분석: orchestrator 표준 경로
        raise NotImplementedError
