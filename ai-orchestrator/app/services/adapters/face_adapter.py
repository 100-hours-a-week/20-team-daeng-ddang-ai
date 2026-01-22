# app/services/adapters/face_adpater.py
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse

class FaceAdapter(ABC):
    @abstractmethod
    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        """동기 분석: 즉시 결과 반환"""
        raise NotImplementedError