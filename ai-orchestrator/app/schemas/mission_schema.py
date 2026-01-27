# app/schemas/mission_schema.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# 공통 API 응답 구조
class ApiResponse(BaseModel):
    message: str
    data: Optional[dict] = None
    errorCode: Optional[str] = None

# 지원하는 미션 종류 (SIT: 앉아, DOWN: 엎드려, PAW: 손, TURN: 돌아, JUMP: 점프)
class MissionType(str, Enum):
    SIT = "SIT"
    DOWN = "DOWN"
    PAW = "PAW"
    TURN = "TURN"
    JUMP = "JUMP"

# 미션 상세 수행 상태 (현재 코드에서는 사용이 적음)
class MissionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"

# 개별 미션 분석 요청 데이터
class MissionInput(BaseModel):
    mission_id: int      # 미션 고유 ID
    mission_type: MissionType  # 수행해야 할 미션 종류
    video_url: str       # 분석할 비디오 URL (S3 등)

# 전체 미션 분석 요청 (메인 요청 본문)
class MissionAnalysisRequest(BaseModel):
    analysis_id: str     # 전체 분석 요청 ID
    walk_id: int         # 산책 ID
    missions: List[MissionInput] = Field(..., min_length = 1) # 하나 이상의 미션 포함 필수

# 개별 미션 분석 결과
class MissionResult(BaseModel):
    mission_id: int
    mission_type: MissionType
    success: bool        # 성공 여부
    confidence: float = Field(..., ge = 0.0, le = 1.0) # AI 신뢰도 (0.0 ~ 1.0)
    reason: Optional[str] = Field(default = None, description = "Explanation (only in debug mode)") # 판정 이유 (디버그용)

# 전체 분석 결과 응답
class MissionAnalysisData(BaseModel):
    analysis_id: str
    walk_id: int
    analyzed_at: str     # 분석 완료 시간 (ISO 포맷)
    missions: List[MissionResult] # 각 미션별 상세 결과 리스트