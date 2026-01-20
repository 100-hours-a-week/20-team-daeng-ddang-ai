from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    message: str
    data: Optional[dict] = None
    errorCode: Optional[str] = None


class MissionType(str, Enum):
    SIT = "SIT"
    DOWN = "DOWN"
    PAW = "PAW"
    TURN = "TURN"
    JUMP = "JUMP"


class MissionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class MissionInput(BaseModel):
    mission_id: str
    mission_type: MissionType
    video_url: str


class MissionAnalysisRequest(BaseModel):
    analysis_id: str
    walk_id: str
    missions: List[MissionInput] = Field(..., min_length = 1)


class MissionResult(BaseModel):
    mission_id: str
    mission_type: MissionType
    success: bool
    confidence: float = Field(..., ge = 0.0, le = 1.0)


class MissionAnalysisData(BaseModel):
    analysis_id: str
    walk_id: str
    analyzed_at: str
    missions: List[MissionResult]
