from __future__ import annotations

import logging
from datetime import datetime
import json
from typing import List

from app.schemas.mission_schema import MissionResult, MissionType, MissionInput
from app.services.gemini_client import GeminiClient
from app.services.prompts.mission_judge import build_prompt

gemini_client = GeminiClient()

logger = logging.getLogger(__name__)

def default_title(mission_type: MissionType) -> str:
    return {
        MissionType.SIT: "Sit",
        MissionType.DOWN: "Down",
        MissionType.PAW: "Paw",
        MissionType.TURN: "Turn",
        MissionType.JUMP: "Jump",
    }[mission_type]

def analyze_sync_mock(missions: List[MissionInput]) -> List[MissionResult]:
    results: List[MissionResult] = []

    for m in missions:
        is_success = m.mission_type in {MissionType.SIT, MissionType.PAW}
        
        results.append(
            MissionResult(
                mission_id = m.mission_id,
                mission_type = m.mission_type,
                success = is_success,
                confidence = 0.92 if is_success else 0.61
            )
        )

    return results

def analyze_sync(missions: List[MissionInput]) -> List[MissionResult]:
    results: List[MissionResult] = []

    for m in missions:
        try:
            prompt = build_prompt(m.mission_type.value)
            
            result_text = gemini_client.generate_from_video_url(
                video_url=m.video_url,
                prompt_text=prompt
            )
            
            result_json = json.loads(result_text)
            
            success = result_json.get("success", False)
            confidence = result_json.get("confidence", 0.0)
            
            results.append(
                MissionResult(
                    mission_id = m.mission_id,
                    mission_type = m.mission_type,
                    success = success,
                    confidence = confidence
                )
            )
            
        except Exception as e:
            logger.error(f"Mission Analysis Failed for {m.mission_id}: {e}")

            results.append(
                MissionResult(
                    mission_id = m.mission_id,
                    mission_type = m.mission_type,
                    success = False,
                    confidence = 0.0
                )
            )

    return results

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()
