# app/services/mission_service.py
import logging, json
from datetime import datetime
from typing import List
from fastapi import HTTPException # For re-raising 429
from app.core.config import DEBUG
from app.schemas.mission_schema import MissionResult, MissionInput
from app.services.gemini_client import GeminiClient
from app.services.prompts.mission_judge import build_prompt

gemini_client = GeminiClient()

logger = logging.getLogger(__name__)

import backoff 
from google.api_core.exceptions import ResourceExhausted

# --- 429 Error (Rate Limit) Handling with Backoff ---
# @backoff.on_exception(
#     backoff.expo,
#     (ResourceExhausted, HTTPException), # 429 에러 등을 잡아서 재시도
#     max_tries=3,
#     factor=2, # 지수 백오프 (2초, 4초, 8초...)
#     giveup=lambda e: getattr(e, "status_code", None) != 429 and not isinstance(e, ResourceExhausted)
# )
# ----------------------------------------------------
def analyze_sync(missions: List[MissionInput]) -> List[MissionResult]:
    results: List[MissionResult] = []

    # 요청받은 모든 미션에 대해 순차적으로 처리
    for m in missions:
        try:
            # --- 429 Error Handling (Retry Logic) ---
            # 재시도 로직이 필요하다면 위에 정의한 @backoff 데코레이터가 적용된 별도 함수를 호출해야 합니다.
            # 예: result_text = _call_gemini_with_retry(gemini_client, m.video_url, prompt)
            
            # 미션 타입에 맞는 심사 기준 프롬프트 생성 (예: SIT -> 앉아 미션 기준)
            prompt = build_prompt(m.mission_type.value)
            
            # Gemini에게 비디오와 프롬프트를 전송하여 판정 요청
            result_text = gemini_client.generate_from_video_url(
                video_url=m.video_url,
                prompt_text=prompt
            )

            # (기존 주석 처리된 예외 처리는 제거하거나 유지)
            # -----------------------------------------------
            
            # JSON 파싱을 위한 전처리 (마크다운 코드 블록 제거 등)
            clean = result_text.strip()
            clean = clean.replace("```json", "").replace("```", "").strip()

            # 중괄호 {} 찾아서 JSON 부분만 추출
            l = clean.find("{")
            r = clean.rfind("}")
            if l != -1 and r != -1 and r > l:
                clean = clean[l:r+1]

            # 문자열을 파이썬 딕셔너리로 변환
            result_json = json.loads(clean)
            
            # 결과 필드 추출
            success = result_json.get("success", False)
            confidence = result_json.get("confidence", 0.0)
            reason = result_json.get("reason", "")
            
            # 결과 리스트에 추가
            results.append(
                MissionResult(
                    mission_id = m.mission_id,
                    mission_type = m.mission_type,
                    success = success,
                    confidence = confidence,
                    reason = reason if DEBUG else None
                )
            )
            
        except Exception as e:
            # 오류 발생 시 로그 남기고 실패 처리 (전체 프로세스 중단 방지)
            logger.exception(f"Mission Analysis Failed for {m.mission_id}")
            results.append(
                MissionResult(
                    mission_id = m.mission_id,
                    mission_type = m.mission_type,
                    success = False,
                    confidence = 0.0
                )
            )
    return results

# --- 비동기 확장용 (Async Expansion) ---
# async def analyze_async(missions: List[MissionInput]) -> List[MissionResult]:
#     # 비동기 처리를 위해서는 gather 등을 사용하여 병렬로 요청할 수 있습니다.
#     import asyncio
#
#     results: List[MissionResult] = []
#     
#     # 각 미션에 대해 비동기 작업을 생성
#     tasks = []
#     for m in missions:
#         tasks.append(_process_single_mission_async(m))
#
#     # 병렬 실행 및 결과 수집
#     results = await asyncio.gather(*tasks)
#     return list(results)
#
# async def _process_single_mission_async(m: MissionInput) -> MissionResult:
#     try:
#         prompt = build_prompt(m.mission_type.value)
#         
#         # GeminiClient에 추가된 async 메서드 사용 필요
#         result_text = await gemini_client.generate_from_video_url_async(
#             video_url=m.video_url,
#             prompt_text=prompt
#         )
#         
#         clean = result_text.strip()
#         clean = clean.replace("```json", "").replace("```", "").strip()
#
#         l = clean.find("{")
#         r = clean.rfind("}")
#         if l != -1 and r != -1 and r > l:
#             clean = clean[l:r+1]
#
#         result_json = json.loads(clean)
#         
#         return MissionResult(
#             mission_id = m.mission_id,
#             mission_type = m.mission_type,
#             success = result_json.get("success", False),
#             confidence = result_json.get("confidence", 0.0),
#             reason = result_json.get("reason", "")
#         )
#         
#     except Exception as e:
#         logger.error(f"Async Mission Analysis Failed for {m.mission_id}: {e}")
#         return MissionResult(
#             mission_id = m.mission_id,
#             mission_type = m.mission_type,
#             success = False,
#             confidence = 0.0,
#             reason = f"Error: {str(e)}"
#         )
# -------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()