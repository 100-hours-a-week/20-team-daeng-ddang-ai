from typing import List, Optional
from pydantic import BaseModel

MISSION_REFERENCE_STORE_DATA = {
    "version": "v1",
    "missions": {
        "SIT": {
            "mission_type": "SIT",
            "mission_name": "앉아",
            "success_criteria": [
                "반려견의 엉덩이가 바닥에 닿아 있음",
                "앞다리가 몸을 지탱하고 있음",
                "해당 자세가 최소 1초 이상 유지됨"
            ],
            "failure_criteria": [
                "엉덩이가 바닥에 닿지 않음",
                "자세가 1초 미만으로 유지됨",
                "동작이 프레임 밖에서 발생함"
            ]
        },
        "DOWN": {
            "mission_type": "DOWN",
            "mission_name": "엎드려",
            "success_criteria": [
                "반려견의 가슴 또는 배가 바닥에 닿아 있음",
                "앞다리가 접힌 상태로 바닥에 놓여 있음",
                "해당 자세가 최소 1초 이상 유지됨"
            ],
            "failure_criteria": [
                "몸통이 바닥에 닿지 않음",
                "엉덩이만 내려가고 가슴이 들려 있음",
                "자세가 충분히 유지되지 않음"
            ]
        },
        "PAW": {
            "mission_type": "PAW",
            "mission_name": "손",
            "success_criteria": [
                "반려견의 앞발이 사람 손 위에 올라가 있음",
                "사람 손과의 접촉이 명확하게 보임",
                "접촉이 최소 1초 이상 유지됨"
            ],
            "failure_criteria": [
                "앞발만 들고 손과 접촉하지 않음",
                "사람 손이 영상 프레임에 보이지 않음",
                "접촉이 불명확하거나 너무 짧음"
            ]
        },
        "TURN": {
            "mission_type": "TURN",
            "mission_name": "돌아",
            "success_criteria": [
                "반려견이 몸을 기준으로 명확히 회전함",
                "회전 동작이 연속적으로 수행됨",
                "의도적인 회전 동작으로 판단됨"
            ],
            "failure_criteria": [
                "회전 각도가 매우 작음",
                "우연한 방향 전환으로 보임",
                "회전 동작이 중간에 끊김"
            ]
        },
        "JUMP": {
            "mission_type": "JUMP",
            "mission_name": "점프",
            "success_criteria": [
                "반려견의 네 발이 동시에 지면에서 떨어짐",
                "공중에 떠 있는 순간이 명확히 보임"
            ],
            "failure_criteria": [
                "앞발만 지면에서 떨어짐",
                "뒷발이 계속 지면에 닿아 있음",
                "공중에 뜬 순간이 불명확함"
            ]
        }
    }
}

class MissionCriteria(BaseModel):
    mission_type: str
    mission_name: str
    success_criteria: List[str]
    failure_criteria: List[str]

def get_mission_criteria(mission_type: str) -> Optional[MissionCriteria]:
    data = MISSION_REFERENCE_STORE_DATA.get("missions", {}).get(mission_type)

    if data:
        return MissionCriteria(**data)
    return None

PROMPT_TEMPLATE = """
You are an AI judge that determines whether a dog successfully performed
a specific training mission based on a short video.

[Mission Information]
- Mission Type: {mission_type}
- Mission Name: {mission_name}

[Success Criteria]
{success_criteria_list}

[Failure Criteria]
{failure_criteria_list}

[Judgment Rules]
- Judge ONLY the specified mission.
- Use ONLY visual information from the video.
- If the mission is not clearly satisfied, mark it as FAILURE.
- Do NOT evaluate other behaviors.
- Do NOT explain training quality or intent.

[Output Format]
Return ONLY the following JSON format.
Do NOT include any additional text.

{{
  "success": boolean,
  "confidence": number
}}
"""
def build_prompt(mission_type: str) -> str:
    criteria = get_mission_criteria(mission_type)

    if not criteria:
        raise ValueError(f"Unknown mission type: {mission_type}")

    success_list = "\n".join([f"- {item}" for item in criteria.success_criteria])
    failure_list = "\n".join([f"- {item}" for item in criteria.failure_criteria])

    return PROMPT_TEMPLATE.format(
        mission_type=criteria.mission_type,
        mission_name=criteria.mission_name,
        success_criteria_list=success_list,
        failure_criteria_list=failure_list
    )
