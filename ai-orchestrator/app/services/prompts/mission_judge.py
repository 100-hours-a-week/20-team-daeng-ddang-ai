from typing import List, Optional
from pydantic import BaseModel

# 미션별 판정 기준 데이터 (추후 DB로 이관 가능)
MISSION_REFERENCE_STORE_DATA = {
    "version": "v1",
    "missions": {
        "SIT": {
            "mission_type": "SIT",
            "mission_name": "앉아",
            "success_criteria": [
                "반려견의 엉덩이가 바닥에 닿아 있음",
                "앞다리가 몸을 지탱하고 있음",
                "해당 자세가 최소 1초 이상 유지됨",
                "정확한 초 단위 측정이 어렵더라도, 앉은 상태가 명확히 지속되는 흐름이 보임"
            ],
            "failure_criteria": [
                "엉덩이가 바닥에 닿지 않음",
                "자세가 1초 미만으로 유지됨",
                "단순히 서 있거나 걷는 상태에서 잠깐 자세를 낮춤",
                "동작이 프레임 밖에서 발생함"
            ]
        },
        "DOWN": {
            "mission_type": "DOWN",
            "mission_name": "엎드려",
            "success_criteria": [
                "반려견의 가슴과 배가 모두 지면에 닿아 있고, 앞다리가 접힌 상태로 바닥에 놓인 자세가 확인됨",
                "카메라 각도상 가슴 전체가 명확히 보이지 않더라도, 몸통이 낮고 엎드린 자세가 분명함",
                "해당 자세가 최소 2초 이상 유지됨",
                "정확한 초 단위 측정이 어렵더라도, 엎드린 상태가 명확히 지속되는 흐름이 보임"
            ],
            "failure_criteria": [
                "몸통이 지면에 닿지 않음",
                "엉덩이만 내려가고 가슴이 들려 있음",
                "자세가 2초 미만으로 유지됨"
            ]
        },
        "PAW": {
            "mission_type": "PAW",
            "mission_name": "손",
            "success_criteria": [
                "반려견의 앞발 중 한 발이 사람 손 위에 올라가 있고, 두 객체의 접촉이 명확하게 보임",
                "사람 손이 화면에 완전히 노출되지 않더라도, 접촉 순간이 분명함"
            ],
            "failure_criteria": [
                "앞발만 들고 손과 접촉하지 않음",
                "접촉이 불명확하거나 너무 짧음"
            ]
        },
        "TURN": {
            "mission_type": "TURN",
            "mission_name": "돌아",
            "success_criteria": [
                "반려견의 몸통이 하나의 동작으로 연속적으로 회전함",
                "회전각도가 충분하여 방향 전환이 아닌 회전 동작으로 인식됨"
            ],
            "failure_criteria": [
                "회전 각도가 매우 작음",
                "우연한 방향 전환으로 보임",
                "회전 동작이 끊기거나 부분적임",
                "고개만 돌림"
            ]
        },
        "JUMP": {
            "mission_type": "JUMP",
            "mission_name": "점프",
            "success_criteria": [
                "반려견의 앞발과 뒷발이 모두 지면에서 분리되거나, 뒷발은 지면에 닿은 상태로 두 앞발을 동시에 들어올림",
                "반려견의 몸 중심이 위로 상승하는 움직임이 관찰됨"
            ],
            "failure_criteria": [
                "반려견의 몸 중심이 위로 상승하는 움직임이 관찰되지 않음",
                "몸 중심 상승 없이 걷거나 서는 동작"
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

# Gemini에게 보낼 시스템 프롬프트 템플릿
PROMPT_TEMPLATE = """
당신은 반려견 훈련 영상을 판정하는 AI 심판입니다.
주어진 짧은 영상에 나타난 시각적 정보만을 사용하여,
지정된 하나의 미션이 성공적으로 수행되었는지를 판단하십시오.

[미션 정보]
- 미션 타입: {mission_type}
- 미션 이름: {mission_name}

[성공 기준]
{success_criteria_list}

[실패 기준]
{failure_criteria_list}

[판정 원칙]
- 오직 지정된 미션 하나만 평가하십시오.
- 영상에서 실제로 보이는 시각적 정보만 사용하십시오.
- 음성 명령, 사람의 말, 훈련 의도, 보상, 감정, 훈련 품질은
  판정 근거로 사용하지 마십시오.
- 성공 기준이 명확히 충족되지 않으면 FAILURE로 판정하십시오.
- 애매한 경우에는 FAILURE로 판정하고 confidence를 낮게 설정하십시오.

[시간 판단 규칙]
- 정확한 초 단위 측정이 어려운 경우,
  동일한 자세나 동작이 연속적으로 유지되는 흐름이
  명확히 보이면 기준 시간을 충족한 것으로 판단할 수 있습니다.
- 유지 여부가 불명확하면 FAILURE입니다.

[Confidence 산정 기준]
- 0.90 ~ 1.00 : 핵심 동작이 명확하고 반복적으로 관찰됨
- 0.75 ~ 0.89 : 핵심 동작은 있으나 일부 불확실성 존재
- 0.60 ~ 0.74 : 단서가 약하거나 해석이 필요한 수준
- 0.00 ~ 0.59 : 시각적 증거 부족 또는 판단 불가

[출력 형식]
아래 JSON 형식만 출력하십시오.
JSON 외의 텍스트는 절대 출력하지 마십시오.

- reason은 1~2문장으로 간결하게 작성하십시오.
- reason에는 판정에 결정적이었던 "시각적 근거"만 포함하십시오.
- 음성/명령/의도/칭찬/간식 등 오디오 기반 추정은 쓰지 마십시오.
- 반려견이 보이지 않거나 동작이 확인 불가하면 그 사실을 reason에 명시하십시오.

{{
  "success": boolean,
  "confidence": number,
  "reason": string
}}
"""
def build_prompt(mission_type: str) -> str:
    # 미션 타입에 해당하는 기준을 조회
    criteria = get_mission_criteria(mission_type)

    if not criteria:
        raise ValueError(f"Unknown mission type: {mission_type}")

    # 리스트 형태의 기준들을 줄바꿈 문자로 연결
    success_list = "\n".join([f"- {item}" for item in criteria.success_criteria])
    failure_list = "\n".join([f"- {item}" for item in criteria.failure_criteria])

    # 템플릿에 데이터 주입하여 최종 프롬프트 생성
    return PROMPT_TEMPLATE.format(
        mission_type=criteria.mission_type,
        mission_name=criteria.mission_name,
        success_criteria_list=success_list,
        failure_criteria_list=failure_list
    )
