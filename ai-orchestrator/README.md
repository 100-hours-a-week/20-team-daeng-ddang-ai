# AI Orchestrator (FastAPI)

이 서비스는 팀 **댕땅 프로젝트**의 AI Orchestrator 서버입니다.  
백엔드로부터 미션 분석 요청을 받아 외부 AI 모델(Gemini)을 활용해
미션 성공 여부를 판단하고 결과를 반환하는 역할을 합니다.

본 레포는 **AI 판단 로직 및 API 연동 검증을 위한 초기 구현 단계**이며,
모델 교체 및 고도화를 고려한 구조로 설계되었습니다.

---

## 주요 기능
- FastAPI 기반 AI Orchestrator 서버
- 미션 분석 API 제공  
  (`/internal/v1/walks/{walk_id}/missions/analysis`)
- Prompt / Schema 기반 Gemini API 연동
- E2E 테스트 스크립트 제공 (백엔드 연동 검증용)

---

## 역할 및 책임 범위
- AI 서버는 **판단 로직만 수행**합니다.
- 유저, 산책, 미션 데이터의 저장 및 비즈니스 로직은 **백엔드 책임**입니다.
- 본 서버는 stateless하게 동작하며, 입력을 기반으로 분석 결과만 반환합니다.

---

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill values
uvicorn app.main:app --reload
