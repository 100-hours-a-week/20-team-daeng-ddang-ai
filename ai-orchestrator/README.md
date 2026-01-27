# AI Orchestrator (FastAPI)

이 서비스는 팀 **댕땅 프로젝트**의 AI Orchestrator 서버입니다.  
백엔드로부터 미션 분석 요청을 받아 외부 AI 모델(Gemini)을 활용해 미션 성공 여부를 판단하고 결과를 반환하는 역할을 합니다.

본 레포는 **AI 판단 로직 및 API 연동 검증을 위한 초기 구현 단계**이며, 모델 교체 및 고도화를 고려한 구조로 설계되었습니다.

---

## 기능 (Features)

### 1. 돌발 미션 분석 (Mission Analysis)
Gemini 1.5 Flash 모델을 활용하여 반려동물 산책 미션(예: "기다려 하기")의 성공 여부를 판단합니다.
- **Endpoint**: `POST /internal/v1/walks/{walk_id}/missions/analysis`
- **Input**: 동영상 URL, 미션 텍스트
- **Output**: 성공 여부(true/false), 판단 이유

### 2. 표정 분석 (Facial Analysis)
반려견의 표정을 분석하여 감정 상태를 추론하고 나레이션을 생성합니다.
- **Endpoint**: `POST /api/face/analyze`
- **Logic**: YOLOv10 (Face Detection) -> EfficientNet (Emotion Classification) -> Weighted Ensemble
- **Input**: 동영상 URL
- **Output**: 감정(happy, sad, etc.), 신뢰도, 나레이션 요약

---

## 설정 (Configuration)

`.env` 파일을 생성하고 다음 환경 변수를 설정해야 합니다. (참고: `.env.example`)

### 필수 변수
- `GEMINI_API_KEY`: Google Gemini API Key
- `HF_TOKEN`: HuggingFace Token (표정 분석 모델 다운로드용)

### 모드 설정 (Face Analysis)
표정 분석 기능을 **내부에서 실행(Monolithic)** 할지, **별도 서버로 분리(Microservice)** 할지 선택할 수 있습니다.

| 모드 (`FACE_MODE`) | 설명 | 설정 예시 |
| --- | --- | --- |
| **`local`** (기본) | 메인 서버 내부에서 직접 표정 분석 수행 | `FACE_MODE=local` |
| **`http`** | 별도의 Face Server로 요청을 보냄 | `FACE_MODE=http`<br>`FACE_SERVICE_URL=http://localhost:8100` |
| **`mock`** | 분석 없이 더미 데이터 반환 (개발용) | `FACE_MODE=mock` |

---

## 실행 (Run)

### 1. 가상 환경 및 의존성 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# 개발용 툴 설치 (Linter 등)
pip install ruff
```

### 2. 메인 서버 실행 (AI Orchestrator)
기본적으로 8000번 포트에서 실행됩니다.
```bash
uvicorn app.main:app --reload
```

### 3. Face Server 실행 (Optional)
`FACE_MODE=http`일 경우, 표정 분석 전용 서버를 별도로 띄워야 합니다. (기본 8100 포트)
```bash
python run_face_server.py
```

---

## 테스트 (Testing)

### 1. 돌발 미션 테스트 (Mission E2E)
실제 비디오 파일(`test_data/`)을 사용하여 Gemini 연동을 테스트합니다.
```bash
# 스크립트 실행 (내부적으로 test_data 폴더의 영상을 사용)
python app/services/mission_judge_02.py
```

### 2. 표정 분석 테스트 (Face Analysis)
#### 로컬 스크립트 테스트 (서버 실행 X)
서버를 띄우지 않고 로직(모델 추론 포함)만 바로 테스트하고 싶을 때 사용합니다.
```bash
python scripts/test_face_local.py <VIDEO_URL>
```

#### API 테스트 (cURL)
서버 실행 후 직접 요청을 보내 테스트합니다.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/face/analyze" \
     -H "Content-Type: application/json" \
     -d '{
           "analysis_id": "test_01",
           "video_url": "https://your-video-url.mp4"
         }'
```
