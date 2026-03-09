# Face Analysis Microservice

이 서비스는 강아지 얼굴을 감지하고 표정을 분석하는 마이크로서비스입니다.

## 1. 시스템 요구사항 (System Requirements)
- **Python**: 3.10 이상
- **OS**: Linux / macOS
- **필수 시스템 패키지**: `ffmpeg`
  - 영상에서 프레임을 추출하기 위해 **FFmpeg가 반드시 설치되어 있어야 합니다.**
  - **Ubuntu/Debian**:
    ```bash
    sudo apt-get update && sudo apt-get install -y ffmpeg
    ```
  - **macOS**: `brew install ffmpeg`

## 2. 설치 (Installation)
```bash
cd face-service
pip install -r requirements.txt
```

## 3. 환경 변수 (Environment Variables)
배포 환경(Cloud Secret 등)에 다음 변수들을 설정해야 합니다.

| 변수명 | 필수 여부 | 설명 | 예시 값 |
|--------|-----------|------|---------|
| `PORT` | 선택 | 서비스 포트 (기본값: 8100) | `8100` |
| `HF_TOKEN` | **필수** | Hugging Face 모델 다운로드용 토큰 | `hf_...` |
| `FACE_DETECTION_MODEL_ID` | 선택 | YOLO 감지 모델 ID | `20-team-daeng-ddang-ai/dog-detection` |
| `FACE_EMOTION_MODEL_ID` | 선택 | 표정 분석 모델 ID | `20-team-daeng-ddang-ai/dog-emotion-classification` |
| `CHECK_MODEL_UPDATE_ON_START` | 선택 | HF 리비전 자동 비교 활성화 | `true` |
| `MODEL_UPDATE_CHECK_INTERVAL_SECONDS` | 선택 | 백그라운드 체크 주기(초) | `86400` |
| `FORCE_REFRESH_MODELS` | 선택 | `true`면 주기마다 강제 리로드 | `false` |
| `FACE_DETECTION_REVISION_FILE` | 선택 | 감지 모델 리비전 저장 파일 | `models/.face_detection_revision` |
| `FACE_EMOTION_REVISION_FILE` | 선택 | 감정 모델 리비전 저장 파일 | `models/.face_emotion_revision` |

## 4. 실행 (Running)
```bash
python run.py
```
Health Check: `GET /health`

### 모델 리비전 체크 동작
- 시작 시 모델 로드 후 현재 HF revision을 로컬 파일(`models/.face_*_revision`)에 기록합니다.
- 백그라운드 체크는 `MODEL_UPDATE_CHECK_INTERVAL_SECONDS` 주기로 동작합니다.
- `FORCE_REFRESH_MODELS=true`면 주기마다 강제 리로드됩니다. 운영에서는 일반적으로 `false` 권장입니다.

## 5. Docker 운영 메모
- 컨테이너는 non-root 사용자로 실행됩니다.
- `HEALTHCHECK`가 `/health`를 주기적으로 확인합니다.

## 6. 주요 기술 스택
- **Web Framework**: FastAPI, Uvicorn
- **ML/Vision**: PyTorch, Ultralytics (YOLO), Torchvision (EfficientNet), OpenCV, Pillow
