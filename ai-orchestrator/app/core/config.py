# app/core/config.py
import os

# 디버그 모드 (True일 경우 상세 로그 및 사유 반환)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

FACE_MODE = os.getenv("FACE_MODE", "mock").lower()
FACE_SERVICE_URL = os.getenv("FACE_SERVICE_URL", "http://localhost:8100").rstrip("/")
FACE_HTTP_TIMEOUT_SECONDS = float(os.getenv("FACE_HTTP_TIMEOUT_SECONDS", "20"))

FACE_JOB_MODE = os.getenv("FACE_JOB_MODE", "off").lower()

# Local Face Analysis (Dog) Config
FACE_DETECTION_MODEL_ID = os.getenv("FACE_DETECTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-detection")
FACE_EMOTION_MODEL_ID = os.getenv("FACE_EMOTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-emotion-classification")
HF_TOKEN = os.getenv("HF_TOKEN")