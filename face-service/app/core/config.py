# app/core/config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# 디버그 모드 (True일 경우 상세 로그 및 사유 반환)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Face Analysis Model Config
FACE_DETECTION_MODEL_ID = os.getenv("FACE_DETECTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-detection")
FACE_EMOTION_MODEL_ID = os.getenv("FACE_EMOTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-emotion-classification")
HF_TOKEN = os.getenv("HF_TOKEN")
CHECK_MODEL_UPDATE_ON_START = os.getenv("CHECK_MODEL_UPDATE_ON_START", "true").lower() in {"1", "true", "yes", "on"}
FORCE_REFRESH_MODELS = os.getenv("FORCE_REFRESH_MODELS", "false").lower() in {"1", "true", "yes", "on"}
MODEL_UPDATE_CHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_UPDATE_CHECK_INTERVAL_SECONDS", "86400"))
FACE_DETECTION_REVISION_FILE = os.getenv("FACE_DETECTION_REVISION_FILE", "models/.face_detection_revision")
FACE_EMOTION_REVISION_FILE = os.getenv("FACE_EMOTION_REVISION_FILE", "models/.face_emotion_revision")

# Force CPU Device (Default: cpu) - can be overridden by env var 'TORCH_DEVICE'
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")
