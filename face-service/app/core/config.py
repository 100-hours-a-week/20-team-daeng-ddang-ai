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

# Force CPU Device (Default: cpu) - can be overridden by env var 'TORCH_DEVICE'
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")