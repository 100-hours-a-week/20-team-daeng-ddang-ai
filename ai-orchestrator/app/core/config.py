# app/core/config.py
import os

# 디버그 모드 (True일 경우 상세 로그 및 사유 반환)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

FACE_MODE = os.getenv("FACE_MODE", "http").lower()
FACE_SERVICE_URL = os.getenv("FACE_SERVICE_URL", "http://localhost:8100").rstrip("/")
FACE_HTTP_TIMEOUT_SECONDS = float(os.getenv("FACE_HTTP_TIMEOUT_SECONDS", "20"))

FACE_JOB_MODE = os.getenv("FACE_JOB_MODE", "off").lower()

# Healthcare Service Config
HEALTHCARE_MODE = os.getenv("HEALTHCARE_MODE", "http").lower()
HEALTHCARE_SERVICE_URL = os.getenv("HEALTHCARE_SERVICE_URL", "http://localhost:8200").rstrip("/")
HEALTHCARE_HTTP_TIMEOUT_SECONDS = float(os.getenv("HEALTHCARE_HTTP_TIMEOUT_SECONDS", "60"))
