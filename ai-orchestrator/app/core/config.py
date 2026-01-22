# app/core/config.py
import os

FACE_MODE = os.getenv("FACE_MODE", "mock").lower()
FACE_SERVICE_URL = os.getenv("FACE_SERVICE_URL", "http://localhost:8100").rstrip("/")
FACE_HTTP_TIMEOUT_SECONDS = float(os.getenv("FACE_HTTP_TIMEOUT_SECONDS", "20"))

FACE_JOB_MODE = os.getenv("FACE_JOB_MODE", "off").lower()