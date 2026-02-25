# app/core/config.py
import os

# 디버그 모드
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 챗봇 모델 설정
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_PATH  = os.getenv("ADAPTER_PATH",  "models/lora-qwen-7b-final")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "models/chroma_db")

# 서버 설정
PORT = int(os.getenv("PORT", 8300))
