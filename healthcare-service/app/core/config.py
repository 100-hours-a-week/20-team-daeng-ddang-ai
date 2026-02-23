import os
from dotenv import load_dotenv

# Load environment variables early so analyze_health picks up DEBUG_MODE
load_dotenv()

# Service settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8200"))

# Model settings
HEALTH_MODEL_ID = os.getenv("HEALTH_MODEL_ID", "20-team-daeng-ddang-ai/dog-pose-estimation")
HEALTH_MODEL_FILENAME = os.getenv("HEALTH_MODEL_FILENAME", "best.pt")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models")

# Output artifacts
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# AWS S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = os.getenv("S3_PREFIX", "healthcare")

# HuggingFace token (optional for private repos)
HF_TOKEN = os.getenv("HF_TOKEN")

# Misc
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_MS", "60000")) / 1000.0  # seconds
