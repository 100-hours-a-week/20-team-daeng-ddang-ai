from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8500"))

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models")

IMAGE_DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "15"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

DEVICE = os.getenv("DEVICE", "auto").strip().lower()

ROUTE_MODEL_ID = os.getenv("ROUTE_MODEL_ID", "MobileCLIP-S1").strip()
ROUTE_MODEL_REVISION = os.getenv("ROUTE_MODEL_REVISION") or None
ROUTE_MODEL_PRETRAINED = os.getenv("ROUTE_MODEL_PRETRAINED", "datacompdr").strip()
ROUTE_MIN_CONFIDENCE = float(os.getenv("ROUTE_MIN_CONFIDENCE", "0.45"))
ROUTE_TOP_K = int(os.getenv("ROUTE_TOP_K", "3"))

EYE_MODEL_ID = os.getenv(
    "EYE_MODEL_ID",
    "20-team-daeng-ddang-ai/vet-chat",
).strip()
EYE_MODEL_SUBDIR = os.getenv("EYE_MODEL_SUBDIR", "eye_disease_classifier").strip().strip("/")
EYE_MODEL_FILENAME = os.getenv("EYE_MODEL_FILENAME", "best.pt").strip()
EYE_RUN_CONFIG_FILENAME = os.getenv("EYE_RUN_CONFIG_FILENAME", "run_config.json").strip()
EYE_MODEL_REVISION = os.getenv("EYE_MODEL_REVISION") or None
EYE_TOP_K = int(os.getenv("EYE_TOP_K", "3"))

SKIN_MODEL_ID = os.getenv("SKIN_MODEL_ID", "").strip()
SKIN_MODEL_SUBDIR = os.getenv("SKIN_MODEL_SUBDIR", "skin_disease_classifier").strip().strip("/")
SKIN_MODEL_FILENAME = os.getenv("SKIN_MODEL_FILENAME", "best.pt").strip()
SKIN_RUN_CONFIG_FILENAME = os.getenv("SKIN_RUN_CONFIG_FILENAME", "run_config.json").strip()
SKIN_MODEL_REVISION = os.getenv("SKIN_MODEL_REVISION") or None
SKIN_TOP_K = int(os.getenv("SKIN_TOP_K", "3"))
