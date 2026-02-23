from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, BotoCoreError
from huggingface_hub import hf_hub_download

from app.core.config import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    DEBUG_MODE,
    HEALTH_MODEL_FILENAME,
    HEALTH_MODEL_ID,
    HF_TOKEN,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    S3_BUCKET_NAME,
    S3_PREFIX,
)
from app.schemas.health_schema import HealthAnalyzeRequest, HealthAnalyzeResponse

# Ensure DEBUG_MODE environment variable is propagated to analyze_health.Config
if os.getenv("DEBUG_MODE") is None:
    os.environ["DEBUG_MODE"] = "true" if DEBUG_MODE else "false"

# Heavy dependency imported after env setup
from scripts.analyze_health import DogHealthAnalyzer  # type: ignore

logger = logging.getLogger(__name__)


class HealthAnalyzerService:
    def __init__(self) -> None:
        self.output_dir = OUTPUT_DIR
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        model_path = self._resolve_model_path()
        logger.info("Loading DogHealthAnalyzer with model=%s", model_path)
        self.analyzer = DogHealthAnalyzer(model_path=model_path, output_dir=self.output_dir)

        self.s3_client = None
        if S3_BUCKET_NAME:
            logger.info("S3 uploads enabled. bucket=%s prefix=%s", S3_BUCKET_NAME, S3_PREFIX)
            self.s3_client = boto3.client(
                "s3",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                config=BotoConfig(connect_timeout=REQUEST_TIMEOUT, read_timeout=REQUEST_TIMEOUT),
            )
        else:
            logger.warning("S3_BUCKET_NAME not set. Overlay videos will not be uploaded.")

    def analyze(self, req: HealthAnalyzeRequest) -> HealthAnalyzeResponse:
        analysis_id = req.analysis_id or str(uuid.uuid4())
        dog_id = int(req.dog_id) if req.dog_id is not None else 123

        logger.info("Healthcare analyze start id=%s dog_id=%s debug=%s", analysis_id, dog_id, DEBUG_MODE)
        report = self.analyzer.analyze_video(video_source=req.video_url, dog_id=dog_id, analysis_id=analysis_id)

        # Upload overlay artifact to S3 if available
        artifacts = report.get("artifacts") or {}
        overlay_name = artifacts.get("keypoint_overlay_video_url")
        if overlay_name:
            local_overlay = Path(self.output_dir) / overlay_name
            uploaded_url = self._upload_overlay(local_overlay, analysis_id)
            if uploaded_url:
                artifacts["keypoint_overlay_video_url"] = uploaded_url
                report["artifacts"] = artifacts

        return HealthAnalyzeResponse.model_validate(report)

    def _upload_overlay(self, local_path: Path, analysis_id: str) -> Optional[str]:
        if not self.s3_client or not S3_BUCKET_NAME:
            return None

        if not local_path.exists():
            logger.warning("Overlay file not found for upload: %s", local_path)
            return None

        s3_key = f"{S3_PREFIX.rstrip('/')}/{analysis_id}/{local_path.name}"
        try:
            self.s3_client.upload_file(str(local_path), S3_BUCKET_NAME, s3_key)
            url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
            logger.info("Uploaded overlay to S3: %s", url)
            return url
        except (ClientError, BotoCoreError) as e:
            logger.error("Failed to upload overlay to S3: %s", e)
            return None
        finally:
            try:
                local_path.unlink()
            except OSError:
                pass

    def _resolve_model_path(self) -> str:
        # If local path exists, use it
        local_candidate = Path(HEALTH_MODEL_ID)
        if local_candidate.exists():
            return str(local_candidate)

        model_dir = Path(MODEL_CACHE_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        cached_path = model_dir / HEALTH_MODEL_FILENAME
        if cached_path.exists():
            return str(cached_path)

        # Try downloading from HuggingFace repo
        if "/" in HEALTH_MODEL_ID:
            logger.info("Downloading healthcare model from HF: %s (%s)", HEALTH_MODEL_ID, HEALTH_MODEL_FILENAME)
            try:
                downloaded = hf_hub_download(
                    repo_id=HEALTH_MODEL_ID,
                    filename=HEALTH_MODEL_FILENAME,
                    token=HF_TOKEN,
                    cache_dir=str(model_dir),
                )
                return downloaded
            except Exception as e:
                logger.error("Failed to download model from HF (%s): %s", HEALTH_MODEL_ID, e)

        # Fallback to provided path even if not found (ultralytics will handle)
        return str(cached_path if cached_path.exists() else HEALTH_MODEL_ID)
