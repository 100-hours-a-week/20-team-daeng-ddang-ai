from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import requests

from app.core.config import (
    JOB_EVENT_AUTH_TOKEN,
    JOB_EVENT_CALLBACK_URL,
    JOB_EVENT_MAX_RETRIES,
    JOB_EVENT_RETRY_BACKOFF_SECONDS,
    JOB_EVENT_SOURCE,
    JOB_EVENT_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PREPARING_INPUT = "PREPARING_INPUT"
    ANALYZING = "ANALYZING"
    REPORT_GENERATING = "REPORT_GENERATING"
    DONE = "DONE"
    FAILED = "FAILED"


def publish_job_event(
    job_id: str,
    status: JobStatus,
    message: str,
    *,
    progress: int | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not JOB_EVENT_CALLBACK_URL or not job_id:
        return

    payload = {
        "job_id": job_id,
        "status": status.value,
        "message": message,
        "progress": progress,
        "error_code": error_code,
        "source": JOB_EVENT_SOURCE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    headers = {"Content-Type": "application/json"}
    if JOB_EVENT_AUTH_TOKEN:
        headers["X-Internal-Token"] = JOB_EVENT_AUTH_TOKEN

    max_attempts = max(1, JOB_EVENT_MAX_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                JOB_EVENT_CALLBACK_URL,
                json=payload,
                headers=headers,
                timeout=JOB_EVENT_TIMEOUT_SECONDS,
            )
            if response.ok:
                return

            # Retry only on temporary upstream failures.
            if response.status_code < 500:
                logger.warning(
                    "Job event rejected without retry. status=%s body=%s job_id=%s event_status=%s",
                    response.status_code,
                    response.text[:300],
                    job_id,
                    status.value,
                )
                return

            raise requests.HTTPError(f"upstream_status={response.status_code}")
        except requests.RequestException as exc:
            if attempt >= max_attempts:
                logger.warning(
                    "Failed to publish job event after retries. job_id=%s event_status=%s error=%s",
                    job_id,
                    status.value,
                    exc,
                )
                return
            sleep_seconds = JOB_EVENT_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)
