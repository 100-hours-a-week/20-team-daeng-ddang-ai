from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from starlette.concurrency import run_in_threadpool

from app.core.config import JOB_EVENT_SOURCE
from app.schemas.health_schema import (
    HealthAnalyzeRequest,
    HealthAnalyzeResponse,
    HealthJobCreateResponse,
    HealthJobStatusResponse,
)
from app.services.health_analyzer import HealthAnalyzerService
from app.services.job_events import JobStatus, publish_job_event

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _JobState:
    job_id: str
    request: HealthAnalyzeRequest
    status: JobStatus
    timestamp: str
    progress: int | None = None
    error_code: str | None = None
    message: str | None = None
    result: HealthAnalyzeResponse | None = None


class HealthcareJobQueue:
    def __init__(
        self,
        get_analyzer_service: Callable[[], Optional[HealthAnalyzerService]],
        *,
        max_queue_size: int,
    ) -> None:
        self._get_analyzer_service = get_analyzer_service
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._jobs: dict[str, _JobState] = {}
        self._jobs_lock = threading.RLock()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Healthcare async job worker started.")

    async def stop(self) -> None:
        if not self._worker_task:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Healthcare async job worker stopped.")

    async def enqueue(self, req: HealthAnalyzeRequest) -> HealthJobCreateResponse:
        if self._queue.full():
            raise RuntimeError("JOB_QUEUE_FULL")

        job_id = req.analysis_id or str(uuid.uuid4())
        req_with_id = req.model_copy(update={"analysis_id": job_id})
        now = _utc_now_iso()

        state = _JobState(
            job_id=job_id,
            request=req_with_id,
            status=JobStatus.QUEUED,
            timestamp=now,
            progress=5,
            message="분석 요청을 접수했습니다.",
        )
        with self._jobs_lock:
            self._jobs[job_id] = state

        dog_id = int(req_with_id.dog_id) if req_with_id.dog_id is not None else 123
        publish_job_event(
            job_id,
            JobStatus.QUEUED,
            "분석 요청을 접수했습니다.",
            progress=5,
            metadata={"dog_id": dog_id},
        )

        await self._queue.put(job_id)
        return HealthJobCreateResponse(
            job_id=job_id,
            status=state.status.value,
            timestamp=state.timestamp,
            progress=state.progress,
            error_code=state.error_code,
            source=JOB_EVENT_SOURCE,
            message=state.message,
        )

    def get_status(self, job_id: str) -> HealthJobStatusResponse | None:
        with self._jobs_lock:
            state = self._jobs.get(job_id)
            if not state:
                return None
            return HealthJobStatusResponse(
                job_id=state.job_id,
                status=state.status.value,
                timestamp=state.timestamp,
                progress=state.progress,
                error_code=state.error_code,
                source=JOB_EVENT_SOURCE,
                message=state.message,
                result=state.result,
            )

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process_job(job_id)
            finally:
                self._queue.task_done()

    async def _process_job(self, job_id: str) -> None:
        with self._jobs_lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            request = state.request

        analyzer = self._get_analyzer_service()
        if analyzer is None:
            self._set_state(
                job_id,
                status=JobStatus.FAILED,
                error_code="ANALYZER_NOT_INITIALIZED",
                message="분석기가 준비되지 않았습니다.",
            )
            return

        def _status_hook(
            status: JobStatus,
            message: str,
            progress: int | None = None,
            error_code: str | None = None,
            metadata: dict | None = None,
        ) -> None:
            del metadata
            self._set_state(
                job_id,
                status=status,
                progress=progress,
                error_code=error_code,
                message=message,
            )

        try:
            result = await run_in_threadpool(
                analyzer.analyze,
                request,
                emit_queued_event=False,
                status_hook=_status_hook,
            )
            self._set_result(job_id, result)
        except Exception as exc:
            logger.exception("Async healthcare job failed. job_id=%s error=%s", job_id, exc)
            self._set_state(
                job_id,
                status=JobStatus.FAILED,
                error_code="ANALYSIS_FAILED",
                message="AI 분석 중 오류가 발생했습니다.",
            )

    def _set_result(self, job_id: str, result: HealthAnalyzeResponse) -> None:
        with self._jobs_lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state.result = result
            state.status = JobStatus.DONE
            state.progress = 100
            state.timestamp = _utc_now_iso()
            state.error_code = None
            state.message = "분석이 완료되었습니다."

    def _set_state(
        self,
        job_id: str,
        *,
        status: JobStatus,
        progress: int | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._jobs_lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state.status = status
            state.progress = progress
            state.error_code = error_code
            state.message = message
            state.timestamp = _utc_now_iso()
