import logging
import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi
from starlette.concurrency import run_in_threadpool

from app.core.config import (
    ASYNC_JOB_MODE_ENABLED,
    ASYNC_JOB_QUEUE_MAX_SIZE,
    CHECK_MODEL_UPDATE_ON_START,
    DEBUG_MODE,
    FORCE_REFRESH_MODELS,
    HEALTH_MODEL_ID,
    HEALTH_MODEL_REVISION_FILE,
    HF_TOKEN,
    MODEL_UPDATE_CHECK_INTERVAL_SECONDS,
)
from app.schemas.health_schema import (
    HealthAnalyzeRequest,
    HealthAnalyzeResponse,
    HealthJobCreateResponse,
    HealthJobStatusResponse,
)
from app.services.health_analyzer import HealthAnalyzerService
from app.services.job_queue import HealthcareJobQueue

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthcare_server")

analyzer_service: HealthAnalyzerService | None = None
analyzer_lock = threading.RLock()
reload_lock = threading.Lock()
job_queue: HealthcareJobQueue | None = None


def _repo_like(model_id: str) -> bool:
    return "/" in model_id and not os.path.exists(model_id)


def _read_revision(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        return revision or None
    except OSError as e:
        logger.warning("Failed to read revision file %s: %s", path, e)
        return None


def _write_revision(path: str, revision: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(revision)
    except OSError as e:
        logger.warning("Failed to write revision file %s: %s", path, e)


def _fetch_remote_revision(repo_id: str) -> Optional[str]:
    if not HF_TOKEN:
        logger.warning("HF_TOKEN missing; skip revision check for %s", repo_id)
        return None
    try:
        return HfApi(token=HF_TOKEN).model_info(repo_id=repo_id, revision="main").sha
    except Exception as e:
        logger.warning("Failed to fetch remote revision for %s: %s", repo_id, e)
        return None


def _reload_if_model_updated() -> bool:
    if not CHECK_MODEL_UPDATE_ON_START:
        return False

    target_revision = None
    should_reload = FORCE_REFRESH_MODELS

    if _repo_like(HEALTH_MODEL_ID):
        remote = _fetch_remote_revision(HEALTH_MODEL_ID)
        local = _read_revision(HEALTH_MODEL_REVISION_FILE)
        if remote:
            target_revision = remote
            if remote != local:
                should_reload = True
                logger.info("Healthcare model update: %s -> %s", local or "none", remote)

    if not should_reload:
        return False

    if not reload_lock.acquire(blocking=False):
        logger.info("Healthcare model reload is already running; skipping this cycle.")
        return False

    try:
        logger.info("Reloading HealthAnalyzerService with latest model revision...")
        new_service = HealthAnalyzerService(model_revision=target_revision)
        global analyzer_service
        with analyzer_lock:
            analyzer_service = new_service

        if target_revision:
            _write_revision(HEALTH_MODEL_REVISION_FILE, target_revision)
        logger.info("HealthAnalyzerService model reload completed.")
        return True
    except Exception:
        logger.exception("Healthcare model reload failed.")
        return False
    finally:
        reload_lock.release()


def _persist_current_revision_after_startup() -> None:
    """Persist known remote revision to avoid an immediate first-cycle reload."""
    if not CHECK_MODEL_UPDATE_ON_START:
        return

    if _repo_like(HEALTH_MODEL_ID):
        remote = _fetch_remote_revision(HEALTH_MODEL_ID)
        if remote:
            _write_revision(HEALTH_MODEL_REVISION_FILE, remote)


async def _background_model_update_checker(stop_event: asyncio.Event):
    interval = MODEL_UPDATE_CHECK_INTERVAL_SECONDS
    logger.info("Healthcare model update checker started (interval=%ss)", interval)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        try:
            await run_in_threadpool(_reload_if_model_updated)
        except Exception:
            logger.exception("Healthcare model update check failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer_service, job_queue
    stop_event = asyncio.Event()
    checker_task = None
    logger.info("Initializing HealthAnalyzerService...")
    try:
        analyzer_service = HealthAnalyzerService()
        _persist_current_revision_after_startup()
        logger.info("HealthAnalyzerService initialized.")
    except Exception as e:
        logger.error("Failed to initialize analyzer: %s", e)
        raise
    if MODEL_UPDATE_CHECK_INTERVAL_SECONDS > 0:
        checker_task = asyncio.create_task(_background_model_update_checker(stop_event))
    else:
        logger.info("Healthcare model update checker disabled (interval<=0)")

    if ASYNC_JOB_MODE_ENABLED:
        job_queue = HealthcareJobQueue(
            get_analyzer_service=lambda: analyzer_service,
            max_queue_size=max(1, ASYNC_JOB_QUEUE_MAX_SIZE),
        )
        await job_queue.start()
        logger.info("Async job mode enabled.")
    else:
        logger.info("Async job mode disabled.")

    yield

    if job_queue:
        await job_queue.stop()
    if checker_task:
        stop_event.set()
        await checker_task
    logger.info("Healthcare service shutting down.")


app = FastAPI(title="Healthcare Analysis Service", lifespan=lifespan)


@app.post("/analyze", response_model=HealthAnalyzeResponse)
async def analyze(req: HealthAnalyzeRequest) -> HealthAnalyzeResponse:
    if ASYNC_JOB_MODE_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="ASYNC_JOB_MODE_ENABLED=true. Use POST /jobs and GET /jobs/{job_id}.",
        )

    with analyzer_lock:
        current_service = analyzer_service

    if not current_service:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    try:
        # analysis may use heavy CPU, execute in thread pool
        return await run_in_threadpool(current_service.analyze, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Healthcare analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs", response_model=HealthJobCreateResponse)
async def create_job(req: HealthAnalyzeRequest) -> HealthJobCreateResponse:
    if not ASYNC_JOB_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Async job mode is disabled.")
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue is not initialized.")
    try:
        return await job_queue.enqueue(req)
    except RuntimeError as exc:
        if str(exc) == "JOB_QUEUE_FULL":
            raise HTTPException(
                status_code=429,
                detail="Job queue is full. Please retry shortly.",
            )
        raise


@app.get("/jobs/{job_id}", response_model=HealthJobStatusResponse)
async def get_job(job_id: str) -> HealthJobStatusResponse:
    if not ASYNC_JOB_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Async job mode is disabled.")
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue is not initialized.")

    status = job_queue.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return status


@app.get("/health")
def health():
    return {
        "status": "ok",
        "analyzer_loaded": analyzer_service is not None,
        "async_job_mode_enabled": ASYNC_JOB_MODE_ENABLED,
        "job_queue_initialized": job_queue is not None if ASYNC_JOB_MODE_ENABLED else False,
    }
