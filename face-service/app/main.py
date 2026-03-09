import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi
from starlette.concurrency import run_in_threadpool

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.face_analyzer import FaceAnalyzer

from app.core.config import (
    CHECK_MODEL_UPDATE_ON_START,
    DEBUG,
    FACE_DETECTION_MODEL_ID,
    FACE_DETECTION_REVISION_FILE,
    FACE_EMOTION_MODEL_ID,
    FACE_EMOTION_REVISION_FILE,
    FORCE_REFRESH_MODELS,
    HF_TOKEN,
    MODEL_UPDATE_CHECK_INTERVAL_SECONDS,
)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("face_server")

# 전역 Analyzer 인스턴스 (앱 시작 시 초기화)
analyzer: FaceAnalyzer | None = None
analyzer_lock = threading.RLock()
reload_lock = threading.Lock()


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


def _create_analyzer(
    detection_revision: Optional[str] = None,
    emotion_revision: Optional[str] = None,
) -> FaceAnalyzer:
    return FaceAnalyzer(
        detection_revision=detection_revision,
        emotion_revision=emotion_revision,
    )


def _reload_if_model_updated() -> bool:
    if not CHECK_MODEL_UPDATE_ON_START:
        return False

    target_detection_revision = None
    target_emotion_revision = None
    should_reload = FORCE_REFRESH_MODELS

    if _repo_like(FACE_DETECTION_MODEL_ID):
        remote = _fetch_remote_revision(FACE_DETECTION_MODEL_ID)
        local = _read_revision(FACE_DETECTION_REVISION_FILE)
        if remote:
            target_detection_revision = remote
            if remote != local:
                should_reload = True
                logger.info("Detection model update: %s -> %s", local or "none", remote)

    if _repo_like(FACE_EMOTION_MODEL_ID):
        remote = _fetch_remote_revision(FACE_EMOTION_MODEL_ID)
        local = _read_revision(FACE_EMOTION_REVISION_FILE)
        if remote:
            target_emotion_revision = remote
            if remote != local:
                should_reload = True
                logger.info("Emotion model update: %s -> %s", local or "none", remote)

    if not should_reload:
        return False

    if not reload_lock.acquire(blocking=False):
        logger.info("Face model reload is already running; skipping this cycle.")
        return False

    try:
        logger.info("Reloading FaceAnalyzer with latest model revisions...")
        new_analyzer = _create_analyzer(
            detection_revision=target_detection_revision,
            emotion_revision=target_emotion_revision,
        )
        global analyzer
        with analyzer_lock:
            analyzer = new_analyzer

        if target_detection_revision:
            _write_revision(FACE_DETECTION_REVISION_FILE, target_detection_revision)
        if target_emotion_revision:
            _write_revision(FACE_EMOTION_REVISION_FILE, target_emotion_revision)
        logger.info("FaceAnalyzer model reload completed.")
        return True
    except Exception:
        logger.exception("FaceAnalyzer model reload failed.")
        return False
    finally:
        reload_lock.release()


def _persist_current_revisions_after_startup() -> None:
    """Persist known remote revisions to avoid an immediate first-cycle reload."""
    if not CHECK_MODEL_UPDATE_ON_START:
        return

    if _repo_like(FACE_DETECTION_MODEL_ID):
        remote = _fetch_remote_revision(FACE_DETECTION_MODEL_ID)
        if remote:
            _write_revision(FACE_DETECTION_REVISION_FILE, remote)

    if _repo_like(FACE_EMOTION_MODEL_ID):
        remote = _fetch_remote_revision(FACE_EMOTION_MODEL_ID)
        if remote:
            _write_revision(FACE_EMOTION_REVISION_FILE, remote)


async def _background_model_update_checker(stop_event: asyncio.Event):
    interval = MODEL_UPDATE_CHECK_INTERVAL_SECONDS
    logger.info("Face model update checker started (interval=%ss)", interval)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        try:
            await run_in_threadpool(_reload_if_model_updated)
        except Exception:
            logger.exception("Face model update check failed.")

# 앱 생명주기 관리 (Startup/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    stop_event = asyncio.Event()
    checker_task = None
    logger.info("Initializing FaceAnalyzer...")
    try:
        # 모델 로드 및 초기화 (시간이 걸릴 수 있음)
        analyzer = _create_analyzer()
        _persist_current_revisions_after_startup()
        logger.info("FaceAnalyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceAnalyzer: {e}")
        raise e

    if MODEL_UPDATE_CHECK_INTERVAL_SECONDS > 0:
        checker_task = asyncio.create_task(_background_model_update_checker(stop_event))
    else:
        logger.info("Face model update checker disabled (interval<=0)")

    yield

    if checker_task:
        stop_event.set()
        await checker_task
    # 종료 시 정리 작업
    logger.info("Face Server shutting down.")

app = FastAPI(title="Face Analysis Service", lifespan=lifespan)

# 얼굴 분석 엔드포인트
@app.post("/analyze", response_model=FaceAnalyzeResponse)
async def analyze_face(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    with analyzer_lock:
        current_analyzer = analyzer

    if not current_analyzer:
        raise HTTPException(status_code=503, detail="Face Analyzer not initialized")
    
    req_id = req.analysis_id or "req_unknown"
    try:
        # 분석 작업은 CPU/IO 집중적이므로 쓰레드 풀로 오프로드
        return await run_in_threadpool(current_analyzer.analyze, req_id, req)
    except Exception as e:
        logger.exception(f"Analysis failed for {req_id}")
        raise HTTPException(status_code=500, detail=str(e))

# 헬스 체크
@app.get("/health")
def health():
    return {"status": "ok", "analyzer_loaded": analyzer is not None}
