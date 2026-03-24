from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import requests
from starlette.concurrency import run_in_threadpool

from app.core.config import DEBUG_MODE
from app.schemas.multimodal_schema import MultimodalAnalyzeRequest, MultimodalAnalyzeResponse
from app.services.multimodal_analyzer import MultimodalAnalyzerService

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multimodal_server")

analyzer_service: MultimodalAnalyzerService | None = None
analyzer_lock = threading.RLock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer_service
    logger.info("Initializing MultimodalAnalyzerService...")
    analyzer_service = MultimodalAnalyzerService()
    logger.info("MultimodalAnalyzerService initialized.")
    yield
    logger.info("Multimodal service shutting down.")


app = FastAPI(title="Multimodal Service", version="1.0.0", lifespan=lifespan)


@app.post("/analyze", response_model=MultimodalAnalyzeResponse)
async def analyze(req: MultimodalAnalyzeRequest) -> MultimodalAnalyzeResponse:
    with analyzer_lock:
        current_service = analyzer_service

    if current_service is None:
        raise HTTPException(status_code=503, detail="Multimodal analyzer not initialized")

    try:
        return await run_in_threadpool(current_service.analyze, req)
    except requests.HTTPError as exc:  # type: ignore[name-defined]
        status_code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status_code, detail=f"Image fetch failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Multimodal analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "analyzer_loaded": analyzer_service is not None}
