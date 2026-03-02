import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import DEBUG_MODE
from app.schemas.health_schema import HealthAnalyzeRequest, HealthAnalyzeResponse
from app.services.health_analyzer import HealthAnalyzerService

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthcare_server")

analyzer_service: HealthAnalyzerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer_service
    logger.info("Initializing HealthAnalyzerService...")
    try:
        analyzer_service = HealthAnalyzerService()
        logger.info("HealthAnalyzerService initialized.")
    except Exception as e:
        logger.error("Failed to initialize analyzer: %s", e)
        raise
    yield
    logger.info("Healthcare service shutting down.")


app = FastAPI(title="Healthcare Analysis Service", lifespan=lifespan)


@app.post("/analyze", response_model=HealthAnalyzeResponse)
async def analyze(req: HealthAnalyzeRequest) -> HealthAnalyzeResponse:
    if not analyzer_service:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    try:
        # analysis may use heavy CPU, execute in thread pool
        return await run_in_threadpool(analyzer_service.analyze, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Healthcare analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "analyzer_loaded": analyzer_service is not None}
