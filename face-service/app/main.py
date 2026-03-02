from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import logging

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.face_analyzer import FaceAnalyzer

from app.core.config import DEBUG

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("face_server")

# 전역 Analyzer 인스턴스 (앱 시작 시 초기화)
analyzer: FaceAnalyzer | None = None

# 앱 생명주기 관리 (Startup/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Initializing FaceAnalyzer...")
    try:
        # 모델 로드 및 초기화 (시간이 걸릴 수 있음)
        analyzer = FaceAnalyzer()
        logger.info("FaceAnalyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceAnalyzer: {e}")
        raise e
    yield
    # 종료 시 정리 작업
    logger.info("Face Server shutting down.")

app = FastAPI(title="Face Analysis Service", lifespan=lifespan)

# 얼굴 분석 엔드포인트
@app.post("/analyze", response_model=FaceAnalyzeResponse)
async def analyze_face(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    if not analyzer:
        raise HTTPException(status_code=503, detail="Face Analyzer not initialized")
    
    req_id = req.analysis_id or "req_unknown"
    try:
        # 분석 작업은 CPU/IO 집중적이므로 쓰레드 풀로 오프로드
        return await run_in_threadpool(analyzer.analyze, req_id, req)
    except Exception as e:
        logger.exception(f"Analysis failed for {req_id}")
        raise HTTPException(status_code=500, detail=str(e))

# 헬스 체크
@app.get("/health")
def health():
    return {"status": "ok", "analyzer_loaded": analyzer is not None}
