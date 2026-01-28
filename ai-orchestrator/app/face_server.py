from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.adapters.face_local_adapter import FaceLocalAdapter

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_server")

# Global Adapter Instance
adapter: FaceLocalAdapter | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global adapter
    logger.info("Initializing FaceLocalAdapter...")
    try:
        adapter = FaceLocalAdapter()
        logger.info("FaceLocalAdapter initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceLocalAdapter: {e}")
        raise e
    yield
    # Cleanup if needed
    logger.info("Face Server shutting down.")

app = FastAPI(title="Face Analysis Service", lifespan=lifespan)

@app.post("/analyze", response_model=FaceAnalyzeResponse)
def analyze_face(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    if not adapter:
        raise HTTPException(status_code=503, detail="Face Adapter not initialized")
    
    # Generate a request ID if not provided, or use analysis_id
    req_id = req.analysis_id or "req_unknown" # Adapter expects a request_id for logging
    
    return adapter.analyze(req_id, req)
    
    try:
        return adapter.analyze(req_id, req)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail={
            "analysis_id": req.analysis_id,
            "status": "failed",
            "error": {
                "code": "FACE_NOT_DETECTED" if "FACE_NOT_DETECTED" in str(e) else "ANALYSIS_FAILED",
                "message": str(e)
            }
        })
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "adapter_loaded": adapter is not None}
