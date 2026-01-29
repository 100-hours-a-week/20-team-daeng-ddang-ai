from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse
from app.services.face_analyzer import FaceAnalyzer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_server")

# Global Analyzer Instance
analyzer: FaceAnalyzer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Initializing FaceAnalyzer...")
    try:
        analyzer = FaceAnalyzer()
        logger.info("FaceAnalyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceAnalyzer: {e}")
        raise e
    yield
    # Cleanup if needed
    logger.info("Face Server shutting down.")

app = FastAPI(title="Face Analysis Service", lifespan=lifespan)

@app.post("/analyze", response_model=FaceAnalyzeResponse)
def analyze_face(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    if not analyzer:
        raise HTTPException(status_code=503, detail="Face Analyzer not initialized")
    
    # Generate a request ID if not provided, or use analysis_id
    req_id = req.analysis_id or "req_unknown" 
    
        return analyzer.analyze(req_id, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "analyzer_loaded": analyzer is not None}
