# app/main.py
import os, hashlib, logging
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에서 읽어옴) -> 가장 먼저 실행되어야 함
load_dotenv()

from fastapi import FastAPI
from app.routers.mission_router import router as mission_router
from app.routers.face_router import router as face_router

# 로깅 설정 (시간, 레벨, 메시지 포맷 정의)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화 (타이틀과 버전 설정)
app = FastAPI(title="DaengDdang AI Orchestrator", version="0.1.0")

# 헬스 체크 엔드포인트 (LB나 모니터링 시스템에서 호출)
@app.get("/health")
def health():
    return {"status": "ok"}

# DEBUG 모드 확인 (기본값: False)
IS_DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 외부 라이브러리 로그 레벨 조정
if IS_DEBUG:
    # 디버그 모드일 때는 INFO 로그 허용
    logging.getLogger("google_genai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
else:
    # 운영 모드일 때는 WARNING 이상만 출력하여 로그 정리
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# API 키 확인 (디버그 모드이거나 키가 없을 때만 로그 출력)
k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if k:
    if IS_DEBUG:
        key_hash = hashlib.sha256(k.encode()).hexdigest()[:12]
        logger.info(f"[SERVER KeyHash] {key_hash}")
else:
    logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

# 라우터 등록 (기능별 API 라우트를 앱에 추가)
app.include_router(mission_router)
app.include_router(face_router)

# --- 라우트 정보 로깅 (디버그 모드일 때만 출력) ---
if IS_DEBUG:
    logger.info("[ROUTES] registered paths:")
    for r in app.routes:
        methods = getattr(r, "methods", None)
        if methods:
            logger.info(f"  {sorted(methods)}  {r.path}")
# ----------------------