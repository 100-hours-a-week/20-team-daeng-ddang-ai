# app/main.py
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.routers.mission_router import router as mission_router
from app.routers.face_router import router as face_router

app = FastAPI(title="DaengDdang AI Orchestrator", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(mission_router)
app.include_router(face_router)