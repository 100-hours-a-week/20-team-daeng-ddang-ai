# app/core/config.py
import os

# 디버그 모드
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 챗봇 모델 설정
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_PATH  = os.getenv("ADAPTER_PATH",  "models/Qwen2.5-7B/7B-LoRA")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "models/chroma_db")

# 임베딩/검색 설정
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "jhgan/ko-sroberta-multitask")
EMBEDDING_NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes", "on"}
RAG_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "3"))
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")

# 생성 설정
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "384"))
GEN_REPETITION_PENALTY = float(os.getenv("GEN_REPETITION_PENALTY", "1.08"))

# 서버 설정
PORT = int(os.getenv("PORT", 8300))
MODEL_UPDATE_CHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_UPDATE_CHECK_INTERVAL_SECONDS", "86400"))
