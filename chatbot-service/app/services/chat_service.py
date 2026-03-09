# app/services/chat_service.py
from __future__ import annotations

import datetime
import os
import time
import logging
import threading
from typing import Optional

from huggingface_hub import HfApi

from app.core.config import (
    BASE_MODEL_ID,
    ADAPTER_PATH,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_ID,
    EMBEDDING_NORMALIZE,
    RAG_RETRIEVAL_K,
    RAG_FINAL_TOP_K,
    RAG_RERANK_ENABLED,
    RERANKER_MODEL_ID,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_MAX_NEW_TOKENS,
    GEN_REPETITION_PENALTY,
    DEBUG,
)
from app.schemas.chat_schema import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)

# 싱글턴 - 앱 시작 시 한 번만 초기화
_chatbot_engine = None
_engine_lock = threading.RLock()
_reload_lock = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_local_revision() -> Optional[str]:
    local_dir = os.getenv("CHATBOT_ASSETS_LOCAL_DIR", "models")
    revision_file = os.getenv("MODEL_REVISION_FILE", os.path.join(local_dir, ".vet_chat_revision"))
    if not os.path.isfile(revision_file):
        return None
    try:
        with open(revision_file, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        return revision or None
    except OSError as e:
        logger.warning(f"로컬 revision 파일 읽기 실패: {e}")
        return None


def _get_remote_revision() -> Optional[str]:
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("HUGGING_FACE_HUB_TOKEN이 없어 원격 모델 revision 확인을 건너뜁니다.")
        return None

    repo_id = os.getenv("CHATBOT_ASSETS_REPO_ID", "20-team-daeng-ddang-ai/vet-chat")
    try:
        model_info = HfApi(token=hf_token).model_info(repo_id=repo_id, revision="main")
        return model_info.sha
    except Exception as e:
        logger.warning(f"원격 모델 revision 조회 실패(repo={repo_id}): {e}")
        return None


def _create_engine():
    from scripts.chatbot_core import VetChatbotCore
    return VetChatbotCore(
        base_model_id=BASE_MODEL_ID,
        adapter_path=ADAPTER_PATH,
        chroma_db_dir=CHROMA_DB_DIR,
        embedding_model_id=EMBEDDING_MODEL_ID,
        embedding_normalize=EMBEDDING_NORMALIZE,
        retrieval_k=RAG_RETRIEVAL_K,
        final_top_k=RAG_FINAL_TOP_K,
        rerank_enabled=RAG_RERANK_ENABLED,
        reranker_model_id=RERANKER_MODEL_ID,
        gen_temperature=GEN_TEMPERATURE,
        gen_top_p=GEN_TOP_P,
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        gen_repetition_penalty=GEN_REPETITION_PENALTY,
    )


def initialize_engine():
    """FastAPI lifespan에서 호출. VetChatbotCore를 메모리에 로드."""
    global _chatbot_engine
    logger.info(f"VetChatbotCore 초기화 시작 (base={BASE_MODEL_ID}, adapter={ADAPTER_PATH})")
    try:
        new_engine = _create_engine()
        with _engine_lock:
            _chatbot_engine = new_engine
        logger.info("✅ VetChatbotCore 초기화 완료")
    except Exception as e:
        logger.error(f"❌ VetChatbotCore 초기화 실패: {e}")
        raise e


def get_engine():
    with _engine_lock:
        return _chatbot_engine


def reload_engine_if_model_updated() -> bool:
    """
    원격 HF 모델 revision 변화를 확인하고, 변경 시 엔진을 재생성/스왑합니다.
    """
    if not _env_bool("CHECK_MODEL_UPDATE_ON_START", True):
        return False

    force_refresh = _env_bool("FORCE_REFRESH_MODELS", False)
    remote_revision = _get_remote_revision()
    local_revision = _read_local_revision()
    if not force_refresh and (not remote_revision or remote_revision == local_revision):
        return False

    if not _reload_lock.acquire(blocking=False):
        logger.info("모델 리로드가 이미 진행 중입니다. 이번 체크는 건너뜁니다.")
        return False

    try:
        if force_refresh:
            logger.info("FORCE_REFRESH_MODELS=true 설정으로 모델 리로드를 수행합니다.")
        else:
            logger.info(f"모델 업데이트 감지: revision {local_revision or 'none'} -> {remote_revision}")

        new_engine = _create_engine()
        global _chatbot_engine
        with _engine_lock:
            _chatbot_engine = new_engine
        logger.info("✅ 모델 업데이트 적용 완료 (엔진 스왑 성공)")
        return True
    except Exception:
        logger.exception("❌ 모델 업데이트 적용 실패")
        return False
    finally:
        _reload_lock.release()


def generate_chat_response(req: ChatRequest) -> ChatResponse:
    """
    VetChatbotCore.generate_answer()를 호출하고 ChatResponse 형태로 반환.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("ChatbotEngine이 초기화되지 않았습니다.")

    start_ms = time.time() * 1000

    # user_context dict 변환
    user_context_dict = {}
    if req.user_context:
        user_context_dict = {
            "dog_age_years": req.user_context.dog_age_years,
            "dog_weight_kg": req.user_context.dog_weight_kg,
            "breed":         req.user_context.breed,
        }

    # history list[dict] 변환
    history_list = [{"role": m.role, "content": m.content} for m in req.history]

    result = engine.generate_answer(
        message=req.message.content,
        user_context=user_context_dict,
        history=history_list,
    )

    elapsed_ms = int(time.time() * 1000 - start_ms)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    citations = [
        Citation(
            doc_id=c.get("doc_id", "unknown"),
            title=c.get("title", "수의학 지식"),
            score=float(c.get("score", 1.0)),
            snippet=c.get("snippet", ""),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        dog_id=req.dog_id,
        conversation_id=req.conversation_id,
        answered_at=now,
        answer=result["answer"],
        citations=citations,
        processing={
            "latency_ms": elapsed_ms,
            "model_used": BASE_MODEL_ID,
            "retrieval_k": RAG_RETRIEVAL_K,
            "final_top_k": RAG_FINAL_TOP_K,
            "rerank_enabled": RAG_RERANK_ENABLED,
        },
    )
