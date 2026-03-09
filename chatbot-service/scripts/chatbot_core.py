import os
from typing import Optional, List, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import HfApi, snapshot_download
from sentence_transformers import CrossEncoder

from app.services.generation import LocalGenerationClient, VllmGenerationClient

class VetChatbotCore:
    def __init__(
        self,
        llm_backend: str = "local",
        base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: str = "models/Qwen2.5-7B/7B-LoRA",
        chroma_db_dir: str = "models/chroma_db",
        embedding_model_id: str = "jhgan/ko-sroberta-multitask",
        embedding_normalize: bool = True,
        retrieval_k: int = 5,
        final_top_k: int = 3,
        rerank_enabled: bool = True,
        reranker_model_id: str = "BAAI/bge-reranker-v2-m3",
        gen_temperature: float = 0.1,
        gen_top_p: float = 0.9,
        gen_max_new_tokens: int = 384,
        gen_repetition_penalty: float = 1.08,
        vllm_base_url: str = "http://localhost:8400",
        vllm_model_name: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        vllm_timeout_seconds: float = 120,
    ):
        self.llm_backend = llm_backend.lower()
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model_id = embedding_model_id
        self.embedding_normalize = embedding_normalize
        self.retrieval_k = max(1, retrieval_k)
        self.final_top_k = max(1, min(final_top_k, self.retrieval_k))
        self.rerank_enabled = rerank_enabled
        self.reranker_model_id = reranker_model_id
        self.gen_temperature = max(0.0, gen_temperature)
        self.gen_top_p = min(max(gen_top_p, 0.0), 1.0)
        self.gen_max_new_tokens = max(64, gen_max_new_tokens)
        self.gen_repetition_penalty = max(1.0, gen_repetition_penalty)
        self.vllm_base_url = vllm_base_url
        self.vllm_model_name = vllm_model_name or base_model_id
        self.vllm_api_key = vllm_api_key
        self.vllm_timeout_seconds = max(1.0, vllm_timeout_seconds)
        self.assets_repo_id = os.getenv("CHATBOT_ASSETS_REPO_ID", "20-team-daeng-ddang-ai/vet-chat")
        self.assets_local_dir = os.getenv("CHATBOT_ASSETS_LOCAL_DIR", "models")
        self.check_model_update = self._env_bool("CHECK_MODEL_UPDATE_ON_START", True)
        self.force_refresh_models = self._env_bool("FORCE_REFRESH_MODELS", False)
        self.revision_file = os.getenv("MODEL_REVISION_FILE", os.path.join(self.assets_local_dir, ".vet_chat_revision"))

        self.generation_client = None
        self.retriever = None
        self.reranker = None

        self._initialize()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        return os.getenv(name, str(default).lower()).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _dir_has_files(path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    def _resolve_local_adapter_path(self) -> str:
        """
        Normalize adapter path to local directory when env points to HF subpath.
        Example:
          20-team-daeng-ddang-ai/vet-chat/Qwen2.5-7B/7B-LoRA -> models/Qwen2.5-7B/7B-LoRA
        """
        if os.path.exists(self.adapter_path):
            return self.adapter_path

        hf_subpath_prefix = "20-team-daeng-ddang-ai/vet-chat/"
        if self.adapter_path.startswith(hf_subpath_prefix):
            local_candidate = os.path.join("models", self.adapter_path[len(hf_subpath_prefix):])
            if os.path.exists(local_candidate):
                print(f"ℹ️ Adapter path remapped to local path: {local_candidate}")
                return local_candidate
        return self.adapter_path

    def _read_local_revision(self) -> Optional[str]:
        if not os.path.isfile(self.revision_file):
            return None
        try:
            with open(self.revision_file, "r", encoding="utf-8") as f:
                revision = f.read().strip()
            return revision or None
        except OSError as e:
            print(f"⚠️ Failed to read revision file '{self.revision_file}': {e}")
            return None

    def _write_local_revision(self, revision: str) -> None:
        try:
            revision_dir = os.path.dirname(self.revision_file)
            if revision_dir:
                os.makedirs(revision_dir, exist_ok=True)
            with open(self.revision_file, "w", encoding="utf-8") as f:
                f.write(revision)
        except OSError as e:
            print(f"⚠️ Failed to write revision file '{self.revision_file}': {e}")

    def _get_remote_revision(self, hf_token: Optional[str]) -> Optional[str]:
        try:
            model_info = HfApi(token=hf_token).model_info(repo_id=self.assets_repo_id, revision="main")
            return model_info.sha
        except Exception as e:
            print(f"⚠️ Failed to fetch remote model revision from HF: {e}")
            return None

    def _download_hf_assets(self, hf_token):
        """HuggingFace Private 리포지토리에서 chroma_db와 LoRA 가중치를 로컬 models/ 폴더로 다운로드"""
        needs_download = self.force_refresh_models
        target_revision = None
        reasons = []
        allow_patterns = ["chroma_db/*"]

        if self.force_refresh_models:
            reasons.append("FORCE_REFRESH_MODELS=true")
        if not self._dir_has_files(self.chroma_db_dir):
            print(f"⚠️ Chroma DB not found at {self.chroma_db_dir}.")
            needs_download = True
            reasons.append("missing chroma_db")
        if self.llm_backend == "local":
            allow_patterns.append("Qwen2.5-7B/7B-LoRA/*")
            local_adapter_path = self._resolve_local_adapter_path()
            if local_adapter_path and not self._dir_has_files(local_adapter_path):
                print(f"⚠️ LoRA Adapter not found at {self.adapter_path}.")
                needs_download = True
                reasons.append("missing LoRA adapter")

        if self.check_model_update:
            remote_revision = self._get_remote_revision(hf_token)
            local_revision = self._read_local_revision()
            if remote_revision:
                target_revision = remote_revision
                if remote_revision != local_revision:
                    needs_download = True
                    reasons.append(f"revision changed ({local_revision or 'none'} -> {remote_revision})")
            else:
                print("⚠️ Remote revision check skipped due to HF metadata fetch failure.")

        if needs_download:
            if not hf_token:
                raise RuntimeError("HUGGING_FACE_HUB_TOKEN is required to download chatbot assets.")
            reason_text = ", ".join(reasons) if reasons else "unknown reason"
            print(f"📥 Downloading required assets ({reason_text}) from {self.assets_repo_id} into '{self.assets_local_dir}/'...")
            snapshot_download(
                repo_id=self.assets_repo_id,
                allow_patterns=allow_patterns,
                local_dir=self.assets_local_dir,
                revision=target_revision,
                token=hf_token
            )
            print("✅ Download complete.")
            if target_revision:
                self._write_local_revision(target_revision)
        else:
            print("✅ Local assets already present. Skipping HF snapshot download.")

    def _initialize(self):
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        # 모델 구동 전 허깅페이스에서 로컬 파일(models/)로 필요한 자산을 다운로드
        self._download_hf_assets(hf_token)
        if self.llm_backend == "local":
            self.adapter_path = self._resolve_local_adapter_path()
            print(f"Loading local generation model... (Base: {self.base_model_id}, Adapter: {self.adapter_path})")
            try:
                self.generation_client = LocalGenerationClient(
                    base_model_id=self.base_model_id,
                    adapter_path=self.adapter_path if os.path.exists(self.adapter_path) else "",
                    hf_token=hf_token,
                    gen_temperature=self.gen_temperature,
                    gen_top_p=self.gen_top_p,
                    gen_max_new_tokens=self.gen_max_new_tokens,
                    gen_repetition_penalty=self.gen_repetition_penalty,
                )
                print("✅ Local generation backend loaded.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize local generation backend: {e}") from e
        elif self.llm_backend == "vllm":
            self.generation_client = VllmGenerationClient(
                base_url=self.vllm_base_url,
                model_name=self.vllm_model_name,
                timeout_seconds=self.vllm_timeout_seconds,
                api_key=self.vllm_api_key,
                gen_temperature=self.gen_temperature,
                gen_top_p=self.gen_top_p,
                gen_max_new_tokens=self.gen_max_new_tokens,
                gen_repetition_penalty=self.gen_repetition_penalty,
            )
            print(f"✅ vLLM generation backend configured: {self.vllm_base_url} (model={self.vllm_model_name})")
        else:
            raise ValueError(f"Unsupported llm backend: {self.llm_backend}")

        print(
            f"Loading Vector DB... (embedding={self.embedding_model_id}, normalize={self.embedding_normalize}, "
            f"retrieval_k={self.retrieval_k}, final_top_k={self.final_top_k})"
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_id,
            encode_kwargs={"normalize_embeddings": self.embedding_normalize},
        )
        vectorstore = Chroma(
            persist_directory=self.chroma_db_dir, 
            embedding_function=embeddings,
            collection_name="vet_qa_collection"
        )
        # return_source_documents is automatically handled when we access metadata if we retrieve directly.
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k})

        if self.rerank_enabled:
            try:
                self.reranker = CrossEncoder(self.reranker_model_id)
                print(f"✅ Reranker loaded: {self.reranker_model_id}")
            except Exception as e:
                self.reranker = None
                print(f"⚠️ Failed to load reranker '{self.reranker_model_id}': {e}. Continue without reranking.")
        print("✅ Core initialization complete.")

    def _rerank_docs(self, message: str, docs) -> List[Tuple[object, float]]:
        if not docs:
            return []
        if not self.rerank_enabled or self.reranker is None:
            return [(doc, 1.0) for doc in docs[: self.final_top_k]]

        pairs = [(message, doc.page_content) for doc in docs]
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            doc_scores = list(zip(docs, [float(s) for s in scores]))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[: self.final_top_k]
        except Exception as e:
            print(f"⚠️ Rerank failed: {e}. Fallback to retrieval order.")
            return [(doc, 1.0) for doc in docs[: self.final_top_k]]

    def generate_answer(self, message: str, user_context: dict, history: list):
        """
        Generate a response based on RAG context, user history, and dog profile.
        
        Args:
            message (str): Current user question
            user_context (dict): {"dog_age_years": 5, "dog_weight_kg": 4.5, "breed": "Maltese"}
            history (list): [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            dict: { "answer": str, "citations": list of dicts }
        """
        # 1. RAG 기반 문서 검색
        docs = self.retriever.invoke(message)
        ranked_docs = self._rerank_docs(message, docs)
        
        context_text = ""
        citations = []
        for idx, (doc, score) in enumerate(ranked_docs):
            context_text += f"[근거 자료 {idx+1}]\n{doc.page_content}\n\n"
            citations.append({
                "doc_id": doc.metadata.get("id", f"doc_{idx}"),
                "title": doc.metadata.get("title", "수집된 수의학 지식"),
                "score": round(float(score), 4),
                "snippet": doc.page_content[:100] + "..."
            })

        # 2. 강아지 프로필 컨텍스트 문자열 생성
        profile_str = ""
        if user_context:
            age = user_context.get("dog_age_years", "알수없음")
            weight = user_context.get("dog_weight_kg", "알수없음")
            breed = user_context.get("breed", "알수없음")
            profile_str = f"- 견종: {breed}\n- 나이: {age}살\n- 체중: {weight}kg\n"
        
        # 3. 고도화된 시스템 프롬프트 작성
        # 답변의 톤앤매너, 환각 방지, 유저 컨텍스트를 종합적으로 지시합니다.
        system_prompt = (
            "당신은 따뜻하고 전문적인 수의학 AI 챗봇입니다.\n"
            "아래 제공된 [환자 정보]와 [참고 문서]만을 바탕으로 사용자의 질문에 답변하세요.\n"
            "의학적 판단이 필요하거나 생명이 위급한 상황이라면, 반드시 '근처 동물병원에 내원하시라'는 권고를 포함하세요.\n"
            "답변은 핵심을 짚어 간결하고 친절한 한국어로 작성하며, 참고 문서에 없는 내용을 절대 지어내지 마세요."
        )

        user_prompt = f"[환자 정보]\n{profile_str}\n[참고 문서]\n{context_text}\n[사용자 질문]\n{message}"

        # 4. History 기반 대화 템플릿 구성
        messages = [{"role": "system", "content": system_prompt}]
        
        # 최대 4개(최근 2번의 턴)의 이전 대화만 포함하여 문맥 유지 및 VRAM 절약
        for past_msg in history[-4:]:
            messages.append(past_msg)
            
        messages.append({"role": "user", "content": user_prompt})

        # 5. 모델 추론
        if self.generation_client is None:
            raise RuntimeError("Generation client is not initialized.")
        response_text = self.generation_client.generate(messages)

        return {
            "answer": response_text,
            "citations": citations
        }

# Example usage (for testing)
if __name__ == "__main__":
    chatbot = VetChatbotCore(
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="../lora-qwen-7b-final", # local path for testing
        chroma_db_dir="../chroma_db"        # local path for testing
    )
    
    test_context = {"dog_age_years": 8, "dog_weight_kg": 3, "breed": "포메라니안"}
    test_history = [{"role": "user", "content": "강아지 예방접종은 언제 맞춰야 해?"}, {"role": "assistant", "content": "매년 맞추는 것이 좋습니다."}]
    test_msg = "우리 애기가 어제부터 노란 토를 해. 나이가 많아서 걱정인데 하루 굶길까?"
    
    res = chatbot.generate_answer(test_msg, test_context, test_history)
    print("\n[답변]:\n", res["answer"])
    print("\n[인용된 문서]:\n", res["citations"])
