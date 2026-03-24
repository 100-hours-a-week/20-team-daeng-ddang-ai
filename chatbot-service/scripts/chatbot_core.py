import os
import re
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
        rag_enabled: bool = True,
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
        self.rag_enabled = rag_enabled
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
        self.vectorstore = None
        self.retriever = None
        self.reranker = None

        self._initialize()

    @staticmethod
    def _prepare_query_text(message: str, embedding_model_id: str) -> str:
        message = (message or "").strip()
        if not message:
            return ""
        if "e5" in embedding_model_id.lower() and not message.lower().startswith("query:"):
            return f"query: {message}"
        return message

    @staticmethod
    def _clean_context_text(text: str) -> str:
        if not text:
            return ""

        cleaned_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in line)
            uppercase_ascii = sum(ch.isascii() and ch.isupper() for ch in line)
            if ascii_letters >= 8 and uppercase_ascii / max(ascii_letters, 1) >= 0.8:
                continue

            line = re.sub(r"\s+", " ", line)
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

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
        allow_patterns = []
        chroma_dir_name = os.path.basename(os.path.normpath(self.chroma_db_dir)) or "chroma_db"
        if self.rag_enabled:
            allow_patterns.append(f"{chroma_dir_name}/*")

        if self.force_refresh_models:
            reasons.append("FORCE_REFRESH_MODELS=true")
        if self.rag_enabled and not self._dir_has_files(self.chroma_db_dir):
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

        if self.rag_enabled:
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
            self.vectorstore = vectorstore
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k})

            if self.rerank_enabled:
                try:
                    self.reranker = CrossEncoder(self.reranker_model_id)
                    print(f"✅ Reranker loaded: {self.reranker_model_id}")
                except Exception as e:
                    self.reranker = None
                    print(f"⚠️ Failed to load reranker '{self.reranker_model_id}': {e}. Continue without reranking.")
        else:
            self.vectorstore = None
            self.retriever = None
            self.reranker = None
            print("ℹ️ RAG disabled. Skipping Vector DB and reranker initialization.")
        print("✅ Core initialization complete.")

    def _retrieve_docs(self, retrieval_source: str, image_priority_terms: List[str]) -> List[object]:
        if self.retriever is None:
            return []

        queries: List[str] = [retrieval_source]
        for term in (image_priority_terms or [])[:3]:
            term = (term or "").strip()
            if not term:
                continue
            queries.append(f"{term} 반려견 안과 질환")

        seen_keys: set[str] = set()
        merged_docs: List[object] = []
        for raw_query in queries:
            retrieval_query = self._prepare_query_text(raw_query, self.embedding_model_id)
            if self.vectorstore is not None and hasattr(self.vectorstore, "similarity_search"):
                per_query_k = max(self.retrieval_k * 3, 10)
                docs = self.vectorstore.similarity_search(retrieval_query, k=per_query_k)
            else:
                docs = self.retriever.invoke(retrieval_query)
            for doc in docs:
                doc_id = str(doc.metadata.get("id", "")).strip()
                key = doc_id or getattr(doc, "page_content", "")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_docs.append(doc)
        return merged_docs

    def _rerank_docs(self, message: str, docs) -> List[Tuple[object, float]]:
        if not docs:
            return []
        if not self.rerank_enabled or self.reranker is None:
            return [(doc, 1.0) for doc in docs]

        pairs = [(message, doc.page_content) for doc in docs]
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            doc_scores = list(zip(docs, [float(s) for s in scores]))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores
        except Exception as e:
            print(f"⚠️ Rerank failed: {e}. Fallback to retrieval order.")
            return [(doc, 1.0) for doc in docs]

    @staticmethod
    def _prioritize_docs_by_terms(doc_scores: List[Tuple[object, float]], priority_terms: List[str]) -> List[Tuple[object, float]]:
        if not doc_scores or not priority_terms:
            return doc_scores

        normalized_terms = [term.strip().lower() for term in priority_terms if term and term.strip()]
        if not normalized_terms:
            return doc_scores

        boosted: List[Tuple[object, float, int]] = []
        for doc, score in doc_scores:
            title = str(doc.metadata.get("title", "")).lower()
            content = str(getattr(doc, "page_content", "")).lower()
            match_count = sum(1 for term in normalized_terms if term in title or term in content)
            boosted.append((doc, score, match_count))

        boosted.sort(key=lambda item: (item[2], item[1]), reverse=True)
        return [(doc, score) for doc, score, _ in boosted]

    @staticmethod
    def _select_diverse_docs(
        doc_scores: List[Tuple[object, float]],
        priority_terms: List[str],
        limit: int,
    ) -> List[Tuple[object, float]]:
        if not doc_scores:
            return []

        normalized_terms = [term.strip().lower() for term in priority_terms if term and term.strip()]
        selected: List[Tuple[object, float]] = []
        selected_keys: set[str] = set()

        def doc_key(doc: object) -> str:
            metadata = getattr(doc, "metadata", {}) or {}
            return str(metadata.get("id", "")).strip() or str(getattr(doc, "page_content", ""))

        def doc_text(doc: object) -> str:
            metadata = getattr(doc, "metadata", {}) or {}
            title = str(metadata.get("title", "")).lower()
            content = str(getattr(doc, "page_content", "")).lower()
            return f"{title}\n{content}"

        for term in normalized_terms:
            for doc, score in doc_scores:
                key = doc_key(doc)
                if key in selected_keys:
                    continue
                if term in doc_text(doc):
                    selected.append((doc, score))
                    selected_keys.add(key)
                    break
            if len(selected) >= limit:
                return selected[:limit]

        for doc, score in doc_scores:
            key = doc_key(doc)
            if key in selected_keys:
                continue
            selected.append((doc, score))
            selected_keys.add(key)
            if len(selected) >= limit:
                break

        return selected[:limit]

    @staticmethod
    def _sanitize_answer_text(text: str) -> str:
        original = (text or "").strip()
        cleaned = original
        if not cleaned:
            return cleaned

        leak_markers = [
            "여기需要保持中文回答",
            "这里需要保持中文回答",
            "请继续",
            "中文回答",
        ]
        cut_index = len(cleaned)

        for marker in leak_markers:
            idx = cleaned.find(marker)
            if idx >= 0:
                cut_index = min(cut_index, idx)

        broken_idx = cleaned.find("�")
        if broken_idx >= 0:
            cut_index = min(cut_index, broken_idx)

        chinese_match = re.search(r"[\u4e00-\u9fff]{2,}", cleaned)
        if chinese_match:
            cut_index = min(cut_index, chinese_match.start())

        cleaned = cleaned[:cut_index].rstrip()
        if cut_index < len(original):
            last_sentence_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
            if last_sentence_end >= 0:
                cleaned = cleaned[: last_sentence_end + 1]
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return "정확한 평가는 동물병원 안과 진료를 통해 확인하는 것이 좋습니다."

        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)

        if cleaned[-1] not in ".!?다요":
            cleaned = cleaned.rstrip(":;,") + "."

        if "동물병원" not in cleaned and "수의사" not in cleaned:
            cleaned += " 정확한 진단은 동물병원에서 확인하는 것이 좋습니다."

        return cleaned

    def generate_answer(
        self,
        message: str,
        user_context: dict,
        history: list,
        image_analysis_note: str = "",
        image_retrieval_hint: str = "",
        image_priority_terms: List[str] | None = None,
    ):
        """
        Generate a response based on RAG context, user history, and dog profile.
        
        Args:
            message (str): Current user question
            user_context (dict): {"dog_age_years": 5, "dog_weight_kg": 4.5, "breed": "Maltese"}
            history (list): [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            dict: { "answer": str, "citations": list of dicts }
        """
        context_text = ""
        citations = []
        if self.rag_enabled and self.retriever is not None:
            retrieval_source = message
            if image_retrieval_hint:
                retrieval_source = f"{message}\n이미지 단서: {image_retrieval_hint}"

            docs = self._retrieve_docs(retrieval_source, image_priority_terms or [])
            ranked_docs = self._rerank_docs(retrieval_source, docs)
            ranked_docs = self._prioritize_docs_by_terms(ranked_docs, image_priority_terms or [])
            ranked_docs = self._select_diverse_docs(
                ranked_docs,
                image_priority_terms or [],
                self.final_top_k,
            )

            for idx, (doc, score) in enumerate(ranked_docs):
                cleaned_content = self._clean_context_text(doc.page_content)
                context_text += f"[근거 자료 {idx+1}]\n{cleaned_content}\n\n"
                citations.append({
                    "doc_id": doc.metadata.get("id", f"doc_{idx}"),
                    "title": doc.metadata.get("title", "수집된 수의학 지식"),
                    "score": round(float(score), 4),
                    "snippet": cleaned_content[:100] + "..."
                })

        # 2. 강아지 프로필 컨텍스트 문자열 생성
        profile_str = ""
        if user_context:
            age = user_context.get("dog_age_years", "알수없음")
            weight = user_context.get("dog_weight_kg", "알수없음")
            breed = user_context.get("breed", "알수없음")
            profile_str = f"- 견종: {breed}\n- 나이: {age}살\n- 체중: {weight}kg\n"
        
        image_analysis_str = ""
        if image_analysis_note:
            image_analysis_str = f"[이미지 분석 보조 정보]\n{image_analysis_note}\n"

        answer_format_str = (
            "[답변 형식]\n"
            "번호 목록 대신 자연스러운 1~2개 짧은 문단으로 답변하세요.\n"
            "가장 가능성이 높은 질환과 다른 가능성을 1~2문장으로 먼저 요약하세요.\n"
            "가능성이 있는 질환이 2개 이상이면 각 질환마다 왜 고려하는지 또는 대표 증상을 1문장씩 덧붙이세요.\n"
            "설명은 짧고 자연스러운 한국어 문단으로 작성하고, 불필요한 장황함은 피하세요.\n"
            "마지막에는 확진이 아니며 필요 시 동물병원 진료가 필요하다는 점을 1문장으로 안내하세요.\n"
        )

        # 3. 고도화된 시스템 프롬프트 작성
        # 답변의 톤앤매너, 환각 방지, 유저 컨텍스트를 종합적으로 지시합니다.
        if self.rag_enabled:
            system_prompt = (
                "당신은 따뜻하고 전문적인 수의학 AI 챗봇입니다.\n"
                "아래 제공된 [환자 정보], [이미지 분석 보조 정보], [참고 문서]만을 바탕으로 사용자의 질문에 답변하세요.\n"
                "답변은 핵심을 짚어 간결하고 친절한 한국어로 작성하며, 참고 문서에 없는 내용을 절대 지어내지 마세요.\n"
                "반드시 자연스러운 한국어로만 답변하고, 중국어·영어·일본어 등 다른 언어를 섞지 마세요.\n"
                "응급 내원 권고는 호흡곤란, 반복 구토, 혈변/흑변, 의식 저하, 경련, 복부 팽만, 보행 불능, 심한 통증처럼 명확한 위험 신호가 참고 문서에 있을 때만 하세요.\n"
                "그 외에는 집에서 관찰할 점과 일반 내원 또는 예약 내원 권고를 우선 제시하세요.\n"
                "참고 문서에 근거가 약하면 단정적으로 말하지 말고 가능성을 나눠 설명하세요.\n"
                "이미지 분석 결과가 있더라도 보조 참고 정보로만 활용하고 최종 진단처럼 단정하지 마세요.\n"
                "참고 문서의 제목, 섹션 헤더, 영어 문구를 답변 첫머리에 그대로 복사하지 마세요.\n"
                "가능성이 있는 질환이 여러 개면 각 질환에 대해 왜 고려하는지 또는 대표 증상을 짧게 설명하되, 참고 문서에 없는 세부사항은 덧붙이지 마세요."
            )
            user_prompt = (
                f"[환자 정보]\n{profile_str}\n"
                f"{image_analysis_str}"
                f"{answer_format_str}"
                f"[참고 문서]\n{context_text}\n"
                f"[사용자 질문]\n{message}"
            )
        else:
            system_prompt = (
                "당신은 따뜻하고 전문적인 수의학 AI 챗봇입니다.\n"
                "아래 제공된 [환자 정보]와 [이미지 분석 보조 정보]를 참고하여 사용자의 질문에 답변하세요.\n"
                "답변은 핵심을 짚어 간결하고 친절한 한국어로 작성하세요.\n"
                "반드시 자연스러운 한국어로만 답변하고, 중국어·영어·일본어 등 다른 언어를 섞지 마세요.\n"
                "근거가 불충분한 내용은 단정적으로 말하지 말고 일반적인 가능성으로 설명하세요.\n"
                "응급 내원 권고는 호흡곤란, 반복 구토, 혈변/흑변, 의식 저하, 경련, 복부 팽만, 보행 불능, 심한 통증처럼 명확한 위험 신호가 있을 때만 하세요.\n"
                "그 외에는 집에서 관찰할 점과 일반 내원 또는 예약 내원 권고를 우선 제시하세요.\n"
                "이미지 분석 결과가 있더라도 보조 참고 정보로만 활용하고 최종 진단처럼 단정하지 마세요.\n"
                "가능성이 있는 질환이 여러 개면 각 질환에 대해 왜 고려하는지 또는 대표 증상을 짧게 설명하세요."
            )
            user_prompt = (
                f"[환자 정보]\n{profile_str}\n"
                f"{image_analysis_str}"
                f"{answer_format_str}"
                f"[사용자 질문]\n{message}"
            )

        # 4. History 기반 대화 템플릿 구성
        messages = [{"role": "system", "content": system_prompt}]
        
        # 최대 4개(최근 2번의 턴)의 이전 대화만 포함하여 문맥 유지 및 VRAM 절약
        for past_msg in history[-4:]:
            messages.append(past_msg)
            
        messages.append({"role": "user", "content": user_prompt})

        # 5. 모델 추론
        if self.generation_client is None:
            raise RuntimeError("Generation client is not initialized.")
        response_text = self._sanitize_answer_text(self.generation_client.generate(messages))

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
