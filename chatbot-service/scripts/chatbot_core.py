import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download

class VetChatbotCore:
    def __init__(self, base_model_id="Qwen/Qwen2.5-7B-Instruct", adapter_path="models/Qwen2.5-7B/7B-LoRA", chroma_db_dir="models/chroma_db"):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.chroma_db_dir = chroma_db_dir
        
        self.tokenizer = None
        self.model = None
        self.retriever = None
        
        self._initialize()

    def _download_hf_assets(self, hf_token):
        """HuggingFace Private 리포지토리에서 chroma_db와 LoRA 가중치를 로컬 models/ 폴더로 다운로드"""
        repo_id = "20-team-daeng-ddang-ai/vet-chat"
        local_dir = "models"
        
        needs_download = False
        if not os.path.exists(self.chroma_db_dir):
            print(f"⚠️ Chroma DB not found at {self.chroma_db_dir}.")
            needs_download = True
        if self.adapter_path and not os.path.exists(self.adapter_path):
            print(f"⚠️ LoRA Adapter not found at {self.adapter_path}.")
            needs_download = True
            
        if needs_download:
            print(f"📥 Downloading required assets from {repo_id} into '{local_dir}/'...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["chroma_db/*", "Qwen2.5-7B/7B-LoRA/*"],
                    local_dir=local_dir,
                    token=hf_token
                )
                print("✅ Download complete.")
            except Exception as e:
                print(f"❌ Failed to download assets: {e}")

    def _initialize(self):
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # 모델 구동 전 허깅페이스에서 로컬 파일(models/)로 강제 다운로드
        self._download_hf_assets(hf_token)

        print(f"Loading tokenizer & models... (Base: {self.base_model_id}, Adapter: {self.adapter_path})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            token=hf_token
        )
        
        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            token=hf_token
        )
        
        # Inject LoRA Adapter
        # 만약 로컬 폴더이거나 '/' 문자가 포함된 허깅페이스 레포지토리 주소라면 어댑터 로드를 시도합니다.
        if self.adapter_path and (os.path.exists(self.adapter_path) or "/" in self.adapter_path):
            try:
                self.model = PeftModel.from_pretrained(
                    base_model, 
                    self.adapter_path,
                    token=hf_token
                )
                print("✅ LoRA Adapter loaded successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA Adapter '{self.adapter_path}': {e}. Running with Base Model only.")
                self.model = base_model
        else:
            self.model = base_model
            print("⚠️ No valid LoRA Adapter path provided. Running with Base Model only.")

        print("Loading Vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vectorstore = Chroma(
            persist_directory=self.chroma_db_dir, 
            embedding_function=embeddings,
            collection_name="vet_qa_collection"
        )
        # return_source_documents is automatically handled when we access metadata if we retrieve directly.
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("✅ Core initialization complete.")

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
        
        context_text = ""
        citations = []
        for idx, doc in enumerate(docs):
            context_text += f"[근거 자료 {idx+1}]\n{doc.page_content}\n\n"
            citations.append({
                "doc_id": doc.metadata.get("id", f"doc_{idx}"),
                "title": doc.metadata.get("title", "수집된 수의학 지식"),
                "score": 1.0, # Chroma DB 래퍼 기본값에서는 점수 생략됨. 필요시 유사도 쿼리로 변경 가능
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
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,  # RAG 환경 환각 억제
                top_p=0.9
            )
        
        input_length = inputs.input_ids.shape[1]
        response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
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
