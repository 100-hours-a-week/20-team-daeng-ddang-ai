# vLLM Service

`chatbot-service`가 내부적으로 호출하는 LLM 추론 서버입니다.

## 역할

- Qwen 계열 모델을 `vLLM`으로 서빙
- 외부 트래픽은 직접 받지 않고 `chatbot-service`가 내부 호출
- GPU는 이 서비스에 우선 할당하고, `chatbot-service`는 RAG/프롬프트 처리 담당

## 환경 변수

- `VLLM_MODEL`: 서빙할 모델 또는 merged checkpoint 경로
- `VLLM_PORT`: 기본 포트
- `VLLM_API_KEY`: 필요 시 OpenAI-compatible 인증 토큰
- `VLLM_GPU_MEMORY_UTILIZATION`: vLLM GPU 메모리 점유율 상한
- `VLLM_MAX_MODEL_LEN`: 최대 컨텍스트 길이
- `VLLM_TENSOR_PARALLEL_SIZE`: GPU 병렬도
- `VLLM_DTYPE`: `auto`, `float16`, `bfloat16` 등
- `VLLM_MODEL_CACHE_DIR`: HF에서 내려받은 모델/LoRA를 저장할 로컬 디렉터리
- `VLLM_ENABLE_LORA`: `true`면 LoRA adapter 활성화
- `VLLM_LORA_NAME`: OpenAI-compatible `model` 필드로 사용할 LoRA 별칭
- `VLLM_LORA_PATH`: HF repo 경로 또는 로컬 LoRA adapter 경로
- `VLLM_LORA_REPO_ID`: LoRA가 별도 HF repo에 있을 때 repo id
- `VLLM_LORA_SUBDIR`: LoRA가 HF repo 하위 폴더에 있을 때 subdir
- `VLLM_LORA_REVISION`: LoRA revision 또는 commit sha
- `VLLM_LORA_MODULES`: `name=path` 형식의 raw vLLM LoRA 옵션. 값이 있으면 `VLLM_LORA_NAME/PATH`보다 우선
- `HF_TOKEN`: 비공개 HF repo 접근 토큰
- `VLLM_EXTRA_ARGS`: 추가 실행 옵션

## 파일 구성

- `docker-compose.yml`: 도커 실행 정의
- `.env.example`: 기본 환경 변수 예시
- `setup.sh`: 직접 실행용 가상환경/패키지 설치
- `run.py`: `python run.py`로 vLLM 서버 실행
- `run.sh`: 활성화된 가상환경이 있으면 사용해 실행
- `healthcheck.py`: `/health` 확인용 스크립트
- `vllm-service.service.example`: systemd 등록 예시

## 실행 예시

```bash
docker compose up -d
```

같은 서버에서 `chatbot-service`는 `http://vllm-service:8400` 또는 내부 주소로 이 서비스를 호출하면 됩니다.

LoRA adapter를 같이 서빙하려면 예를 들어 다음처럼 설정합니다.

```env
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_ENABLE_LORA=true
VLLM_LORA_NAME=vet-chat-lora
VLLM_LORA_REPO_ID=20-team-daeng-ddang-ai/vet-chat
VLLM_LORA_SUBDIR=Qwen2.5-7B/7B-LoRA
HF_TOKEN=hf_xxx
```

이 경우 `chatbot-service`의 `VLLM_MODEL_NAME`도 `vet-chat-lora`로 맞춰야 합니다. 그렇지 않으면 베이스 모델 이름으로 요청이 들어가 LoRA가 적용되지 않습니다.

`VLLM_LORA_PATH=20-team-daeng-ddang-ai/vet-chat/Qwen2.5-7B/7B-LoRA`처럼 `repo/subdir`를 한 줄로 넣어도 `run.py`가 자동으로 repo와 subdir를 분리해 로컬 캐시에 내려받도록 처리합니다.

## 도커 없이 실행

```bash
chmod +x setup.sh run.sh
./setup.sh
cp .env.example .env
source .venv/bin/activate
python run.py
```

헬스 체크:

```bash
source .venv/bin/activate
python healthcheck.py
```

## systemd 예시

`vllm-service.service.example`를 서버 경로에 맞게 수정한 뒤 `/etc/systemd/system/vllm-service.service`로 배치하면 됩니다.
