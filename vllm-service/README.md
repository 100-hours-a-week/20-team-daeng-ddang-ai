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
- `VLLM_EXTRA_ARGS`: 추가 실행 옵션

## 실행 예시

```bash
docker compose up -d
```

같은 서버에서 `chatbot-service`는 `http://vllm-service:8400` 또는 내부 주소로 이 서비스를 호출하면 됩니다.
