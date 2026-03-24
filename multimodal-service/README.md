# Multimodal Service

`multimodal-service`는 이미지 URL을 입력받아 다음 순서로 내부 분석을 수행합니다.

1. 표정분석 서비스와 같은 `open_clip` 기반 `MobileCLIP-S1` 로더로 이미지가 반려견 안구 확대 사진인지, 피부 확대 사진인지, 기타 사진인지 분류
2. 안구 확대 사진이면 EfficientNet-B0 기반 안구 질환 분류 모델(`best.pt`)로 상위 질환 예측
3. 피부 확대 사진이면 별도 EfficientNet-B0 기반 피부 질환 분류 모델로 상위 질환 예측
4. 결과를 `chatbot-service`가 받아 LLM 프롬프트 보조 정보로 사용

## Endpoint

- `POST /analyze`
- `GET /health`

요청 예시:

```json
{
  "image_url": "https://example.com/dog-eye.jpg"
}
```

## 주요 환경변수

- `ROUTE_MODEL_ID`: 기본값 `MobileCLIP-S1`
- `ROUTE_MODEL_PRETRAINED`: 기본값 `datacompdr`
- `EYE_MODEL_ID`: 기본값 `20-team-daeng-ddang-ai/vet-chat`
- `EYE_MODEL_SUBDIR`: 기본값 `eye_disease_classifier`
- `EYE_MODEL_FILENAME`: 기본값 `best.pt`
- `EYE_RUN_CONFIG_FILENAME`: 기본값 `run_config.json`
- `SKIN_MODEL_ID`: 기본값 빈 문자열, 값이 있으면 피부 질환 분류 활성화
- `SKIN_MODEL_SUBDIR`: 기본값 `skin_disease_classifier`
- `SKIN_MODEL_FILENAME`: 기본값 `best.pt`
- `SKIN_RUN_CONFIG_FILENAME`: 기본값 `run_config.json`
- `HF_TOKEN`: 비공개 HF Hub 접근 토큰

안구/피부 모델이 허브 루트가 아니라 서브디렉터리에 올라가 있으면 각 `*_MODEL_SUBDIR`를 유지하면 됩니다. 로컬 디렉터리를 직접 쓰고 싶으면 `EYE_MODEL_ID`나 `SKIN_MODEL_ID`를 `/root/.../eye153_efficientnet_b0_hardlink`, `/root/.../skin152_efficientnet_b0_hardlink` 같은 경로로 바꿀 수 있습니다. `SKIN_MODEL_ID`가 비어 있거나 로드 실패하면 피부 분기만 비활성화되고 서비스는 계속 동작합니다.
