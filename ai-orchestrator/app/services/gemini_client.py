# app/services/gemini_client.py
import os
from typing import Optional
from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # API 키 로드 (GEMINI_API_KEY 또는 GOOGLE_API_KEY 모두 지원)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment variables.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        
        # 판정 결과용 JSON 스키마 정의 (Gemini가 이 구조에 맞춰 응답하도록 강제)
        self._judgment_schema = types.Schema(
            type = types.Type.OBJECT,
            required = ["success", "confidence", "reason"],
            properties = {
                "success": types.Schema(
                    type = types.Type.BOOLEAN,
                    description = "Whether the mission succeeded based on the video."
                ),
                "confidence": types.Schema(
                    type = types.Type.NUMBER,
                    description = "Confidence score between 0 and 1."
                ),
                "reason": types.Schema(
                    type = types.Type.STRING,
                    description = "1-2 sentence explanation referencing decisive visual cues."
                ),
            },
        )

    def generate_from_video_url(
        self,
        video_url: str,
        prompt_text: str,
        *,
        response_mime_type: str = "application/json",
        response_schema: Optional[types.Schema] = None,
    ) -> str:
        """
        Generates content using Gemini.
        - URL is sent directly to Gemini (S3 etc).
        """
        # Production 환경에서는 S3 URL을 그대로 Gemini에 넘깁니다.
        parts = [
            types.Part.from_uri(
                file_uri=video_url,
                mime_type="video/mp4"
            ),
            types.Part.from_text(text=prompt_text),
        ]

        # 응답 설정 (MIME 타입 및 JSON 스키마 적용)
        config_kwargs = {"response_mime_type": response_mime_type}
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        # 실제 Gemini API 호출 (동기)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return resp.text or ""

    # --- 비동기 확장용 (Async Expansion) ---
    # async def generate_from_video_url_async(
    #     self,
    #     video_url: str,
    #     prompt_text: str,
    #     *,
    #     response_mime_type: str = "application/json",
    #     response_schema: Optional[types.Schema] = None,
    # ) -> str:
    #     parts = [
    #         types.Part.from_uri(
    #             file_uri=video_url,
    #             mime_type="video/mp4"
    #         ),
    #         types.Part.from_text(text=prompt_text),
    #     ]
    #
    #     config_kwargs = {"response_mime_type": response_mime_type}
    #     if response_schema is not None:
    #         config_kwargs["response_schema"] = response_schema
    #
    #     # google.genai Client의 비동기 지원 여부에 따라 구현이 달라질 수 있습니다.
    #     # 만약 aio 모듈이 있다면:
    #     resp = await self.client.aio.models.generate_content(
    #         model=self.model_name,
    #         contents=[types.Content(parts=parts)],
    #         config=types.GenerateContentConfig(**config_kwargs)
    #     )
    #     return resp.text or ""
    # -------------------------------------

    def generate_judgment_from_video_url(self, video_url: str, prompt_text: str) -> str:
        return self.generate_from_video_url(
            video_url = video_url,
            prompt_text = prompt_text,
            response_schema = self._judgment_schema,
        )

    # 단순히 텍스트만 생성하는 메서드 (스키마 없이 사용)
    def generate_text_from_video_url(self, video_url: str, prompt_text: str) -> str:
        return self.generate_from_video_url(
            video_url = video_url,
            prompt_text = prompt_text,
            response_mime_type = "text/plain",
        )