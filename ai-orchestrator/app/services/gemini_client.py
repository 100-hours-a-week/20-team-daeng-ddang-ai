import os
import requests
from typing import Optional

from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment variables.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def generate_from_video_url(self, video_url: str, prompt_text: str) -> str:
        """
        Generates content using Gemini.
        - If URL is local (localhost/127.0.0.1), downloads bytes and sends inline.
        - If URL is public, sends URL directly to Gemini.
        """
        is_local = "localhost" in video_url or "127.0.0.1" in video_url
        
        parts = []
        
        if is_local:
            try:
                r = requests.get(video_url, timeout=30)
                r.raise_for_status()
                parts = [
                    types.Part.from_bytes(data=r.content, mime_type="video/mp4"),
                    types.Part.from_text(text=prompt_text),
                ]
            except Exception as e:
                raise RuntimeError(f"Failed to fetch local video: {e}")
        else:
            parts = [
                types.Part.from_uri(
                    file_uri=video_url,
                    mime_type="video/mp4"
                ),
                types.Part.from_text(text=prompt_text),
            ]

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return resp.text or ""
