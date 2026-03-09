from __future__ import annotations

from typing import List, Dict, Any

import requests


class VllmGenerationClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        api_key: str | None,
        gen_temperature: float,
        gen_top_p: float,
        gen_max_new_tokens: int,
        gen_repetition_penalty: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_repetition_penalty = gen_repetition_penalty

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": self.gen_temperature,
                "top_p": self.gen_top_p,
                "max_tokens": self.gen_max_new_tokens,
                "repetition_penalty": self.gen_repetition_penalty,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM returned no choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("vLLM returned an empty response.")
        return content.strip()
