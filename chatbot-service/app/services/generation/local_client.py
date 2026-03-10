from __future__ import annotations

from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalGenerationClient:
    def __init__(
        self,
        *,
        base_model_id: str,
        adapter_path: str,
        hf_token: str | None,
        gen_temperature: float,
        gen_top_p: float,
        gen_max_new_tokens: int,
        gen_repetition_penalty: float,
    ) -> None:
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.hf_token = hf_token
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_repetition_penalty = gen_repetition_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=self.hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=self._select_torch_dtype(),
            token=self.hf_token,
        )

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path, token=self.hf_token)
        else:
            self.model = base_model

    @staticmethod
    def _select_torch_dtype() -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    repetition_penalty=self.gen_repetition_penalty,
                )
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(
                    "GPU OOM while generating answer. "
                    "Reduce max_new_tokens, shorten history/context, or use a larger GPU."
                ) from e

        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
