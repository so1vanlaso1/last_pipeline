from __future__ import annotations

import json
from typing import Any

import requests


class QwenClient:
    """Thin client for Qwen3-8B.

    Supported backends:
    - none: deterministic fallback, no model loaded.
    - hf: local Hugging Face Transformers model.
    - openai_compatible: local vLLM server exposing /v1/chat/completions.

    The JSON helpers are intentionally conservative: they first prefer native
    schema-guided decoding when an OpenAI-compatible/vLLM endpoint is used, and
    otherwise fall back to strict extraction + retry. This keeps zero-shot parser
    experiments reliable without requiring fine-tuning.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.backend = str(self.config.get("backend", "none"))
        self.model_name = str(self.config.get("name", "Qwen/Qwen3-8B"))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 512))
        self.vllm_base_url = str(self.config.get("vllm_base_url", "http://localhost:8001/v1"))
        self.lora_adapter_path = str(self.config.get("lora_adapter_path") or "").strip()
        self.load_in_4bit = bool(self.config.get("load_in_4bit", True))

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Public generation helpers
    # ------------------------------------------------------------------
    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object.

        Args:
            messages: Chat messages.
            schema: Optional JSON Schema. For openai_compatible/vLLM, this is
                sent as a structured-output constraint. For HF, it is included
                in retry prompts only; no extra dependency is required.
            max_retries: Number of repair retries after the first parse failure.
        """
        attempts = 0
        last_error: Exception | None = None
        working_messages = list(messages)

        while attempts <= max(0, max_retries):
            text = self.generate(working_messages, schema=schema if attempts == 0 else None)
            try:
                data = self._loads_json_object(text)
                if isinstance(data, dict):
                    return data
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            except Exception as exc:  # intentionally broad: keep retry robust
                last_error = exc
                attempts += 1
                if attempts > max_retries:
                    break
                repair = (
                    "Your previous response was not a valid JSON object that can be parsed by json.loads. "
                    "Return exactly one JSON object and no markdown, no comments, no prose."
                )
                if schema:
                    repair += " The object must conform to this JSON Schema: " + json.dumps(schema, ensure_ascii=False)
                working_messages = list(messages) + [{"role": "user", "content": repair}]

        raise ValueError(f"LLM did not return valid JSON. Last parse error: {last_error}")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> str:
        if self.backend == "none":
            return '{"note":"LLM_BACKEND=none; deterministic fallback active."}'
        if self.backend == "openai_compatible":
            return self._generate_openai_compatible(messages, schema=schema)
        if self.backend == "hf":
            return self._generate_hf(messages)
        raise ValueError(f"Unsupported LLM_BACKEND: {self.backend}")

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------
    def _loads_json_object(self, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise ValueError(f"JSON was {type(data).__name__}, not object")
        except json.JSONDecodeError:
            candidate = self._extract_first_json_object(text)
            if candidate is None:
                raise
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError(f"Extracted JSON was {type(data).__name__}, not object")
            return data

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        """Extract the first balanced JSON object while respecting strings."""
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _generate_openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> str:
        url = self.vllm_base_url.rstrip("/") + "/chat/completions"
        # Disable Qwen3 <think> blocks for speed + clean JSON. This does not change
        # the model's answer (the reasoning prefix is non-answer text); it only
        # avoids slow, hard-to-parse reasoning dumps. '/no_think' is Qwen3's soft
        # switch and works even if a vLLM build ignores chat_template_kwargs.
        if messages and "qwen3" in (self.model_name or "").lower():
            messages = list(messages)
            last = dict(messages[-1])
            if last.get("role") == "user" and "/no_think" not in str(last.get("content", "")):
                last["content"] = f"{last.get('content', '')}\n/no_think"
                messages[-1] = last
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        if schema:
            # vLLM current API uses structured_outputs.json. Some older vLLM
            # builds used guided_json, so we gracefully retry with that below.
            payload["structured_outputs"] = {"json": schema}

        resp = requests.post(url, json=payload, timeout=180)

        if schema and resp.status_code >= 400:
            fallback_payload = dict(payload)
            fallback_payload.pop("structured_outputs", None)
            fallback_payload["guided_json"] = schema
            resp = requests.post(url, json=fallback_payload, timeout=180)

        if resp.status_code >= 400:
            # A vLLM build that rejects chat_template_kwargs as an unknown field:
            # retry once without it (the '/no_think' token still disables thinking).
            retry_payload = dict(payload)
            retry_payload.pop("chat_template_kwargs", None)
            resp = requests.post(url, json=retry_payload, timeout=180)

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _ensure_hf_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "load_in_4bit=true but BitsAndBytesConfig/bitsandbytes is unavailable. "
                    "Install requirements-train.txt or set load_in_4bit=false."
                ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        if self.lora_adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "LORA_ADAPTER_PATH is set but peft is unavailable. "
                    "Install requirements-train.txt or clear LORA_ADAPTER_PATH."
                ) from exc

            self._model = PeftModel.from_pretrained(self._model, self.lora_adapter_path)

        self._model.eval()

    def _generate_hf(self, messages: list[dict[str, str]]) -> str:
        self._ensure_hf_loaded()

        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
