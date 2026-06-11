"""Chat-model backends.

`ChatModel` wraps a HuggingFace causal-LM (or multimodal-it, e.g. Gemma) with:
  * a precision switch — 4bit (NF4) | 8bit (int8) | bf16 (also fp16 / fp32),
  * a robust loader that falls back across model classes / tokenizer vs processor,
  * a chat renderer that tolerates templates with no `system` role (Gemma),
  * `unload()` that actually frees VRAM, so the cascade can keep at most two 4B
    models OR one 8B model resident at any moment.

`StubModel` is a zero-dependency stand-in for wiring tests (no torch needed).
"""

from __future__ import annotations

import gc
import logging
import time

log = logging.getLogger(__name__)

PRECISIONS = ("4bit", "8bit", "bf16", "fp16", "fp32")


class ChatModel:
    def __init__(
        self,
        model_id: str,
        precision: str = "4bit",
        device_map: str = "auto",
        compute_dtype: str = "bfloat16",
        enable_thinking: bool = False,
        label: str = "",
        vote_weight: float = 1.0,
        model_class: str = "4b",
        always_cot: bool = False,
    ):
        import torch  # lazy: keep the package importable without torch
        from transformers import AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.label = label or model_id
        self.enable_thinking = enable_thinking
        # Soft-vote metadata, copied onto every ModelReply (see cascade.query).
        self.vote_weight = vote_weight
        self.model_class = model_class
        # Models that always emit a chain of thought (e.g. LFM2.5) need more room
        # before the ANSWER line appears — the caller bumps max_new_tokens for them.
        self.always_cot = always_cot

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

        has_cuda = torch.cuda.is_available()
        if precision in ("4bit", "8bit") and not has_cuda:
            log.warning("%s: %s needs a CUDA GPU (bitsandbytes); falling back to fp32/CPU.",
                        self.label, precision)
            precision = "fp32"
        self.precision = precision
        compute = getattr(torch, compute_dtype, torch.bfloat16)

        # Tokenizer (fall back to a processor for multimodal repos like Gemma 3n/4).
        self.processor = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: AutoTokenizer failed (%s); trying AutoProcessor.", self.label, e)
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:  # noqa: BLE001
                pass

        quant = None
        load_dtype = compute
        if precision == "4bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True,
            )
        elif precision == "8bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "bf16":
            load_dtype = torch.bfloat16
        elif precision == "fp16":
            load_dtype = torch.float16
        elif precision == "fp32":
            load_dtype = torch.float32

        # Device placement. On a SINGLE CUDA GPU, "auto" makes accelerate keep a
        # memory margin and can spuriously offload a model that WOULD fit (e.g. an
        # 8B MoE at 8-bit on a 12 GB card) to CPU/disk — which bitsandbytes int8
        # then refuses to load. Pin the whole model to GPU 0 instead; a genuinely
        # over-budget load then OOMs honestly rather than silently offloading.
        # Multi-GPU "auto" and any explicit device_map are left untouched.
        if not has_cuda:
            resolved_device_map = "cpu"
        elif device_map == "auto" and torch.cuda.device_count() == 1:
            resolved_device_map = {"": 0}
        else:
            resolved_device_map = device_map

        self.model = self._load_model(model_id, load_dtype, resolved_device_map, quant)
        self.model.eval()
        log.info("loaded %s (%s, precision=%s)", self.label, model_id, self.precision)

    def _load_model(self, model_id: str, dtype, device_map, quant):
        from transformers import AutoModelForCausalLM

        def _from(cls):
            kwargs = dict(trust_remote_code=True, device_map=device_map)
            if quant is not None:
                kwargs["quantization_config"] = quant
            # transformers>=5 renamed torch_dtype→dtype; fall back for older versions.
            try:
                return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
            except TypeError:
                return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)

        try:
            return _from(AutoModelForCausalLM)
        except Exception as e:  # noqa: BLE001
            # Only retry as image-text-to-text when the failure is an ARCHITECTURE
            # mismatch (some Gemma multimodal repos map there, not to CausalLM). A
            # quantization / OOM / device-offload error is NOT about the arch —
            # re-raise it so the real cause surfaces instead of a misleading
            # "Unrecognized configuration class … AutoModelForImageTextToText".
            if "Unrecognized configuration class" not in str(e):
                raise
            log.warning("%s: AutoModelForCausalLM can't map this architecture (%s); "
                        "trying AutoModelForImageTextToText.", self.label, e)
            from transformers import AutoModelForImageTextToText
            return _from(AutoModelForImageTextToText)

    # ── prompt rendering ──────────────────────────────────────────────────────
    def _apply_template(self, messages: list[dict]) -> str:
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kw
            )
        except TypeError:
            # Template doesn't accept `enable_thinking` (most non-Qwen models).
            return self.tokenizer.apply_chat_template(messages, **kw)

    def render(self, system: str, user: str) -> str:
        """Render a system+user chat. If the template rejects a `system` role
        (e.g. Gemma), fold the system text into the user turn."""
        try:
            return self._apply_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception:  # noqa: BLE001
            return self._apply_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}]
            )

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, system: str, user: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> tuple[str, str, float]:
        """Return (raw_completion, rendered_prompt, elapsed_seconds)."""
        torch = self._torch
        prompt = self.render(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs = dict(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        prompt_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return text, prompt, elapsed

    def unload(self) -> None:
        """Drop the model + tokenizer and release VRAM."""
        for attr in ("model", "tokenizer", "processor"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:  # noqa: BLE001
                    pass
        gc.collect()
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        log.info("unloaded %s", self.label)


class StubModel:
    """Deterministic, dependency-free model for wiring tests.

    `answer_fn(user_prompt) -> str` decides the raw answer token, letting tests
    drive both the agreement and the disagreement branch without a GPU. The stub
    answers in whichever format the prompt demands: the generate→judge JSON
    (with leading "thinking" prose so last-JSON extraction is exercised) or the
    vote flow's ANSWER:/WHY: text.
    """

    def __init__(self, label: str, answer_fn, vote_weight: float = 1.0,
                 model_class: str = "4b", always_cot: bool = False):
        self.label = label
        self.model_id = f"stub:{label}"
        self.precision = "stub"
        self._answer_fn = answer_fn
        self.vote_weight = vote_weight
        self.model_class = model_class
        self.always_cot = always_cot

    def generate(self, system: str, user: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> tuple[str, str, float]:
        import json
        ans = self._answer_fn(user)
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if "JSON object" in system:
            data = {"answer": ans, "premises_used": [1],
                    "explanation": f"stub explanation from {self.label}, citing premise 1."}
            if "SENIOR examiner" in system:
                data = {"chosen": 1, **data}
            raw = "Let me reason it out {step by step} first...\n" + json.dumps(data)
        else:
            raw = f"ANSWER: {ans}\nWHY: stub explanation from {self.label}."
        return raw, prompt, 0.0

    def unload(self) -> None:
        pass
