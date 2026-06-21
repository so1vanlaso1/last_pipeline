"""OpenAI-compatible chat client for the shared vLLM server.

The gateway uses this for:
  * the logic (Type 1) answer + premises_used extraction, and
  * the physics (Type 2) numeric fallback when the deterministic solver abstains.

The physics ExactFamaPipeline talks to vLLM through its OWN QwenClient (configured
by the gateway); this client is independent so the gateway controls thinking-mode,
token budget, and JSON extraction directly.

A `stub` mode returns deterministic canned completions so the full request/response
wiring can be exercised with no GPU and no model (see tests/).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .io_log import append_model_io, model_label

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
# Qwen3-4B-Thinking-2507 PRE-FILLS "<think>" in its chat template, so the model's
# output is "reasoning…</think>final answer" with NO opening tag in the content.
# Drop everything from the start up to and including the first lone </think>.
_THINK_CLOSE_ONLY = re.compile(r"^.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    """Remove a reasoning chain: a closed <think>…</think> block, a dangling
    <think>… (no close), or a lone …</think> with no opening tag (the form
    Qwen3-4B-Thinking-2507 emits because its template pre-fills the opening tag)."""
    text = text or ""
    text = _THINK_BLOCK.sub(" ", text)
    if "<think>" in text.lower():
        text = _THINK_OPEN.sub(" ", text)
    if "</think>" in text.lower():          # lone closing tag → strip the lead-in too
        text = _THINK_CLOSE_ONLY.sub(" ", text)
    return text.strip()


def extract_json_object(text: str) -> Optional[dict]:
    """Return the first balanced top-level JSON object in `text`, or None."""
    text = strip_think(text or "")
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_last_json_object(text: str) -> Optional[dict]:
    """Return the LAST balanced top-level JSON object in `text`, or None.

    The generator/judge prompts tell the model to END its reply with the JSON
    object; reasoning prose before it (some models think in plain text, untagged)
    can itself contain braces, so the last object is the verdict."""
    text = strip_think(text or "")
    best: Optional[dict] = None
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            return best
        depth, in_str, esc, end = 0, False, False, -1
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:                  # unbalanced from this '{' — try the next one
            i = start + 1
            continue
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:  # balanced but not JSON (prose braces)
            i = start + 1
            continue
        if isinstance(obj, dict):
            best = obj
            i = end + 1              # next TOP-LEVEL object starts after this one
        else:
            i = start + 1


class LLMClient:
    def __init__(
        self,
        mode: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 240.0,
        thinking: bool = True,
    ):
        self.mode = (mode or os.environ.get("GATEWAY_LLM", "vllm")).lower()
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")).rstrip("/")
        self.model = model or os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
        # 240s (was 55s): a THINKING generation of up to LOGIC_THINK_TOKENS tokens on
        # the 4B/8B models at eager-FP8 can take 60-150s. A 55s cap timed out the judge
        # on EVERY query, silently dropping its verdict and falling back to a generator's
        # answer — defeating the arbiter design. Env GATEWAY_LLM_TIMEOUT overrides.
        self.timeout = float(os.environ.get("GATEWAY_LLM_TIMEOUT", timeout))
        # Per-model default for the reasoning calls (from serve/logic_config.yaml
        # `thinking:`). Used when a chat() caller passes enable_thinking=None.
        self.thinking = thinking

    # ── public API ────────────────────────────────────────────────────────────
    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: Optional[bool] = None,
        log_context: Optional[str] = None,
        loaded_models: Optional[List[str]] = None,
    ) -> str:
        """Single-turn chat completion. Returns the assistant text (think-stripped).

        `enable_thinking=None` (the default) uses this client's configured
        `thinking` mode; pass True/False to force it (helper JSON calls force
        False for clean output)."""
        t0 = time.perf_counter()
        raw = ""
        error: Optional[str] = None
        effective_enable = self.thinking if enable_thinking is None else enable_thinking
        effective_user = user
        # Qwen3 soft-switch: '/no_think' reliably disables the <think> block
        # regardless of the vLLM version's handling of chat_template_kwargs.
        if not effective_enable and "qwen3" in (self.model or "").lower():
            effective_user = f"{user}\n/no_think"

        def _set_raw(text: str) -> None:
            nonlocal raw
            raw = text

        try:
            if self.mode == "stub":
                raw = _stub_completion(system, user)
                return raw
            return self._chat_vllm(
                system,
                effective_user,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=effective_enable,
                raw_out=_set_raw,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if log_context:
                append_model_io(
                    context=log_context,
                    model=model_label(self),
                    loaded_models=loaded_models or [model_label(self)],
                    system=system,
                    user=effective_user,
                    raw=raw,
                    elapsed_s=time.perf_counter() - t0,
                    error=error,
                )

    def _chat_vllm(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: Optional[bool],
        raw_out,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Disable Qwen3 thinking for speed + clean parsing. Harmless for models
            # whose chat template ignores the kwarg.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        url = self.base_url + "/chat/completions"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            # Retry once without the (vLLM-version-specific) chat_template_kwargs.
            payload.pop("chat_template_kwargs", None)
            resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        raw_out(content or "")
        return strip_think(content or "")

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        log_context: Optional[str] = None,
        loaded_models: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Chat and parse the first JSON object out of the reply. Forces no-think
        (these are small helper calls — premises_used, option pick — where a
        reasoning chain only risks polluting the JSON and costs latency)."""
        text = self.chat(
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
            log_context=log_context,
            loaded_models=loaded_models,
        )
        return extract_json_object(text)

    def models(self) -> dict:
        """Proxy payload for GET /v1/models."""
        if self.mode == "stub":
            return {"object": "list", "data": [{"id": self.model, "object": "model", "owned_by": "vllm"}]}
        resp = requests.get(self.base_url + "/models", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ── stub backend (no GPU) ─────────────────────────────────────────────────────
def _stub_completion(system: str, user: str) -> str:
    """Deterministic canned replies that satisfy each parser the gateway uses.
    Branches on a unique phrase from each gateway prompt. Only for offline wiring
    tests — never used when GATEWAY_LLM=vllm."""
    s = (system or "").lower()
    u = (user or "")
    has_options = "options:" in u.lower()
    if "option_index" in s:                                    # MCQ pick fallback
        return '{"option_index": 0}'
    if "senior examiner and arbiter" in s:                     # arbiter (choice)
        ans = "A" if has_options else "Yes"
        return ('{"chosen": 1, "answer": "%s", "premises_used": [1, 2], '
                '"explanation": "Arbiter re-derives the answer from premises 1 and 2."}' % ans)
    if "senior arbiter" in s:                                  # arbiter (free-form)
        return '{"chosen": 1, "answer": "2", "premises_used": [1, 2], "explanation": "Arbiter stub."}'
    if "physics problem solver" in s:                          # physics LLM fallback
        return '{"answer": "5", "unit": "A", "steps": ["Stub physics fallback."]}'
    if "a number or short text" in s or "final answer as a number or short text" in s:
        return '{"answer": "2", "premises_used": [1, 2], "explanation": "Stub free-form answer from premises 1 and 2."}'
    if "finish your reply with one json object" in s:          # arbiter generator (choice)
        ans = "A" if has_options else "Yes"
        return ('{"answer": "%s", "premises_used": [1, 2], '
                '"explanation": "Junior cites premises 1 and 2."}' % ans)
    if "which premise numbers are needed" in s or "identify which premises" in s:
        return '{"premises_used": [0, 1]}'                      # premises_used extraction
    # Default: the cascade ANSWER:/WHY: format.
    if "option" in s or "Options:" in u:
        return "ANSWER: A\nWHY: Premise 1 entails the first option."
    return "ANSWER: Yes\nWHY: Premises 1 and 2 together entail the statement."
