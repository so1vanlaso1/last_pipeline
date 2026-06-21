"""Resident LLM line-up config (serve/logic_config.yaml).

Defines which models are loaded during the slot, what ports their vLLM servers
use, their vote weights, and their ROLE in the Type 1 flow ("generator" — the
concurrent thinking 4B juniors — or "judge" — the 8B that arbitrates). Shared by:
  * run_server.sh  — to download + launch one vLLM server per model, and
  * the gateway    — to know each server's base URL + weight/role for Type 1.

The launch guard compares sum(params_b) against `max_resident_b` (yaml key, env
MAX_RESIDENT_B overrides; default 8). The committee's own limit is 8B TOTAL
resident at any moment (MoE counted by total params) — raising max_resident_b is
an explicit, logged decision, not something this module does silently.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "serve" / "logic_config.yaml"

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str) -> str:
    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")
    return _ENV.sub(repl, str(value))


def _num(value: Any, default: float) -> float:
    """Parse a numeric yaml field, allowing ${ENV:-x} substitution (so e.g.
    `params_b: ${JUDGE_PARAMS_B:-8.3}` works). Falls back to `default` on anything
    non-numeric (None, empty, an unexpanded var)."""
    try:
        return float(_expand(str(value)))
    except (TypeError, ValueError):
        return float(default)


# Friendly shortcuts for the Type-1 JUDGE. Set env JUDGE_MODEL to one of these keys
# (or to any full HF repo id, which passes through unchanged) and the judge entry in
# logic_config.yaml — `id: ${JUDGE_MODEL:-google/gemma-4-E4B-it}` — resolves to the
# repo below. Keys are matched case-insensitively.
_MODEL_ALIASES: Dict[str, str] = {
    "gemma":     "google/gemma-4-E4B-it",
    "gemma-e4b": "google/gemma-4-E4B-it",
    "gemma4":    "google/gemma-4-E4B-it",
}


def _read_yaml() -> Dict[str, Any]:
    path = Path(os.environ.get("LOGIC_CONFIG", str(DEFAULT_CONFIG)))
    if not path.exists():
        return {}
    import yaml  # lazy: only needed where PyYAML is installed
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ── Quantization ─────────────────────────────────────────────────────────────
# A model can be served in reduced precision to shrink its VRAM (and CPU-offload)
# footprint. `quantization:` in the yaml (or per-model `quantization:`) selects it;
# env QUANTIZATION overrides the global default. The label maps to the vLLM serve
# flags below (run_server.sh appends them verbatim):
#   none / bf16  -> full precision (the default; ~2 bytes/param)
#   8bit         -> online FP8 weight quant (~1 byte/param; needs an Ada/Hopper/
#                   Blackwell GPU, e.g. 4090/5070 — fp8 is unsupported on Ampere)
#   4bit         -> bitsandbytes NF4 (~0.5 byte/param; needs the `bitsandbytes`
#                   package, which serve/requirements.txt installs)
_QUANT_FLAGS: Dict[str, List[str]] = {
    "none": [],
    "bf16": [],
    "fp16": [],
    "8bit": ["--quantization", "fp8"],
    "fp8":  ["--quantization", "fp8"],
    "4bit": ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
    "nf4":  ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
}


def _norm_quant(q: Any) -> str:
    s = str(q or "").strip().lower()
    if s in ("", "false", "off", "no", "fp16", "float16", "bf16", "bfloat16", "auto"):
        return "none" if s not in ("fp16", "float16", "bf16", "bfloat16") else s
    if s in ("int4", "bnb4", "bnb"):
        return "4bit"
    if s in ("int8", "bnb8"):
        return "8bit"
    return s


def load_quant_global() -> str:
    """Default quantization for every model (env QUANTIZATION/QUANT beats the yaml
    `quantization:` key beats 'none')."""
    env = os.environ.get("QUANTIZATION") or os.environ.get("QUANT")
    if env:
        return _norm_quant(env)
    return _norm_quant(_read_yaml().get("quantization"))


def quant_flags(label: str) -> List[str]:
    """vLLM serve flags for a quantization label (unknown labels pass through as
    `--quantization <label>` so a future backend works without a code change)."""
    label = _norm_quant(label)
    if label in _QUANT_FLAGS:
        return list(_QUANT_FLAGS[label])
    return ["--quantization", label]


# ── Thinking mode ────────────────────────────────────────────────────────────
# Per-model switch for the reasoning calls (generators' THINK pass + the judge's
# arbitration): True lets the model emit a <think>…</think> chain before its JSON
# verdict; False asks for a direct answer (faster, cleaner JSON). This is a
# gateway-side request flag (chat_template_kwargs / Qwen3 `/no_think`), NOT a vLLM
# serve flag — so it never appears in the launch plan. Auxiliary JSON helper calls
# (premises_used, option-pick fallback) stay no-think regardless, for clean JSON.
def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def load_thinking_global() -> bool:
    """Default thinking mode for every model (env THINKING beats the yaml
    `thinking:` key beats True)."""
    env = os.environ.get("THINKING")
    if env is not None:
        return _truthy(env)
    val = _read_yaml().get("thinking")
    return True if val is None else _truthy(val)


def load_models() -> List[Dict[str, Any]]:
    """Return the resident model list with assigned ports + base URLs.

    Falls back to a single judge-class model (env MODEL_ID, default Gemma-4-E4B)
    if the config file is absent or empty.
    """
    raw = list(_read_yaml().get("models") or [])
    if not raw:
        raw = [{"id": os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"),
                "params_b": 8, "weight": 1.5, "role": "judge"}]

    g_quant = load_quant_global()
    g_think = load_thinking_global()
    base_port = int(os.environ.get("VLLM_BASE_PORT", os.environ.get("VLLM_PORT", "8001")))
    host = os.environ.get("VLLM_HOST", "localhost")
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(raw):
        mid = _expand(m.get("id", "")).strip()
        mid = _MODEL_ALIASES.get(mid.lower(), mid)   # 'gemma' shortcut -> repo id
        if not mid:
            continue
        params_b = _num(m.get("params_b"), 8)
        port = base_port + i
        # Per-model `quantization:` / `thinking:` override the global defaults.
        quant = _norm_quant(m["quantization"]) if m.get("quantization") is not None else g_quant
        thinking = _truthy(m["thinking"]) if m.get("thinking") is not None else g_think
        # Optional per-model `gpu_memory_utilization:` (0–1) pins this server's vLLM
        # --gpu-memory-utilization instead of the auto split; ignored if out of range.
        gmu = m.get("gpu_memory_utilization")
        gmu = _num(gmu, 0.0) if gmu is not None else None
        if gmu is not None and not (0.0 < gmu <= 1.0):
            gmu = None
        out.append({
            "id": mid,
            "params_b": params_b,
            "weight": _num(m.get("weight"), 1.5 if params_b >= 6 else 1.0),
            "model_class": "8b" if params_b >= 6 else "4b",
            "role": str(m.get("role", "")).strip().lower(),   # "generator"|"judge"|""
            "quant": quant,                                   # "none"|"4bit"|"8bit"|...
            "thinking": thinking,                             # reasoning-call think mode
            "gpu_mem_util": gmu,                              # explicit override or None
            "port": port,
            "base_url": f"http://{host}:{port}/v1",
        })
    return out


def load_mode() -> str:
    """Logic flow: 'arbiter' (2 thinking generators -> the judge picks + re-derives
    premises_used/explanation) or 'vote' (cascade weighted vote). Env LOGIC_MODE
    overrides the yaml `mode:` key. Default 'arbiter'."""
    env = os.environ.get("LOGIC_MODE")
    if env:
        return env.strip().lower()
    m = _read_yaml().get("mode")
    return str(m).strip().lower() if m else "arbiter"


def load_swap() -> bool:
    """Whether to SLEEP/WAKE-swap the line-up instead of holding it all resident.

    When on (the default) and the line-up has a `judge` role plus >=1 generator,
    the generators stay co-resident and the judge is woken (generators slept)
    only for its arbitration call — so peak VRAM is max(generators, judge), not
    their sum. This lets the 2x4B + 8B line-up run on a 24 GB card. Env SWAP
    overrides the yaml `swap:` key; default True. Ignored outside `arbiter` mode
    and for line-ups without an explicit judge (nothing to swap)."""
    env = os.environ.get("SWAP")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    val = _read_yaml().get("swap")
    if val is None:
        return True
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def swap_active(models: List[Dict[str, Any]]) -> bool:
    """True iff sleep/wake swap should actually engage for this line-up."""
    if not load_swap() or load_mode() != "arbiter":
        return False
    roles = [m.get("role", "") for m in models]
    return ("judge" in roles) and any(r != "judge" for r in roles)


def max_resident_b() -> float:
    """The residency budget the launch guard enforces. The committee limit is 8B
    total at any moment; a larger value here is the operator's explicit call (env
    MAX_RESIDENT_B beats the yaml `max_resident_b:` key beats the 8.0 default)."""
    env = os.environ.get("MAX_RESIDENT_B")
    if env:
        return float(env)
    val = _read_yaml().get("max_resident_b")
    return float(val) if val is not None else 8.0


def total_params_b(models: List[Dict[str, Any]]) -> float:
    return sum(m["params_b"] for m in models)


def gpu_fractions(models: List[Dict[str, Any]]) -> List[float]:
    """Per-server `--gpu-memory-utilization` for each model.

    SWAP line-up (swap_active): only one *group* is ever awake at a time, so each
    group is sized to the WHOLE card. The judge (woken alone) gets the full budget;
    the co-resident generators split the full budget among themselves (they are
    awake together). Because the groups never overlap on the GPU, both can claim
    ~the whole card without oversubscription.

    Non-swap line-up: every model is resident simultaneously, so the budget is
    split across all of them in proportion to parameter count (the old behaviour).
    A 0.05 floor keeps a tiny model loadable.

    A per-model `gpu_memory_utilization:` in the yaml (carried as `gpu_mem_util`)
    pins that server's fraction verbatim and bypasses the auto split for it."""
    total = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
    if not models:
        return []

    def _pinned(m: Dict[str, Any]) -> float | None:
        v = m.get("gpu_mem_util")
        return round(float(v), 4) if v is not None else None

    if swap_active(models):
        gens = [m for m in models if m.get("role") != "judge"]
        gen_sum = sum(max(float(m["params_b"]), 0.5) for m in gens) or 1.0
        out: List[float] = []
        for m in models:
            pin = _pinned(m)
            if pin is not None:
                out.append(pin)                                   # explicit override
            elif m.get("role") == "judge":
                out.append(round(total, 4))                       # alone on the card
            else:
                w = max(float(m["params_b"]), 0.5)
                out.append(round(max(total * w / gen_sum, 0.05), 4))  # co-resident split
        return out
    weights = [max(float(m["params_b"]), 0.5) for m in models]
    s = sum(weights)
    return [
        _pinned(m) if _pinned(m) is not None else round(max(total * w / s, 0.05), 4)
        for m, w in zip(models, weights)
    ]


def print_launch_plan() -> None:
    """Emit the launch plan for run_server.sh on stdout, after enforcing the
    residency budget (`max_resident_b`, default 8); exit non-zero if exceeded so
    setup.sh refuses to launch. When the budget was raised past the committee's
    8B, a loud warning states the compliance risk.

    Lines (TAB-separated):
      #swap<TAB>{0|1}                                  meta: sleep/wake swap on?
      {id}<TAB>{port}<TAB>{gpu_frac}<TAB>{role}<TAB>{quant_flags}   one per model

    `role` is generator/judge/- ; `quant_flags` is the space-joined vLLM serve
    flags (or '-' for full precision). With swap on, the judge's gpu_frac assumes
    it is awake ALONE — run_server.sh starts it first, then sleeps it, so the
    generators boot into the freed memory.

    For the residency BUDGET check, swap means peak VRAM is the largest awake
    group, not the sum — but the committee counts TOTAL params loaded, and slept
    weights still sit in CPU RAM, so we keep checking the full total (compliance
    is about params that exist, not what is momentarily on the GPU)."""
    import sys
    models = load_models()
    total = total_params_b(models)
    limit = max_resident_b()
    swap = swap_active(models)
    if total > limit + 1e-9:
        sys.stderr.write(
            f"[config] ERROR: resident models total {total:g}B > the {limit:g}B budget. "
            f"Edit serve/logic_config.yaml (pick a smaller line-up, or raise "
            f"max_resident_b if you explicitly accept the compliance risk).\n"
        )
        sys.exit(2)
    if total > 8.0 + 1e-9:
        sys.stderr.write(
            f"[config] WARNING: resident models total {total:g}B — over the committee's "
            f"8B-at-any-moment limit (Submission Guide 6.3; MoE counts TOTAL params). "
            f"Launching anyway because max_resident_b={limit:g} was set explicitly.\n"
        )
    sys.stdout.write(f"#swap\t{1 if swap else 0}\n")
    for m, frac in zip(models, gpu_fractions(models)):
        flags = " ".join(quant_flags(m["quant"])) or "-"
        sys.stdout.write(f"{m['id']}\t{m['port']}\t{frac}\t{m['role'] or '-'}\t{flags}\n")
    desc = ", ".join(
        "{0} ({1:g}B, {2}, w={3:g}, q={4}, think={5})".format(
            m["id"], m["params_b"], m["role"] or "voter", m["weight"], m["quant"],
            "on" if m["thinking"] else "off")
        for m in models)
    if swap:
        gens = [m for m in models if m.get("role") != "judge"]
        gsum = sum(m["params_b"] for m in gens)
        jud = next((m for m in models if m.get("role") == "judge"), None)
        peak = max(gsum, jud["params_b"] if jud else 0.0)
        sys.stderr.write(
            f"[config] line-up: {desc} = {total:g}B total; SWAP on -> peak resident "
            f"~{peak:g}B (generators {gsum:g}B co-resident; judge "
            f"{jud['params_b']:g}B swapped in)\n" if jud else
            f"[config] line-up: {desc} = {total:g}B (budget {limit:g}B)\n")
    else:
        sys.stderr.write(f"[config] line-up: {desc} = {total:g}B resident "
                         f"(budget {limit:g}B)\n")


if __name__ == "__main__":
    print_launch_plan()
