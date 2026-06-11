"""EXACT 2026 gateway — the single /predict endpoint the committee calls.

    POST /predict      competition I/O (Section 3 in, Section 4 list out)
    GET  /v1/models    aggregated model list across every resident vLLM server
    GET  /health       liveness

Routing by `type`:  type1 -> logic pipeline (vote over resident models),
                    type2 -> physics pipeline (solver, via the primary model).
Models are loaded once (singletons) and reused across the sequential grading slot.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Tuple

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from . import config as cfg
from . import gpu
from .logic_adapter import answer_type1
from .physics_adapter import PhysicsAdapter
from .residency import get_manager
from .schema import PredictQuery, PredictResult, empty_result
from .vllm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

# run_server.sh writes vllm_<port>.pid here; /health maps them to per-model VRAM.
_LOGDIR = Path(__file__).resolve().parents[1] / "logs"   # <repo>/serve/logs

_judges: List[Tuple[LLMClient, float, str, str]] | None = None
_physics: PhysicsAdapter | None = None


def get_judges() -> List[Tuple[LLMClient, float, str, str]]:
    """The resident line-up as [(client, vote_weight, model_class, role), ...].
    `role` is "generator" / "judge" / "" (from serve/logic_config.yaml)."""
    global _judges
    if _judges is None:
        models = cfg.load_models()
        _judges = [
            (LLMClient(base_url=m["base_url"], model=m["id"], thinking=m["thinking"]),
             m["weight"], m["model_class"], m.get("role", ""))
            for m in models
        ]
        total = cfg.total_params_b(models)
        log.info("logic line-up: %s = %gB resident",
                 ", ".join(f"{m['id']}({m.get('role') or 'voter'}, w={m['weight']:g})"
                           for m in models), total)
        if total > 8.0 + 1e-9:
            if cfg.swap_active(models):
                gens_b = sum(m["params_b"] for m in models if m.get("role") != "judge")
                judge_b = max((m["params_b"] for m in models if m.get("role") == "judge"),
                              default=0.0)
                log.info("line-up DECLARES %gB total, but SWAP keeps only one group on the GPU "
                         "-> peak ~%gB resident (<= 8B at any moment; load/unload per Q3)",
                         total, max(gens_b, judge_b))
            else:
                log.warning("resident models total %gB with SWAP OFF EXCEEDS the committee's "
                            "8B-at-any-moment limit (max_resident_b=%g in serve/logic_config.yaml "
                            "— this is YOUR compliance call)", total, cfg.max_resident_b())
    return _judges


def primary_client() -> LLMClient:
    return get_judges()[0][0]


def get_physics() -> PhysicsAdapter:
    global _physics
    if _physics is None:
        _physics = PhysicsAdapter(primary_client())
        if _physics.import_error:
            log.warning("physics pipeline import failed: %s", _physics.import_error)
        else:
            log.info("physics pipeline ready (model=%s)", primary_client().model)
    return _physics


@asynccontextmanager
async def _lifespan(app: FastAPI):
    get_judges()
    get_manager()      # log + build the sleep/wake residency plan
    get_physics()
    yield


app = FastAPI(title="EXACT 2026 Gateway", version="1.1.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness + a LIVE proof of the ≤8B-at-any-moment invariant. For each declared
    vLLM server it reports role, params, whether it is asleep (weights offloaded to
    CPU RAM), and its current GPU VRAM (via nvidia-smi). `params_loaded_running_b`
    sums only the AWAKE servers — the figure the committee verifies against 8B
    (Submission Guide §6.3). The swap keeps a slept model's weights off the GPU, so
    only one group (≤8B) is ever loaded-and-running."""
    c = primary_client()
    models_cfg = cfg.load_models()
    mgr = get_manager()
    by_pid = gpu.by_pid()
    servers: List[dict] = []
    loaded_b = 0.0
    for m in models_cfg:
        asleep = mgr.server_asleep(m["base_url"]) if mgr.enabled else False
        pid = gpu.pid_from_portfile(_LOGDIR, m["port"])
        if not asleep:                       # a slept model contributes only CUDA
            loaded_b += float(m["params_b"])  # context residual, not its weights
        servers.append({
            "id": m["id"], "role": m["role"] or "voter", "params_b": m["params_b"],
            "port": m["port"], "asleep": bool(asleep),
            "gpu_mib": by_pid.get(pid) if pid else None,
        })
    return {
        "status": "ok",
        "mode": c.mode,
        "swap": mgr.enabled,
        "models": [s["id"] for s in servers],
        "servers": servers,
        "params_loaded_running_b": round(loaded_b, 2),
        "limit_b": 8,
        "gpu": gpu.overall(),
    }


@app.get("/v1/models")
def models() -> JSONResponse:
    """Aggregate every resident server's /v1/models so a single host lists all
    DECLARED models. The combined-size compliance (≤8B loaded-and-running at any
    moment) is enforced by the residency swap and shown live by GET /health, not by
    this list — which honestly reports every model that exists in the line-up."""
    data: List[dict] = []
    seen = set()
    for client, *_ in get_judges():
        try:
            payload = client.models()
            for item in payload.get("data", []):
                mid = item.get("id")
                if mid not in seen:
                    seen.add(mid)
                    data.append(item)
        except Exception as exc:  # pragma: no cover
            log.warning("/v1/models proxy failed for %s: %s", client.model, exc)
    return JSONResponse({"object": "list", "data": data})


@app.get("/servers")
def servers() -> JSONResponse:
    """Index of every vLLM server with its OWN /v1/models path (proxied through this
    host). Lets the committee verify each model independently per Submission Guide
    §6.3 ("a /v1/models URL for each vLLM server")."""
    out = [
        {"id": m["id"], "role": m["role"] or "voter", "params_b": m["params_b"],
         "port": m["port"], "models_url": f"/vllm/{m['port']}/v1/models"}
        for m in cfg.load_models()
    ]
    return JSONResponse({"servers": out})


@app.get("/vllm/{port}/v1/models")
def server_models(port: int) -> JSONResponse:
    """Proxy ONE vLLM server's /v1/models so every server has an independently
    reachable /v1/models URL through the single public tunnel (§6.3 / §2.2). The port
    is validated against the configured line-up (no arbitrary forwarding). Works even
    while that server is swapped to sleep — the model list is metadata, not weights."""
    for m in cfg.load_models():
        if m["port"] == port:
            client = LLMClient(base_url=m["base_url"], model=m["id"])
            try:
                return JSONResponse(client.models())
            except Exception as exc:  # server briefly down / unreachable
                return JSONResponse(
                    {"object": "list", "data": [], "error": str(exc)}, status_code=502)
    return JSONResponse({"error": f"unknown vLLM server port {port}"}, status_code=404)


def _handle_one(raw: Any) -> PredictResult:
    if not isinstance(raw, dict):
        return empty_result("", "Malformed query: expected a JSON object.")
    try:
        q = PredictQuery(**raw)
    except Exception as exc:
        return empty_result(str(raw.get("query_id", "")), f"Could not parse query: {exc}")

    qtype = (q.type or "").strip().lower()
    t0 = time.perf_counter()

    def _physics() -> PredictResult:
        # Physics talks to the primary GENERATOR; make sure it is awake (not slept
        # under a judge swap from a prior query) before calling it.
        get_manager().ensure_generators()
        return get_physics().answer(q)

    try:
        if qtype == "type2":
            result = _physics()
        elif qtype == "type1":
            result = answer_type1(get_judges(), q)
        else:
            # Unknown/blank type: route by shape — premises/options present -> logic.
            if q.options or q.premises:
                result = answer_type1(get_judges(), q)
            else:
                result = _physics()
    except Exception as exc:  # never let one query take down the batch
        log.exception("query %s failed", q.query_id)
        result = empty_result(q.query_id, f"Internal error: {exc}")

    result.query_id = q.query_id or result.query_id
    dt = time.perf_counter() - t0
    log.info("query_id=%s type=%s answer=%r unit=%r premises_used=%s %.2fs",
             q.query_id, qtype, result.answer, result.unit, result.premises_used, dt)
    return result


@app.post("/predict")
def predict(payload: Any = Body(...)) -> JSONResponse:
    """Accepts a single query object (the competition default) or a list of them.
    Always returns a JSON list of result objects."""
    items = payload if isinstance(payload, list) else [payload]
    results: List[dict] = [
        _handle_one(item).model_dump(exclude_none=False) for item in items
    ]
    return JSONResponse(results)
