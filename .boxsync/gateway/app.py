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
from typing import Any, List, Tuple

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from . import config as cfg
from .logic_adapter import answer_type1
from .physics_adapter import PhysicsAdapter
from .residency import get_manager
from .schema import PredictQuery, PredictResult, empty_result
from .vllm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

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
            log.warning("resident models total %gB EXCEEDS the committee's 8B-at-any-moment "
                        "limit (allowed locally by max_resident_b=%g in serve/logic_config.yaml "
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
    c = primary_client()
    return {
        "status": "ok",
        "mode": c.mode,
        "models": [j[0].model for j in get_judges()],
    }


@app.get("/v1/models")
def models() -> JSONResponse:
    """Aggregate every resident server's /v1/models so a single host verifies all
    declared models (and shows the line-up is <= 8B)."""
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
