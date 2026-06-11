"""Sleep/wake residency manager for the Type 1 (logic) swap line-up.

The 2x4B generators + 8B judge line-up needs ~40 GB if held all-resident. To run
it on a 24 GB card we keep the two 4B GENERATORS co-resident (their combined
weights fit) and SWAP the 8B JUDGE in only for its arbitration call:

    resting state            -> generators AWAKE,  judge ASLEEP
    judge() context entered  -> generators ASLEEP, judge AWAKE   (one query)
    judge() context exited   -> generators AWAKE,  judge ASLEEP  (back to resting)

so peak GPU usage is max(generators, judge) ~17 GB, not their sum. The swap uses
vLLM's sleep mode (level 1: weights offload to CPU RAM, KV cache freed; wake_up
restores them in a few seconds — far cheaper than a cold start). Both servers must
be launched with `--enable-sleep-mode`, and run_server.sh sets VLLM_SERVER_DEV_MODE=1
so the /sleep and /wake_up admin endpoints exist. run_server.sh also leaves the
line-up in the resting state (judge slept right after it loads), which this
manager assumes on startup.

Discipline that keeps the 24 GB card from OOMing: a group is only ever woken once
the other group is asleep (sleep-before-wake), and a process-wide lock serialises
the judge phase so two queries can't both wake the judge. The committee drives the
endpoint sequentially (see gateway/app.py), so this lock is essentially free.

A line-up without a `judge` role, with swap disabled, or in `vote` mode degrades
to a no-op (`enabled=False`): every model is simply resident, as before.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import List, Optional

import requests

from . import config as cfg

log = logging.getLogger("gateway.residency")

# Sleep/wake are blocking on the vLLM side; give them generous headroom (a cold
# wake from CPU offload is seconds, not minutes, but the first one can recompile).
_OP_TIMEOUT = float(os.environ.get("RESIDENCY_OP_TIMEOUT", "180"))
_SLEEP_LEVEL = int(os.environ.get("RESIDENCY_SLEEP_LEVEL", "1"))


def _root(base_url: str) -> str:
    """Server root (admin endpoints live there, not under /v1)."""
    r = (base_url or "").rstrip("/")
    if r.endswith("/v1"):
        r = r[: -len("/v1")]
    return r.rstrip("/")


class Residency:
    def __init__(self, generator_urls: List[str], judge_url: Optional[str], enabled: bool):
        self._gens = [_root(u) for u in generator_urls]
        self._judge = _root(judge_url) if judge_url else None
        self.enabled = bool(enabled and self._judge and self._gens)
        self._lock = threading.RLock()
        # run_server.sh leaves the judge asleep + generators awake.
        self._judge_awake = False

    # ── low-level vLLM admin calls ───────────────────────────────────────────
    def _post(self, root: str, path: str, **params) -> bool:
        try:
            resp = requests.post(root + path, params=params or None, timeout=_OP_TIMEOUT)
            resp.raise_for_status()
            return True
        except Exception as exc:  # a failed swap is logged, never fatal to the query
            log.warning("residency %s%s failed: %s", root, path, exc)
            return False

    def _sleep(self, root: str) -> None:
        self._post(root, "/sleep", level=_SLEEP_LEVEL)

    def _wake(self, root: str) -> None:
        self._post(root, "/wake_up")

    # ── orchestration ────────────────────────────────────────────────────────
    def ensure_generators(self) -> None:
        """Put the line-up into the resting state (generators awake, judge asleep)
        before a generation phase. Cheap no-op when already resting."""
        if not self.enabled:
            return
        with self._lock:
            if self._judge_awake:
                self._sleep(self._judge)
                self._judge_awake = False
            for g in self._gens:
                self._wake(g)

    @contextmanager
    def judge(self):
        """Swap the judge in for the duration of the block: sleep the generators,
        wake the judge; on exit sleep the judge and wake the generators back to the
        resting state. The lock serialises this across concurrent queries."""
        if not self.enabled:
            yield
            return
        with self._lock:
            for g in self._gens:
                self._sleep(g)
            self._wake(self._judge)
            self._judge_awake = True
            try:
                yield
            finally:
                self._sleep(self._judge)
                self._judge_awake = False
                for g in self._gens:
                    self._wake(g)


_manager: Optional[Residency] = None


def get_manager() -> Residency:
    """Lazy process-wide singleton built from serve/logic_config.yaml roles."""
    global _manager
    if _manager is None:
        models = cfg.load_models()
        # No sleep/wake against the stub backend (no real vLLM servers to swap).
        stub = os.environ.get("GATEWAY_LLM", "vllm").lower() == "stub"
        active = cfg.swap_active(models) and not stub
        gens = [m["base_url"] for m in models if m.get("role") != "judge"]
        judge = next((m["base_url"] for m in models if m.get("role") == "judge"), None)
        _manager = Residency(generator_urls=gens, judge_url=judge, enabled=active)
        if _manager.enabled:
            log.info("residency: SWAP on — generators co-resident, judge swapped in "
                     "per query (sleep level %d)", _SLEEP_LEVEL)
        else:
            log.info("residency: SWAP off — line-up held all-resident")
    return _manager
