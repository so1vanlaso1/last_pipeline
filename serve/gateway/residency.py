"""Sleep/wake residency manager for the Type 1 (logic) swap line-up.

The 2x4B generators + 8B judge line-up needs ~40 GB if held all-resident. To run
it on a 24 GB card we keep the two 4B GENERATORS co-resident (their combined
weights fit) and SWAP the 8B JUDGE in only for its arbitration call:

    resting state            -> generators AWAKE,  judge ASLEEP
    judge() context entered  -> generators ASLEEP, judge AWAKE   (one query)
    judge() context exited   -> generators AWAKE,  judge ASLEEP  (back to resting)

so peak GPU usage is max(generators, judge) ~17 GB, not their sum. The swap uses
vLLM's sleep mode. DEFAULT is level 1 (RESIDENCY_SLEEP_LEVEL=1): the slept group's
already-quantized weights are offloaded to CPU RAM and copied back verbatim on wake
(~1s) — LOSSLESS, which the FP8 (8bit) line-up requires. Level 2 (discard + reload
from disk) re-quantizes on wake and corrupts these FP8 models; reserve it for
4bit/bf16. Either level frees the GPU, so only the awake group's weights are on-card.
Both servers must be launched with `--enable-sleep-mode`, and run_server.sh sets
VLLM_SERVER_DEV_MODE=1 so the /sleep and /wake_up admin endpoints exist. run_server.sh
also leaves the line-up in the resting state (judge slept right after it loads),
which this manager assumes on startup.

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
# Level 1 (DEFAULT): offload slept weights to CPU RAM, copy back verbatim on wake —
# fast (~1s) and, crucially, LOSSLESS for the FP8 (8bit) line-up. Level 2 (discard +
# reload-from-disk) re-quantizes on wake and CORRUPTS these FP8 models (garbage "!!!!"
# generations after the first sleep/wake cycle); only use it with 4bit/bf16 weights.
# Either level moves weights OFF the GPU, so the ≤8B-on-GPU rule holds either way.
_SLEEP_LEVEL = int(os.environ.get("RESIDENCY_SLEEP_LEVEL", "1"))
# A /sleep that is not CONFIRMED leaves that group's weights on the GPU, so the swap
# retries before giving up — and if it still cannot confirm, it REFUSES to wake the
# other group. Waking anyway is the only normal path to >8B on the GPU, which would
# fail the committee's GPU-memory inspection (Submission Guide §6.3).
_SLEEP_RETRIES = int(os.environ.get("RESIDENCY_SLEEP_RETRIES", "3"))


class ResidencySwapError(RuntimeError):
    """The swap could not guarantee the ≤8B-on-GPU invariant (a group would not
    confirm asleep). Raised instead of loading more weights on top — the caller
    degrades the query rather than letting the GPU exceed the 8B budget."""


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

    def _is_sleeping(self, root: str) -> Optional[bool]:
        """Best-effort confirmation via vLLM's GET /is_sleeping. Returns None when
        the endpoint is unavailable (older vLLM), so callers fall back to trusting
        the synchronous /sleep response."""
        try:
            resp = requests.get(root + "/is_sleeping", timeout=_OP_TIMEOUT)
            resp.raise_for_status()
            return bool(resp.json().get("is_sleeping", False))
        except Exception:
            return None

    def _sleep(self, root: str) -> bool:
        """Sleep a server and CONFIRM its weights left the GPU. vLLM /sleep is
        synchronous (it returns only after the offload to CPU RAM completes), so a
        200 already means the weights are off the GPU; we retry on failure and, when
        the endpoint exists, double-check /is_sleeping. Returns True ONLY when the
        group is confirmed asleep — the caller must not wake another group otherwise."""
        for attempt in range(1, _SLEEP_RETRIES + 1):
            ok = self._post(root, "/sleep", level=_SLEEP_LEVEL)
            confirmed = self._is_sleeping(root)          # True / False / None(unknown)
            if ok and confirmed is not False:
                return True
            log.warning("residency: %s sleep attempt %d/%d unconfirmed "
                        "(post_ok=%s, is_sleeping=%s)", root, attempt, _SLEEP_RETRIES,
                        ok, confirmed)
        log.critical("residency: could NOT confirm %s asleep after %d attempts — "
                     "REFUSING to wake another group so the GPU stays <= 8B (this "
                     "query degrades to the awake group's answer)", root, _SLEEP_RETRIES)
        return False

    def server_asleep(self, base_url: str) -> Optional[bool]:
        """Public, for /health: is the server at base_url asleep right now? Uses the
        live /is_sleeping when available, else falls back to the known resting state
        (judge asleep unless inside a judge() block; generators awake)."""
        root = _root(base_url)
        live = self._is_sleeping(root)
        if live is not None:
            return live
        if root == self._judge:
            return not self._judge_awake
        return False

    def _wake(self, root: str) -> bool:
        ok = self._post(root, "/wake_up")
        if not ok:  # a model that won't wake is called while asleep -> empty/garbage
            log.error("residency: FAILED to wake %s — its stage may run against a "
                      "sleeping model", root)
        return ok

    # ── orchestration ────────────────────────────────────────────────────────
    def ensure_generators(self) -> None:
        """Put the line-up into the resting state (generators awake, judge asleep)
        before a generation phase. Cheap no-op when already resting (the judge is
        slept at boot and after every judge() block, so normally no sleep is needed).

        Compliance: the judge's 8B must be CONFIRMED off the GPU before the 2×4B
        generators are loaded — otherwise the card briefly holds 8+8=16B. If the
        judge will not sleep we raise instead of co-loading (the query degrades)."""
        if not self.enabled:
            return
        with self._lock:
            if self._judge_awake:
                if not self._sleep(self._judge):
                    raise ResidencySwapError(
                        "judge would not sleep; refusing to also load the generators "
                        "(would exceed 8B on the GPU)")
                self._judge_awake = False
            for g in self._gens:
                self._wake(g)

    @contextmanager
    def judge(self):
        """Swap the judge in for the duration of the block and YIELD whether the
        judge is actually awake. ALL generators must be CONFIRMED asleep before the
        8B judge is woken (else the card holds 4+…+8 > 8B). If any generator will not
        sleep, the judge is NOT woken: we restore the resting state and yield False so
        the caller falls back to the generators' answer — keeping the GPU ≤ 8B. On
        exit the judge sleeps and the generators wake back to the resting state. The
        lock serialises this across concurrent queries."""
        if not self.enabled:
            yield True
            return
        with self._lock:
            slept = [self._sleep(g) for g in self._gens]
            if not all(slept):
                for g, ok in zip(self._gens, slept):     # undo the partial swap
                    if ok:
                        self._wake(g)
                log.critical("residency: not all generators confirmed asleep — judge "
                             "NOT woken; Type 1 uses the generator answer (GPU stays <= 8B)")
                yield False
                return
            self._wake(self._judge)
            self._judge_awake = True
            try:
                yield True
            finally:
                # Symmetric to ensure_generators: only wake the generators once the
                # judge is CONFIRMED back asleep. If its /sleep does not confirm,
                # waking the 4B generators on top of the still-resident 8B judge
                # would put 8+4+4=16B on the GPU (breaks the <=8B rule, risks OOM).
                # Leave _judge_awake=True so the next ensure_generators() retries the
                # sleep (and raises if it still fails) instead of silently co-loading.
                if self._sleep(self._judge):
                    self._judge_awake = False
                    for g in self._gens:
                        self._wake(g)
                else:
                    log.critical(
                        "residency: judge would NOT sleep on judge() exit — NOT waking "
                        "generators (GPU would exceed 8B). Generators stay asleep until "
                        "the next query's ensure_generators() can evict the judge.")


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
