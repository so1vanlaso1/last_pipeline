"""Best-effort NVIDIA GPU memory introspection via `nvidia-smi` (no extra deps).

Used by GET /health to make the residency swap VERIFIABLE: it reports total VRAM
in use and the per-process VRAM of each vLLM server (mapped from the pidfiles that
run_server.sh writes). A slept model then shows only its small CUDA-context
residual — not its weights — so the committee can confirm at a glance that
≤ 8B of model weights are resident-and-running at the inspected instant
(Submission Guide §6.3, "the committee may inspect GPU memory usage").

Everything here is best-effort: on a host without `nvidia-smi` (e.g. a CPU dev box
or the stub backend) every call returns None / {} and /health still works.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _smi(args: List[str]) -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def overall() -> Optional[Dict[str, int]]:
    """Whole-GPU memory: {"used_mib", "total_mib"} (first GPU), or None."""
    s = _smi(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    if not s:
        return None
    try:
        used, total = (int(x.strip()) for x in s.splitlines()[0].split(","))
        return {"used_mib": used, "total_mib": total}
    except Exception:
        return None


def by_pid() -> Dict[int, int]:
    """Map each GPU compute process PID -> its used VRAM in MiB ({} if unavailable)."""
    s = _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    out: Dict[int, int] = {}
    if not s:
        return out
    for line in s.splitlines():
        try:
            pid_s, mem_s = line.split(",")
            out[int(pid_s.strip())] = int(mem_s.strip())
        except Exception:
            continue
    return out


def pid_from_portfile(logdir: Path, port: int) -> Optional[int]:
    """The vLLM server PID for a port, read from run_server.sh's vllm_<port>.pid."""
    try:
        return int((Path(logdir) / f"vllm_{port}.pid").read_text().strip())
    except Exception:
        return None
