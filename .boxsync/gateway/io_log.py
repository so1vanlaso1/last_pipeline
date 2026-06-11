"""Append-only model I/O log for the live gateway.

The offline logic pipeline writes Result/log.txt at the end of a run. The gateway
serves requests continuously, so it appends each Type 1 model call to
serve/logs/log.txt as the calls happen.
"""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import config as cfg

_LOCK = threading.Lock()


def log_path() -> Path:
    path = Path(os.environ.get("LOGIC_IO_LOG", str(cfg.REPO_ROOT / "serve" / "logs" / "log.txt")))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model_label(client) -> str:
    model = getattr(client, "model", "unknown")
    base_url = getattr(client, "base_url", "")
    return f"{model} ({base_url})" if base_url else str(model)


def model_labels(clients: Iterable) -> list[str]:
    return [model_label(c) for c in clients]


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        return "nvidia-smi unavailable"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi error: {type(exc).__name__}: {exc}"
    text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"nvidia-smi failed ({proc.returncode}): {text or 'no output'}"
    return text or "(no output)"


def vram_snapshot() -> str:
    gpu = _run_nvidia_smi([
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    procs = _run_nvidia_smi([
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    return f"GPU memory (index, name, used MiB, total MiB):\n{gpu}\nProcesses (pid, name, used MiB):\n{procs}"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines()) or (prefix + "(empty)")


def append_model_io(
    *,
    context: str,
    model: str,
    loaded_models: list[str],
    system: str,
    user: str,
    raw: str,
    elapsed_s: float,
    error: str | None = None,
) -> None:
    lines = [
        "",
        "=" * 78,
        f"{datetime.now():%Y-%m-%d %H:%M:%S}  {context}",
        f"MODEL: {model}  [{elapsed_s:.2f}s]",
        "VRAM LOADED MODELS AT CALL:",
    ]
    if loaded_models:
        lines.extend(f"  - {name}" for name in loaded_models)
    else:
        lines.append("  (not recorded)")
    lines.extend([
        "VRAM SNAPSHOT BEFORE GENERATION:",
        _indent(vram_snapshot()),
        "IN (system):",
        _indent(system),
        "IN (user):",
        _indent(user),
        "OUT (raw):",
        _indent(raw),
    ])
    if error:
        lines.extend(["ERROR:", _indent(error)])
    with _LOCK:
        with log_path().open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
