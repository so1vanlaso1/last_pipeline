from __future__ import annotations

from typing import Any


def select_dominant_failures(report: dict[str, Any], top_k: int = 3) -> list[str]:
    top = report.get("top_failures") or []
    return [name for name, _count in top[:top_k]]
