from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2) or ""
            return os.environ.get(key, default)
        expanded = _ENV_PATTERN.sub(repl, value)
        if expanded.lower() in {"true", "false"}:
            return expanded.lower() == "true"
        try:
            if "." in expanded:
                return float(expanded)
            return int(expanded)
        except ValueError:
            return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class Settings:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.raw.get("pipeline", {})

    @property
    def fama(self) -> dict[str, Any]:
        return self.raw.get("fama", {})


def load_settings(path: str | None = None) -> Settings:
    load_dotenv(override=False)
    config_path = Path(path or os.environ.get("EXACT_FAMA_CONFIG", "configs/default.yaml"))
    if not config_path.exists():
        return Settings(raw={})
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(raw=_expand_env(data))
