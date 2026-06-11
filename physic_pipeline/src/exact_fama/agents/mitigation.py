from __future__ import annotations

from .failure_taxonomy import MITIGATION_MAP, FailureType


def build_mitigation_config(failure_names: list[str]) -> dict[str, list[str]]:
    config: dict[str, list[str]] = {}
    for name in failure_names:
        try:
            ft = FailureType(name)
        except ValueError:
            continue
        config[name] = MITIGATION_MAP.get(ft, [])
    return config
