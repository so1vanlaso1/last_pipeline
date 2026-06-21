"""Make both sibling pipelines importable no matter where uvicorn is launched.

The deployable lives at <repo>/serve; the two pipelines live at
<repo>/logic_pipeline and <repo>/physic_pipeline. We import:
  * `prompts` + `schema` from logic_pipeline/src  (pure-python: re + dataclasses,
    NO torch) — the cascade's prompt-building + answer-parsing IP, reused verbatim.
  * `exact_fama.*` from physic_pipeline/src        — the physics pipeline, reused
    wholesale for Type 2.

Importing this module (side-effect) inserts those two src dirs on sys.path. It is
imported at the top of the adapters so path setup never depends on PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGIC_SRC = REPO_ROOT / "logic_pipeline" / "src"
PHYSICS_SRC = REPO_ROOT / "physic_pipeline" / "src"


def ensure_paths() -> None:
    for p in (PHYSICS_SRC, LOGIC_SRC):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            # Append (not insert-0) so the pipelines' own modules win over any
            # same-named stdlib shadowing only when they actually exist here.
            sys.path.append(sp)


ensure_paths()
