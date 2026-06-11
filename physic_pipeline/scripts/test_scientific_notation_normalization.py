"""Regression tests for school/Vietnamese scientific notation parsing.

Run from repo root after applying/copying the evaluator hotfix:
    python scripts/test_scientific_notation_normalization.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from exact_fama.physics.physics_solvers.shared.base import _parse_number
from exact_fama.utils.answer_normalizer import _try_float, answers_match

EVAL_PATH = ROOT / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", EVAL_PATH)
evaluate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evaluate)

CASES = {
    "1.10^5": 1e5,
    "1 . 10^5": 1e5,
    "1.10^{-5}": 1e-5,
    "1.10-5": 1e-5,
    "1 x 10^5": 1e5,
    "1 × 10^5": 1e5,
    "1*10^5": 1e5,
    "1e5": 1e5,
    "1.1.10^5": 1.1e5,
    "5.2323.10^9": 5.2323e9,
    "10^-6": 1e-6,
    "-10^6": -1e6,
    # These are ordinary decimals, not dot-multiplication notation.
    "0.109": 0.109,
    "1.109": 1.109,
    "1.10": 1.10,
}

for raw, expected in CASES.items():
    got_solver = _parse_number(raw)
    got_norm = _try_float(raw)
    got_eval = evaluate._extract_numbers(raw)[0]
    assert math.isclose(got_solver, expected, rel_tol=1e-12, abs_tol=1e-12), (raw, got_solver, expected, "solver")
    assert got_norm is not None and math.isclose(got_norm, expected, rel_tol=1e-12, abs_tol=1e-12), (raw, got_norm, expected, "answer_normalizer")
    assert math.isclose(got_eval, expected, rel_tol=1e-12, abs_tol=1e-12), (raw, got_eval, expected, "evaluator")

assert answers_match("100000", "1.10^5")
assert evaluate._answers_match_robust("100000", "1.10^5", 0.02)
assert evaluate._answers_match_robust("0.109", "0.109", 0.02)
assert not evaluate._answers_match_robust("0.109", "0.10^9", 0.02)

print("OK: scientific notation and decimal-dot normalization regression tests passed.")
