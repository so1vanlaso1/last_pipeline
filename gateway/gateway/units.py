"""Answer / unit canonicalization helpers for the competition output.

* Type 2 units must be ASCII (Section 4.2 / 5): ohm not Omega, uF not microF, etc.
* Type 1 choice answers must be EXACTLY one of the provided options.
* Type 1 premises_used must be 0-based indices into the input `premises` array.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Superscripts/subscripts → ASCII digits, and a few non-ASCII operators.
_SUPERSCRIPT = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
    "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
    "₇": "7", "₈": "8", "₉": "9",
})

# Greek / symbol → ASCII for the unit field. Order matters (multi-char first).
_UNIT_REPLACEMENTS = [
    ("Ω", "ohm"), ("ω", "ohm"), ("Ω".lower(), "ohm"),
    ("µ", "u"), ("μ", "u"),         # micro prefix → u  (uF, uC, uH, uJ)
    ("×", "x"), ("·", "."), ("−", "-"), ("–", "-"),
    ("Ω", "ohm"),
]

_UNIT_WORD_FIX = {
    "ohms": "ohm", "volts": "V", "volt": "V", "amperes": "A", "ampere": "A",
    "amps": "A", "amp": "A", "watts": "W", "watt": "W", "joules": "J",
    "joule": "J", "coulombs": "C", "coulomb": "C", "farads": "F", "farad": "F",
    "henries": "H", "henry": "H", "teslas": "T", "tesla": "T", "newtons": "N",
    "newton": "N", "hertz": "Hz", "seconds": "s", "second": "s",
    "meters": "m", "meter": "m", "metres": "m", "metre": "m",
    "microfarad": "uF", "microfarads": "uF", "millihenry": "mH",
    "millihenries": "mH", "microhenry": "uH", "microhenries": "uH",
}


def to_ascii_unit(unit: Optional[str]) -> str:
    """Render a unit in ASCII for the Type 2 `unit` field. Empty -> ''."""
    if unit is None:
        return ""
    u = str(unit).strip()
    if not u or u.lower() in {"none", "null"}:
        return ""
    u = u.translate(_SUPERSCRIPT)
    for a, b in _UNIT_REPLACEMENTS:
        u = u.replace(a, b)
    low = u.lower()
    if low in _UNIT_WORD_FIX:
        return _UNIT_WORD_FIX[low]
    # Normalize spacing around a fraction slash (V / m -> V/m).
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"\s+", " ", u).strip()
    return u


# ── Type 1 option matching ───────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_UNCERTAIN_WORDS = {
    "uncertain", "unknown", "notgiven", "cannotbedetermined", "cannotdetermine",
    "undetermined", "insufficient", "noneoftheabove", "none", "na", "nei",
    "notenoughinformation",
}


_YES_WORDS = {"yes", "true", "correct", "entailed", "valid"}
_NO_WORDS = {"no", "false", "incorrect", "contradicted", "invalid"}


def find_uncertain_option(options: List[str]) -> Optional[int]:
    """Index of the option that expresses 'uncertain / not given', if any."""
    for i, o in enumerate(options):
        if _norm(o) in _UNCERTAIN_WORDS:
            return i
    return None


def find_yes_option(options: List[str]) -> Optional[int]:
    for i, o in enumerate(options):
        if _norm(o) in _YES_WORDS:
            return i
    return None


def find_no_option(options: List[str]) -> Optional[int]:
    for i, o in enumerate(options):
        if _norm(o) in _NO_WORDS:
            return i
    return None


def looks_like_ynn(options: List[str]) -> bool:
    """True if the option set is a Yes/No(/Uncertain)-style entailment choice —
    routed to the cascade's purpose-built Yes/No/Not-Given examiner rather than the
    generic multiple-choice one."""
    if not (2 <= len(options) <= 3):
        return False
    return find_yes_option(options) is not None and find_no_option(options) is not None


def map_letter_to_option(letter: Optional[str], options: List[str]) -> Optional[str]:
    """'A'..'H' -> the exact option string at that 0-based position."""
    if not letter or len(letter) != 1 or not letter.isalpha():
        return None
    idx = ord(letter.upper()) - 65
    if 0 <= idx < len(options):
        return options[idx]
    return None


def match_text_to_option(text: str, options: List[str]) -> Optional[str]:
    """Match a free-text answer to one of the options (exact normalized match)."""
    nt = _norm(text)
    if not nt:
        return None
    for o in options:
        if _norm(o) == nt:
            return o
    return None


# ── Type 1 premises_used extraction (0-based) ────────────────────────────────
_PREMISE_CITE = re.compile(r"premise[s]?\s*#?\s*([0-9]+(?:\s*(?:,|and|&|to|-|–|through)\s*[0-9]+)*)", re.I)
_NUM = re.compile(r"[0-9]+")


def premises_from_text(text: str, n_premises: int) -> List[int]:
    """Parse 1-based 'premise N' citations from a WHY/explanation into 0-based
    indices, clamped to the available range. Handles 'premises 1 and 2',
    'premises 1-3', 'premise 7'."""
    if not text or n_premises <= 0:
        return []
    found: set[int] = set()
    for m in _PREMISE_CITE.finditer(text):
        span = m.group(1)
        nums = [int(x) for x in _NUM.findall(span)]
        if not nums:
            continue
        # A connector implying a range ("1-3", "1 to 3", "1 through 3").
        if len(nums) == 2 and re.search(r"(?:-|–|to|through)", span, re.I) and nums[0] <= nums[1]:
            rng = range(nums[0], nums[1] + 1)
        else:
            rng = nums
        for one_based in rng:
            zero = one_based - 1
            if 0 <= zero < n_premises:
                found.add(zero)
    return sorted(found)


def clamp_indices(values, n_premises: int) -> List[int]:
    """Coerce an arbitrary list of indices (assumed already 0-based) into a sorted,
    deduped, in-range list."""
    out: set[int] = set()
    for v in values or []:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= iv < n_premises:
            out.add(iv)
    return sorted(out)
