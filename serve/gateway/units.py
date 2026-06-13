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


def to_ascii_answer(answer: Optional[str]) -> str:
    """Render a Type 2 numeric answer in ASCII (Section 4.2 / 5 require ASCII output).

    Only the unit field was being normalized; a solver/LLM answer like '2.5 × 10^7'
    or '1.23×10^-4' otherwise ships the non-ASCII '×' (U+00D7) and would miss an
    exact-match / ASCII grader. We map the multiply sign to 'x' (the same choice the
    unit normalizer's _UNIT_REPLACEMENTS already uses) and the few other non-ASCII
    operators to ASCII. The exponent is left in caret form ('10^7'); superscripts are
    deliberately NOT collapsed here (turning '10⁷' into '107' would drop the
    exponent) — the deterministic solver emits caret+ASCII exponents already."""
    if answer is None:
        return ""
    a = str(answer).strip()
    if not a:
        return ""
    a = (a.replace("×", "x").replace("·", ".")
          .replace("−", "-").replace("–", "-"))
    return a.strip()


# ── LaTeX → ASCII pre-extraction normalization (Type 2 physics) ──────────────
# The committee sends problems in LaTeX; the deterministic physics extractor parses
# ASCII (e-notation, 'ohm', 'uF', 'R1'). This pass converts the LaTeX our notation
# map declares (serve/submission/notation_mapping.csv) so the pipeline parses
# correctly EVEN IF the committee's regex substitution misses something. It also
# fixes the cases a CSV cell cannot: numeric \frac, '\mu F' spacing, and Unicode
# subscripts (R₁). Order-locked — scientific notation is converted FIRST so a generic
# multiply can never turn '3 \times 10^{-6}' into '3*10^{-6}' (which the extractor
# rejects). Idempotent: already-ASCII competition input passes through unchanged.
_SUBSCRIPT_DIGITS = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})

# \greek -> spelled-out ASCII (varepsilon/varphi before epsilon/phi; longest first).
_GREEK = [
    (r"\\varepsilon", "epsilon"), (r"\\epsilon", "epsilon"),
    (r"\\varphi", "phi"), (r"\\Phi", "Phi"), (r"\\phi", "phi"),
    (r"\\alpha", "alpha"), (r"\\beta", "beta"), (r"\\gamma", "gamma"),
    (r"\\Delta", "Delta"), (r"\\delta", "delta"), (r"\\theta", "theta"),
    (r"\\lambda", "lambda"), (r"\\sigma", "sigma"), (r"\\tau", "tau"),
    (r"\\rho", "rho"), (r"\\pi", "pi"),
    (r"\\nabla", "nabla"), (r"\\partial", "partial"),
    (r"\\int", "integral"), (r"\\sum", "sum"),
]


def _frac_sub(m: "re.Match") -> str:
    """Numeric \\frac{a}{b} -> its decimal (so the value extractor sees a number);
    a non-numeric fraction becomes (a)/(b) for the symbolic evaluators."""
    a, b = m.group(1), m.group(2)
    try:
        v = float(a) / float(b)
        return str(int(v)) if v == int(v) else repr(v)
    except (ValueError, ZeroDivisionError):
        return f"({a})/({b})"


def latex_to_ascii(text: Optional[str]) -> str:
    """Normalize a LaTeX physics problem into the ASCII the extractor parses."""
    if not text:
        return text or ""
    t = str(text)
    # 1. Scientific notation FIRST: <num> (\times|\cdot|×|·|x|*) 10^{n} -> <num>e<n>.
    #    Anchored to a preceding digit so 'max 10^3' is untouched.
    t = re.sub(
        r"(\d)\s*(?:\\times|\\cdot|×|·|[xX*])\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        r"\1e\2", t)
    # 2. Micro prefix glued to its unit: \mu F / µ F / μ F -> uF ; standalone -> u.
    t = re.sub(r"(?:\\mu|µ|μ)\s*([FCHJAVWgsm])", r"u\1", t)
    t = re.sub(r"(?:\\mu|µ|μ)", "u", t)
    # 3. Ohm unit: \Omega / Ω -> ohm ; \omega right after a number is an ohm unit.
    t = re.sub(r"\\Omega|Ω", "ohm", t)
    t = re.sub(r"(?<=\d)\s*\\omega\b", " ohm", t)
    t = re.sub(r"\\omega", "omega", t)
    # 4. Subscripts: R₁ / R_1 / R_{1} -> R1 (letter + digits only).
    t = t.translate(_SUBSCRIPT_DIGITS)
    t = re.sub(r"([A-Za-z])_\{?(\d+)\}?", r"\1\2", t)
    # 5. Numeric \frac{a}{b} -> decimal.
    t = re.sub(r"\\frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}",
               _frac_sub, t)
    # 6. Roots and remaining operators (sci-notation already handled above).
    t = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", t)
    t = t.replace(r"\sqrt", "sqrt")
    for a, b in ((r"\leq", "<="), (r"\geq", ">="), (r"\neq", "!="),
                 (r"\approx", "~"), (r"\propto", "~"), (r"\div", "/"),
                 (r"\pm", "+/-"), (r"\mp", "-/+"), (r"\infty", "infinity"),
                 (r"\degree", "deg"), (r"\angle", "angle"),
                 (r"\times", "*"), (r"\cdot", "*")):
        t = t.replace(a, b)
    # 7. Greek symbols -> spelled out; drop \vec{}/\hat{} decoration (keep the symbol).
    for pat, word in _GREEK:
        t = re.sub(pat, word, t)
    t = re.sub(r"\\(?:vec|hat)\s*\{([^}]*)\}", r"\1", t)
    return t


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
