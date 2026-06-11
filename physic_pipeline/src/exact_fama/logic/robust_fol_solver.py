"""Hybrid general logic solver for EXACT-FAMA.

Design goals:
- no public-set answer/signature calibration;
- use provided FOL when available, with a conservative symbolic verifier;
- use NL/LLM structured parse only as a parser aid, never as final-answer oracle;
- support hidden/general logic forms: unary predicates, simple quantified rules,
  negation, implication, meta-implications, MCQ options, and yes/no statements.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from ..schemas import PredictRequest, SolverResult  # type: ignore
except Exception:  # pragma: no cover
    PredictRequest = Any  # type: ignore
    @dataclass
    class SolverResult:  # type: ignore
        answer: str
        explanation: str = ""
        unit: Any = None
        fol: str | None = None
        cot: list[str] = field(default_factory=list)
        premises: list[str] = field(default_factory=list)
        confidence: float = 0.0
        warnings: list[str] = field(default_factory=list)
        debug: dict[str, Any] = field(default_factory=dict)

# ------------------------------- text helpers -------------------------------

def _clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()

def _norm_text(x: Any) -> str:
    s = str(x or "").lower()
    s = s.replace("’", "'").replace("¬", " not ")
    s = s.replace("≥", " greater_equal ").replace("<=", " less_equal ")
    s = s.replace(">=", " greater_equal ").replace(">", " greater_than ").replace("<", " less_than ")
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

NL_STOP = {
    "a","an","the","that","then","therefore","is","are","was","were","be","been","being",
    "has","have","had","of","to","for","with","and","or","it","its","their","they","them",
    "he","she","his","her","this","these","those","in","on","at","by","from","if","all","every",
    "student","students","person","people","someone","anyone","object","objects","thing","things",
    "based","above","premises","following","statement","true","which","can","could","will","would",
    "must","should","do","does","did","know","we","given","according","about","when","where",
}
STEM = {
    "receives":"receive","received":"receive","receiving":"receive",
    "completed":"complete","completes":"complete","completing":"complete",
    "passed":"pass","passes":"pass","passing":"pass",
    "qualified":"qualify","qualifies":"qualify","qualifying":"qualify",
    "graduated":"graduate","graduates":"graduate","graduating":"graduate",
    "submitted":"submit","submits":"submit","submitting":"submit",
    "attends":"attend","attending":"attend","attended":"attend",
    "studies":"study","studying":"study","studied":"study",
    "uses":"use","using":"use","used":"use",
    "allows":"allow","allowed":"allow","allowing":"allow",
    "requires":"require","required":"require","requiring":"require",
    "needed":"need","needs":"need","eligible":"eligible","registered":"register","registration":"register",
    "selected":"select","chooses":"choose","chosen":"choose","published":"publish","publishes":"publish",
    "awarded":"award","awards":"award","obtained":"obtain","gets":"get","got":"get",
    "participating":"participate","participates":"participate","participated":"participate",
    "training":"train","trained":"train","certification":"certificate","certifications":"certificate",
}

def _nl_tokens(x: Any) -> set[str]:
    toks: set[str] = set()
    for t in _norm_text(x).split():
        if not t or t in NL_STOP:
            continue
        toks.add(STEM.get(t, t))
    return toks

def _pred_slug(name: Any) -> str:
    """Predicate-safe slug. Unlike NL slug, this must preserve A/P/R etc."""
    s = str(name or "").strip()
    s = s.replace("¬", "not_")
    s = re.sub(r"^not\s+", "not_", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()
    return s or "unknown"

def _pred_tokens(pred: str) -> set[str]:
    raw = re.sub(r"^not_", "", pred)
    # Split snake/camel-ish names and keep single-letter predicates as tokens.
    parts = re.split(r"_+", raw)
    out: set[str] = set()
    for p in parts:
        if not p: continue
        for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+|[a-z]+", p):
            tt = STEM.get(t.lower(), t.lower())
            if tt and tt not in NL_STOP:
                out.add(tt)
    if raw and len(raw) == 1:
        out.add(raw)
    return out

def _neg_pred(pred: str) -> str:
    return pred[4:] if pred.startswith("not_") else "not_" + pred

def _neg_key(key: str | None) -> str | None:
    if not key: return None
    if key.startswith(("all:", "exists:")):
        scope, pred = key.split(":", 1)
        return f"{scope}:{_neg_pred(pred)}"
    if key.startswith("ent:"):
        _, ent, pred = key.split(":", 2)
        return f"ent:{ent}:{_neg_pred(pred)}"
    if key.startswith("not:"):
        return key[4:]
    return "not:" + key

def _logical_neg_key(key: str | None) -> str | None:
    """Logical negation of a formula key. For predicate-level rule contraposition, use _neg_key."""
    if not key: return None
    if key.startswith("all:"):
        return "exists:" + _neg_pred(key.split(":",1)[1])
    if key.startswith("exists:"):
        return "all:" + _neg_pred(key.split(":",1)[1])
    if key.startswith("ent:"):
        _, ent, pred = key.split(":", 2)
        return f"ent:{ent}:{_neg_pred(pred)}"
    if key.startswith("not:"):
        return key[4:]
    return "not:" + key

# ------------------------------- FOL parser ---------------------------------

def _normalize_fol(s: Any) -> str:
    out = _clean(s)
    out = out.replace("→", "->").replace("⇒", "->").replace("↔", "<->")
    out = out.replace("∧", "&").replace("∨", "|").replace("¬", " not ")
    out = re.sub(r"ForAll\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,", r"forall \1 (", out, flags=re.I)
    out = re.sub(r"Exists\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,", r"exists \1 (", out, flags=re.I)
    out = re.sub(r"∀\s*([A-Za-z][A-Za-z0-9_]*)", r"forall \1 ", out)
    out = re.sub(r"∃\s*([A-Za-z][A-Za-z0-9_]*)", r"exists \1 ", out)
    out = out.replace("≥", ">=").replace("≤", "<=")
    return _clean(out)

def _strip_outer(s: str) -> str:
    s = _clean(s).strip()
    changed = True
    while changed and s.startswith("(") and s.endswith(")"):
        changed = False; depth = 0; ok = True
        for i,ch in enumerate(s):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s)-1:
                    ok = False; break
                if depth < 0:
                    ok = False; break
        if ok and depth == 0:
            s = s[1:-1].strip(); changed = True
    return s

def _split_top(s: str, op: str) -> tuple[str,str] | None:
    s = _strip_outer(s)
    depth = 0; i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif depth == 0 and s.startswith(op, i):
            return s[:i].strip(), s[i+len(op):].strip()
        i += 1
    return None

def _split_top_multi(s: str, op: str) -> list[str]:
    s = _strip_outer(s)
    depth = 0; start = 0; parts: list[str] = []; i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif depth == 0 and s.startswith(op, i):
            parts.append(s[start:i].strip()); start = i + len(op); i = start; continue
        i += 1
    parts.append(s[start:].strip())
    return [p for p in parts if p]

def _balanced_parens(s: str) -> bool:
    d=0
    for ch in s:
        if ch=='(': d+=1
        elif ch==')':
            d-=1
            if d<0: return False
    return d==0

def _fix_wrapped_quantifier(s: str) -> str:
    """Best-effort repair for ForAll(x, ... ) converted to 'forall x (...' with an extra wrapper."""
    s = _clean(s)
    # Add a closing ')' if the replacement created one missing close.
    # We avoid aggressive repair because malformed official rows should remain conservative.
    if s.count("(") > s.count(")"):
        s += ")" * (s.count("(") - s.count(")"))
    return s

def _atom_preds(expr: str) -> list[tuple[str, tuple[str, ...]]]:
    expr = _normalize_fol(expr)
    out: list[tuple[str, tuple[str, ...]]] = []

    # Comparisons like membership_duration(x) >= 6 become membership_duration_ge_6(x).
    comp_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*(>=|<=|>|<|=)\s*([-+]?[0-9]+(?:\.[0-9]+)?)")
    for name, args_s, op, num in comp_re.findall(expr):
        op_name = {">=":"ge", "<=":"le", ">":"gt", "<":"lt", "=":"eq"}[op]
        pred = _pred_slug(f"{name}_{op_name}_{num.replace('.', '_')}")
        args = tuple(_clean(a) for a in args_s.split(",") if _clean(a))
        out.append((pred, args))

    # Remove comparisons so the functional predicate itself does not get double-counted as a plain atom.
    expr_no_comp = comp_re.sub(" ", expr)
    atom_re = re.compile(r"(?:(not)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")
    for neg, name, args_s in atom_re.findall(expr_no_comp):
        if name.lower() in {"forall", "exists"}:
            continue
        pred = _pred_slug(name)
        if neg:
            pred = _neg_pred(pred)
        args = tuple(_clean(a) for a in args_s.split(",") if _clean(a))
        out.append((pred, args))
    return out

def _arg_is_variable(arg: str) -> bool:
    return bool(re.fullmatch(r"[a-z]", arg.strip()))

def _entity_slug(arg: str) -> str:
    return _pred_slug(arg)

def _atom_key_from_pred(pred: str, args: tuple[str, ...], scope: str = "all") -> str:
    if args and not _arg_is_variable(args[0]):
        return f"ent:{_entity_slug(args[0])}:{pred}"
    return f"{scope}:{pred}"

def _formula_key(expr: str, scope: str = "all") -> str | None:
    s = _strip_outer(_fix_wrapped_quantifier(_normalize_fol(expr)))
    if not s:
        return None

    m = re.match(r"^(forall|exists)\s+([A-Za-z][A-Za-z0-9_]*)\s+(.+)$", s, flags=re.I)
    if m:
        new_scope = "all" if m.group(1).lower() == "forall" else "exists"
        return _formula_key(m.group(3), new_scope)

    # negation. Inside a quantifier, not P(x) means the predicate is negated
    # under the current quantifier; but not ForAll/Exists means logical negation.
    if re.match(r"^not\s+", s, flags=re.I):
        rest = re.sub(r"^not\s+", "", s, flags=re.I).strip()
        simple_atoms = _atom_preds(rest)
        simple_atom_text = bool(len(simple_atoms) == 1 and not any(op in _strip_outer(rest) for op in ["->", "<->", "&", "|"]) and not re.match(r"^(forall|exists)\b", _strip_outer(rest), flags=re.I))
        inner = _formula_key(rest, scope)
        if not inner: return None
        nk = _neg_key(inner) if simple_atom_text else _logical_neg_key(inner)
        return nk if nk else None

    for op, name in [("<->", "iff"), ("->", "imp")]:
        pair = _split_top(s, op)
        if pair:
            left = _formula_key(pair[0], scope)
            right = _formula_key(pair[1], scope)
            if left and right:
                return f"{name}:{left}->{right}"
            return None

    parts = _split_top_multi(s, "&")
    if len(parts) > 1:
        keys = [_formula_key(p, scope) for p in parts]
        if all(keys):
            return "and:" + "&&".join(keys)  # type: ignore[arg-type]
        return None
    parts = _split_top_multi(s, "|")
    if len(parts) > 1:
        keys = [_formula_key(p, scope) for p in parts]
        if all(keys):
            return "or:" + "||".join(keys)  # type: ignore[arg-type]
        return None

    atoms = _atom_preds(s)
    if len(atoms) == 1:
        pred,args = atoms[0]
        return _atom_key_from_pred(pred, args, scope)
    return None

# ------------------------------- KB and closure -----------------------------

@dataclass
class _Rule:
    ants: tuple[str, ...]
    cons: str
    raw: str
    source: str = "fol"

@dataclass
class _KB:
    facts: set[str] = field(default_factory=set)
    rules: list[_Rule] = field(default_factory=list)
    formula_facts: set[str] = field(default_factory=set)
    pred_phrases: dict[str, set[str]] = field(default_factory=dict)
    pred_sources: dict[str, list[str]] = field(default_factory=dict)
    entities: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def add_fact(self, key: str | None, raw: str = "") -> None:
        if not key: return
        self.facts.add(key)
        if key.startswith("all:"):
            # Competition data assumes non-empty educational domain.
            self.facts.add(key.replace("all:", "exists:", 1))
        if key.startswith("ent:"):
            parts = key.split(":", 2)
            if len(parts) == 3:
                self.entities.add(parts[1])
                self.facts.add("exists:" + parts[2])

    def add_rule(self, ants: list[str], cons: str | None, raw: str = "", source: str = "fol") -> None:
        ants = [a for a in ants if a]
        if ants and cons:
            r = _Rule(tuple(ants), cons, raw, source)
            if r not in self.rules:
                self.rules.append(r)

    def add_phrase(self, pred: str, text: str) -> None:
        pred = _pred_slug(pred)
        toks = _nl_tokens(text) | _pred_tokens(pred)
        if toks:
            self.pred_phrases.setdefault(pred, set()).update(toks)
        self.pred_sources.setdefault(pred, [])
        if text and text not in self.pred_sources[pred]:
            self.pred_sources[pred].append(text)

    def all_preds(self) -> set[str]:
        out=set(self.pred_phrases)
        for key in self.facts | self.formula_facts:
            for p in re.findall(r"(?:all|exists):([^:>&|\-]+)|ent:[^:]+:([^:>&|\-]+)", key):
                out.update(x for x in p if x)
        for r in self.rules:
            for k in list(r.ants)+[r.cons]:
                for p in re.findall(r"(?:all|exists):([^:>&|\-]+)|ent:[^:]+:([^:>&|\-]+)", k):
                    out.update(x for x in p if x)
        return out

def _key_to_exists(k: str) -> str:
    return k.replace("all:", "exists:", 1) if k.startswith("all:") else k

def _key_to_ent(k: str, ent: str) -> str:
    if k.startswith("all:"):
        return f"ent:{ent}:{k.split(':',1)[1]}"
    if k.startswith("exists:"):
        return f"ent:{ent}:{k.split(':',1)[1]}"
    return k

def _add_fol_to_kb(kb: _KB, fol: str, nl: str = "") -> None:
    raw = _clean(fol)
    if not raw: return
    s = _strip_outer(_fix_wrapped_quantifier(_normalize_fol(raw)))

    # Split top-level conjunctions as separate premises.
    parts = _split_top_multi(s, "&")
    if len(parts) > 1:
        for part in parts:
            _add_fol_to_kb(kb, part, nl)
        return

    # Quantified formulas.
    m = re.match(r"^(forall|exists)\s+([A-Za-z][A-Za-z0-9_]*)\s+(.+)$", s, flags=re.I)
    if m:
        scope = "all" if m.group(1).lower() == "forall" else "exists"
        body = _strip_outer(m.group(3))
        pair = _split_top(body, "->")
        if pair and scope == "all":
            lhs, rhs = pair
            ants = [_formula_key(x, "all") for x in _split_top_multi(lhs, "&")]
            ants = [a for a in ants if a]
            cons_parts = _split_top_multi(rhs, "&")
            for cp in cons_parts:
                cons = _formula_key(cp, "all")
                if not cons: continue
                kb.add_rule(ants, cons, raw)
                # Existential/concrete propagation for unary rules.
                kb.add_rule([_key_to_exists(a) for a in ants], _key_to_exists(cons), raw)
                # Entity-level propagation for known named facts.
                # Generated dynamically in closure as well.
                if len(ants) == 1:
                    kb.formula_facts.add(f"imp:{ants[0]}->{cons}")
                    # Safe single-antecedent contraposition.
                    na, nc = _neg_key(ants[0]), _neg_key(cons)
                    if na and nc:
                        kb.add_rule([nc], na, raw, source="contrapositive")
                        kb.add_rule([_key_to_exists(nc)], _key_to_exists(na), raw, source="contrapositive")
            return
        # Quantified fact or quantified compound.
        key = _formula_key(body, scope)
        if key:
            kb.add_fact(key, raw)
            if key.startswith(("all:","exists:")):
                kb.add_phrase(key.split(":",1)[1], nl or raw)
        return

    # Meta implication between formulas.
    pair = _split_top(s, "->")
    if pair:
        left = _formula_key(pair[0], "all")
        right = _formula_key(pair[1], "all")
        if left and right:
            kb.formula_facts.add(f"imp:{left}->{right}")
            kb.add_rule([left], right, raw, source="meta")
            return

    key = _formula_key(s, "exists")
    if key:
        kb.add_fact(key, raw)
        if key.startswith(("all:","exists:")):
            kb.add_phrase(key.split(":",1)[1], nl or raw)

_SPLIT_RULE_SEPS = [" implies ", " leads to ", " guarantees ", " ensures ", " grants ", " enables ", " allows ", " causes ", " helps ", " requires ", " require "]

def _split_rule_nl(nl: str) -> tuple[str,str] | None:
    raw = _clean(nl)
    m = re.match(r"if\s+(.+?)(?:,?\s+then\s+|,\s*)(.+)$", raw, flags=re.I)
    if m: return m.group(1), m.group(2)
    m = re.match(r"(.+?)\s+if\s+(.+)$", raw, flags=re.I)
    if m: return m.group(2), m.group(1)
    low = raw.lower()
    for sep in _SPLIT_RULE_SEPS:
        if sep in low:
            a,b = re.split(re.escape(sep.strip()), raw, maxsplit=1, flags=re.I)
            return a,b
    return None

def _learn_phrases_from_fol_nl(kb: _KB, fol: str, nl: str) -> None:
    atoms = _atom_preds(fol)
    if not atoms: return
    # Quantified implication: align antecedent atoms with left NL and consequent atoms with right NL.
    sides = None
    s = _strip_outer(_normalize_fol(fol))
    m = re.match(r"^(forall|exists)\s+\w+\s+(.+)$", s, flags=re.I)
    body = _strip_outer(m.group(2)) if m else s
    pair = _split_top(body, "->")
    split = _split_rule_nl(nl)
    if pair and split:
        lhs_preds = [p for p,_ in _atom_preds(pair[0])]
        rhs_preds = [p for p,_ in _atom_preds(pair[1])]
        left_text, right_text = split
        for p in lhs_preds: kb.add_phrase(p, left_text)
        for p in rhs_preds: kb.add_phrase(p, right_text)
        return
    for pred,_args in atoms:
        kb.add_phrase(pred, nl or fol)

def _build_kb(premises_nl: list[str], premises_fol: list[str], structured_parse: dict[str, Any] | None = None) -> _KB:
    kb = _KB()
    # First load FOL + phrase alignment.
    for i, fol in enumerate(premises_fol):
        nl = premises_nl[i] if i < len(premises_nl) else ""
        _learn_phrases_from_fol_nl(kb, fol, nl)
        _add_fol_to_kb(kb, fol, nl)
    # If FOL absent or incomplete, use structured parse as extra trusted parser output.
    if structured_parse:
        _add_structured_parse(kb, structured_parse)
    # NL facts are important for named examples whose FOL only contains rules.
    _add_nl_facts(kb, premises_nl)
    _add_predicate_alias_rules(kb)
    return kb

def _add_structured_parse(kb: _KB, sp: dict[str, Any]) -> None:
    data = sp.get("data", sp) if isinstance(sp, dict) else {}
    for fact in data.get("facts") or []:
        text = fact.get("fact") if isinstance(fact, dict) else str(fact)
        _add_nl_fact_text(kb, str(text or ""))
    for rule in data.get("rules") or []:
        if not isinstance(rule, dict): continue
        ants_raw = rule.get("if") or rule.get("antecedents") or []
        cons_raw = rule.get("then") or rule.get("consequent") or []
        if isinstance(ants_raw, str): ants_raw=[ants_raw]
        if isinstance(cons_raw, str): cons_raw=[cons_raw]
        ants=[_nl_key(a, kb, "all") for a in ants_raw]
        ants=[a for a in ants if a]
        for c in cons_raw:
            ck=_nl_key(c, kb, "all")
            kb.add_rule(ants, ck, f"structured: {ants_raw} -> {c}", source="structured")

def _entity_from_nl(text: str) -> str | None:
    # Prefer first capitalized proper name, but ignore generic sentence starters.
    ignore = {"All","Every","Everyone","There","If","Based","According","A","An","The","Some","At","No","GPA","GPAs","University","Research","Quantum","Thesis","Scholarship","Scholarships"}
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9_'-]{1,})\b", text):
        w = m.group(1)
        if w not in ignore:
            return _pred_slug(w)
    return None

def _best_preds_for_text(text: str, kb: _KB, *, threshold: float = 0.34, max_preds: int = 3) -> list[tuple[str,float]]:
    tt = _nl_tokens(text)
    low = _norm_text(text)
    text_is_neg = bool(re.search(r"\b(not|no|cannot|can't|won't|doesn't|don't|didn't|without)\b", low))
    out: list[tuple[str,float]] = []
    candidate_preds: set[str] = set()
    for p in kb.all_preds():
        candidate_preds.add(p[4:] if p.startswith("not_") else p)
    for base in candidate_preds:
        # Use both positive and negative phrase evidence, but choose the base predicate;
        # polarity is applied once by _nl_key/_add_nl_fact_text from the text itself.
        cand = set(_pred_tokens(base))
        cand |= kb.pred_phrases.get(base, set())
        cand |= kb.pred_phrases.get("not_" + base, set())
        if not cand:
            continue
        inter = tt & cand
        if not inter:
            continue
        score = len(inter) / max(1, min(len(tt) or 1, len(cand) or 1))
        # More confidence if the exact phrase mapping came from same polarity.
        if text_is_neg and kb.pred_phrases.get("not_" + base): score += 0.06
        if not text_is_neg and kb.pred_phrases.get(base): score += 0.06
        pslug = " ".join(_pred_tokens(base))
        if pslug and pslug in " ".join(tt): score += 0.12
        if len(base) == 1 and not (kb.pred_phrases.get(base) or kb.pred_phrases.get("not_" + base)):
            score *= 0.2
        if score >= threshold:
            out.append((base, score))
    out.sort(key=lambda x: x[1], reverse=True)
    chosen: list[tuple[str,float]]=[]
    used: set[str]=set()
    for p,sc in out:
        toks = _pred_tokens(p)
        if toks and toks <= used and len(chosen) >= 1:
            continue
        chosen.append((p,sc)); used |= toks
        if len(chosen) >= max_preds: break
    return chosen

def _add_nl_fact_text(kb: _KB, text: str) -> None:
    raw = _clean(text).strip(" .")
    if not raw: return
    # Skip rule-like text.
    low = raw.lower()
    if low.startswith("if ") or " then " in f" {low} " or " implies " in f" {low} ":
        return
    neg = bool(re.search(r"\b(not|no|cannot|can't|doesn't|do not|did not|without)\b", low))
    scope = "exists"
    if re.search(r"\b(all|every|everyone|any)\b", low): scope = "all"
    if re.search(r"\b(there exists|exists|some|at least one|a few)\b", low): scope = "exists"
    ent = _entity_from_nl(raw)
    preds = _best_preds_for_text(raw, kb, threshold=0.30 if ent else 0.38, max_preds=4)
    for pred, _score in preds:
        pp = _neg_pred(pred) if neg else pred
        key = f"ent:{ent}:{pp}" if ent else f"{scope}:{pp}"
        kb.add_fact(key, raw)

def _add_nl_facts(kb: _KB, premises_nl: list[str]) -> None:
    for p in premises_nl:
        _add_nl_fact_text(kb, p)

def _add_predicate_alias_rules(kb: _KB) -> None:
    preds = sorted(kb.all_preds())
    for i,p in enumerate(preds):
        if p.startswith("not_"): continue
        pt = _pred_tokens(p)
        if not pt: continue
        for q in preds[i+1:]:
            if q.startswith("not_"): continue
            qt = _pred_tokens(q)
            if not qt: continue
            inter = pt & qt
            if not inter: continue
            # Conservative synonym bridge for generated educational predicates.
            # Examples: eligible_trainer <-> has_trainer; registered_course <-> registered.
            sim = len(inter) / max(len(pt), len(qt))
            if sim >= 0.66 or (len(inter) >= 1 and min(len(pt),len(qt)) == 1 and max(len(pt),len(qt)) <= 2):
                for scope in ["all", "exists"]:
                    kb.add_rule([f"{scope}:{p}"], f"{scope}:{q}", f"alias:{p}<->{q}", source="alias")
                    kb.add_rule([f"{scope}:{q}"], f"{scope}:{p}", f"alias:{p}<->{q}", source="alias")
                # Entity aliases are handled dynamically by closure.

def _closure(kb: _KB, extra: list[str] | None = None, max_steps: int = 100) -> tuple[set[str], list[dict[str, Any]]]:
    facts = set(kb.facts) | set(kb.formula_facts) | set(extra or [])
    proof: list[dict[str, Any]] = []
    # all facts apply to known entities.
    for ent in list(kb.entities):
        for f in list(facts):
            if f.startswith("all:"):
                facts.add(_key_to_ent(f, ent))
    for _ in range(max_steps):
        changed=False
        # Dynamic entity-level rules from universal rules.
        dyn_rules: list[_Rule] = []
        ents = {k.split(":",2)[1] for k in facts if k.startswith("ent:")}
        for r in kb.rules:
            for ent in ents:
                if all(a.startswith(("all:","exists:")) for a in r.ants) and r.cons.startswith(("all:","exists:")):
                    dyn_rules.append(_Rule(tuple(_key_to_ent(a, ent) for a in r.ants), _key_to_ent(r.cons, ent), r.raw, r.source+":entity"))
        for r in kb.rules + dyn_rules:
            if all(_entails_in_facts(a, facts) for a in r.ants) and not _entails_in_facts(r.cons, facts):
                facts.add(r.cons); changed=True
                if r.cons.startswith("all:"):
                    facts.add(r.cons.replace("all:", "exists:", 1))
                    for ent in ents:
                        facts.add(_key_to_ent(r.cons, ent))
                if r.cons.startswith("ent:"):
                    parts = r.cons.split(":",2)
                    facts.add("exists:"+parts[2]); kb.entities.add(parts[1])
                proof.append({"rule": r.raw, "source": r.source, "antecedents": list(r.ants), "conclusion": r.cons})
        if not changed: break
    return facts, proof

def _entails_in_facts(key: str | None, facts: set[str]) -> bool:
    if not key: return False
    if key in facts: return True
    if key.startswith("and:"):
        return all(_entails_in_facts(p, facts) for p in key[4:].split("&&"))
    if key.startswith("or:"):
        return any(_entails_in_facts(p, facts) for p in key[3:].split("||"))
    if key.startswith("not:"):
        inner_neg = _neg_key(key[4:])
        if inner_neg == key:
            return key in facts
        return _entails_in_facts(inner_neg, facts) if inner_neg else False
    if key.startswith("exists:"):
        pred = key.split(":",1)[1]
        return any(f.endswith(":"+pred) and f.startswith("ent:") for f in facts) or key in facts
    if key.startswith("ent:"):
        _, _ent, pred = key.split(":",2)
        return key in facts or f"all:{pred}" in facts
    if key.startswith("imp:"):
        if key in facts: return True
        pair = key[4:].rsplit("->",1)
        if len(pair)!=2: return False
        left,right = pair
        # Deduction theorem check: assume left, close, then see right.
        # This is supplied by _entails using KB, not pure facts.
        return False
    return key in facts

def _entails(kb: _KB, key: str | None, extra: list[str] | None = None) -> bool:
    if not key: return False
    if key.startswith("imp:"):
        facts,_ = _closure(kb, extra)
        if key in facts: return True
        try: left,right = key[4:].rsplit("->",1)
        except ValueError: return False
        return _entails(kb, right, list(extra or []) + [left])
    facts,_ = _closure(kb, extra)
    return _entails_in_facts(key, facts)

def _contradicts(kb: _KB, key: str | None, extra: list[str] | None = None) -> bool:
    return _entails(kb, _logical_neg_key(key), extra)

# ------------------------------- query parser -------------------------------

_OPTION_RE = re.compile(r"\b([A-D])[\.)]\s*(.*?)(?=\n\s*[A-D][\.)]\s*|\Z)", re.S)

def _extract_options(question: str) -> dict[str,str]:
    return {m.group(1).upper(): _clean(m.group(2)) for m in _OPTION_RE.finditer(question or "")}

def _strip_options(question: str) -> str:
    return re.split(r"\n\s*A[\.)]\s*", question or "", maxsplit=1)[0].strip()

def _extract_statement(question: str) -> str:
    q = question or ""
    m = re.search(r"Statement:\s*(.+)$", q, flags=re.I|re.S)
    if m: return _clean(m.group(1).strip(" '\""))
    # Remove options and leading question wrapper.
    q = _strip_options(q)
    q = re.sub(r"^Based on .*?(?:is|are)\s+(?:the\s+)?(?:following\s+)?(?:statement\s+)?(?:true|correct)\??", "", q, flags=re.I|re.S).strip()
    return _clean(q.strip(" ?'\""))

def _looks_fol(text: str) -> bool:
    return bool(re.search(r"[∀∃¬]|\bForAll\b|\bExists\b|->|→|\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)", text or ""))

def _nl_key(text: str, kb: _KB, scope: str = "all") -> str | None:
    raw = _clean(text).strip(" .")
    if not raw: return None
    # Strip wrappers.
    raw = re.sub(r"^(statement:|it is true that|it is indeed true that)\s+", "", raw, flags=re.I).strip()
    low = raw.lower()
    if low.startswith("it is not true that "):
        inner = _nl_key(raw[len("it is not true that "):], kb, scope)
        return "not:" + inner if inner else None
    if "both true and false" in low:
        return "contradiction"
    if _looks_fol(raw):
        k = _formula_key(raw, scope)
        if k: return k
    # NL conditional.
    m = re.match(r"if\s+(.+?)(?:,?\s+then\s+|,\s*)(.+)$", raw, flags=re.I)
    if m:
        left = _nl_key(m.group(1), kb, "all")
        right = _nl_key(m.group(2), kb, "all")
        if left and right: return f"imp:{left}->{right}"
    m = re.match(r"(.+?)\s+if\s+(.+)$", raw, flags=re.I)
    if m:
        left = _nl_key(m.group(2), kb, "all")
        right = _nl_key(m.group(1), kb, "all")
        if left and right: return f"imp:{left}->{right}"
    this_scope = scope
    if re.search(r"\b(all|every|everyone|any)\b", low): this_scope = "all"
    if re.search(r"\b(there exists|exists|some|at least one|a few)\b", low): this_scope = "exists"
    ent = _entity_from_nl(raw)
    best = _best_preds_for_text(raw, kb, threshold=0.26 if ent else 0.31, max_preds=1)
    if not best: return None
    pred = best[0][0]
    if re.search(r"\b(not|no|cannot|can't|won't|doesn't|don't|didn't|without)\b", low):
        pred = _neg_pred(pred)
    if ent and not re.search(r"\b(all|every|everyone|any|there exists|exists|some|at least one)\b", low):
        return f"ent:{ent}:{pred}"
    return f"{this_scope}:{pred}"

def _question_to_key(question: str, kb: _KB) -> str | None:
    st = _extract_statement(question)
    if st and st != _strip_options(question):
        return _nl_key(st, kb, "all")
    q = _strip_options(question).strip(" ?")
    # Direct yes/no forms: remove auxiliary while preserving quantifier words.
    q2 = re.sub(r"^(is|are|do|does|did|can|could|will|would|should|has|have)\s+", "", q, flags=re.I)
    # "does it follow that X" / "is it true that X"
    q2 = re.sub(r"^(it\s+)?(follow|follows|true)\s+that\s+", "", q2, flags=re.I)
    q2 = re.sub(r"^.*?does it follow that\s+", "", q2, flags=re.I)
    return _nl_key(q2, kb, "all")

def _extract_question_assumptions(question: str, kb: _KB) -> list[str]:
    prefix = _strip_options(question)
    out: list[str] = []
    for pat in [r"if we know that (.+?)(?:\?|$)", r"given that (.+?)(?:\?|$)", r"assuming that (.+?)(?:\?|$)"]:
        m = re.search(pat, prefix, flags=re.I|re.S)
        if m:
            k = _nl_key(m.group(1), kb, "all")
            if k: out.append(k)
    return out

def _is_mcq(question: str, answer_hint: str | None = None) -> bool:
    if _extract_options(question): return True
    if answer_hint and str(answer_hint).strip().upper() in {"A","B","C","D"}: return True
    return False

def _is_truth_statement_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\b(is|whether)\s+(?:the\s+)?(?:following\s+)?statement\s+true\b|\bdoes it follow\b|\blogically follow\b", q))

# ------------------------------- inference ----------------------------------


def _structured_data(structured_parse: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(structured_parse, dict):
        return {}
    data = structured_parse.get("data") if isinstance(structured_parse.get("data"), dict) else structured_parse
    return data if isinstance(data, dict) else {}


def _structured_option_clauses(structured_parse: dict[str, Any] | None, letter: str) -> list[str]:
    data = _structured_data(structured_parse)
    opts = data.get("options") or {}
    item = opts.get(letter) if isinstance(opts, dict) else None
    if not isinstance(item, dict):
        return []
    clauses = item.get("query_clauses") or item.get("clauses") or []
    if isinstance(clauses, str):
        clauses = [clauses]
    return [_clean(c) for c in clauses if _clean(c)]


def _structured_query_clauses(structured_parse: dict[str, Any] | None) -> list[str]:
    data = _structured_data(structured_parse)
    q = data.get("query") or {}
    if isinstance(q, str):
        return [_clean(q)] if _clean(q) else []
    if not isinstance(q, dict):
        return []
    vals = q.get("query_clauses") or q.get("clauses") or q.get("claims") or q.get("claim") or q.get("text") or q.get("statement") or []
    if isinstance(vals, str):
        vals = [vals]
    return [_clean(v) for v in vals if _clean(v)]

def _answer_mcq(question: str, kb: _KB, structured_parse: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    opts = _extract_options(question)
    assumptions = _extract_question_assumptions(question, kb)
    parsed: dict[str, Any] = {}; proved=[]; contrad=[]
    for letter,text in opts.items():
        clauses = _structured_option_clauses(structured_parse, letter)
        if clauses:
            keys = [_nl_key(c, kb, "all") for c in clauses]
            keys = [x for x in keys if x]
            k = "and:" + "&&".join(keys) if len(keys) > 1 else (keys[0] if keys else None)
        else:
            k = _nl_key(text, kb, "all")
        parsed[letter] = k
        if k == "contradiction":
            # Only true if we can find any explicit P and not_P in closure.
            facts,_ = _closure(kb, assumptions)
            ok = any(_neg_key(f) in facts for f in facts if f.startswith(("all:","exists:","ent:")))
        elif k:
            ok = _entails(kb, k, assumptions)
        else:
            ok = False
        if ok: proved.append(letter)
        elif k and _contradicts(kb, k, assumptions): contrad.append(letter)
    if len(proved) == 1:
        return proved[0], {"parsed_options": parsed, "proved_options": proved, "contradicted_options": contrad, "assumptions": assumptions, "decision": "unique_proved"}
    return "Unknown", {"parsed_options": parsed, "proved_options": proved, "contradicted_options": contrad, "assumptions": assumptions, "decision": "no_unique_option"}

def _answer_yes_no(question: str, kb: _KB, structured_parse: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    qclauses = _structured_query_clauses(structured_parse)
    if qclauses:
        keys = [_nl_key(c, kb, "all") for c in qclauses]
        keys = [x for x in keys if x]
        key = "and:" + "&&".join(keys) if len(keys) > 1 else (keys[0] if keys else None)
    else:
        key = _question_to_key(question, kb)
    ent = _entails(kb, key) if key else False
    con = _contradicts(kb, key) if key else False
    if ent and not con:
        ans = "Yes"
    elif con and not ent:
        ans = "No"
    else:
        ans = "No" if _is_truth_statement_question(question) else "Unknown"
    return ans, {"query_key": key, "entailed": ent, "contradicted": con, "truth_statement_mode": _is_truth_statement_question(question)}

# ------------------------------- request access -----------------------------

def _extra(request: Any) -> dict[str, Any]:
    return getattr(request, "model_extra", None) or {}

def _get(request: Any, *names: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        for n in names:
            if n in request: return request[n]
        return default
    for n in names:
        if hasattr(request, n): return getattr(request, n)
    ex = _extra(request)
    for n in names:
        if n in ex: return ex[n]
    return default

def _as_str_list(v: Any) -> list[str]:
    if v is None: return []
    if isinstance(v, str): return [v] if v.strip() else []
    if isinstance(v, list): return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []

# ------------------------------- public API ---------------------------------

def solve_logic(request: PredictRequest, structured_parse: dict[str, Any] | None = None) -> SolverResult:
    question = str(_get(request, "question", default="") or "")
    premises_nl = _as_str_list(_get(request, "premises_nl", "premises-NL", default=[]))
    premises_fol = _as_str_list(_get(request, "premises_fol", "premises-FOL", default=[]))

    kb = _build_kb(premises_nl, premises_fol, structured_parse)
    if _is_mcq(question):
        answer, decision_debug = _answer_mcq(question, kb, structured_parse)
    else:
        answer, decision_debug = _answer_yes_no(question, kb, structured_parse)

    facts, proof = _closure(kb)
    warnings = list(kb.warnings)
    if answer == "Unknown":
        warnings.append("LOGIC_UNCERTAIN: no unique entailed answer was found by the symbolic verifier.")

    explanation = _make_explanation(answer, decision_debug, proof)
    confidence = 0.78 if answer not in {"Unknown", "Uncertain"} else 0.42
    if decision_debug.get("entailed") or decision_debug.get("proved_options"):
        confidence = 0.86

    return SolverResult(
        answer=answer,
        unit=None,
        explanation=explanation,
        fol="\n".join(premises_fol) if premises_fol else None,
        cot=[
            "Step 1: Load provided FOL premises and align predicate phrases with NL premises.",
            "Step 2: Add grounded NL facts when the FOL contains only general rules.",
            "Step 3: Close the knowledge base under universal rules, existential propagation, and safe single-rule contraposition.",
            "Step 4: For MCQ, prove each option and select only a unique proved option; for yes/no, check entailment and contradiction.",
            f"Step 5: Final answer = {answer}.",
        ],
        premises=premises_nl,
        confidence=confidence,
        warnings=warnings,
        debug={
            "logic_solver_version": "hybrid_general_v1_no_calibration",
            "decision": decision_debug,
            "facts_preview": sorted(facts)[:160],
            "rule_count": len(kb.rules),
            "proof_steps": proof[:50],
            "predicates": sorted(kb.all_preds()),
            "predicate_phrases": {k: sorted(v) for k,v in kb.pred_phrases.items()},
        },
    )

def _make_explanation(answer: str, dbg: dict[str, Any], proof: list[dict[str, Any]]) -> str:
    if "parsed_options" in dbg:
        proved = dbg.get("proved_options") or []
        if len(proved) == 1:
            return f"The symbolic verifier proved option {proved[0]} from the supplied premises and did not find another uniquely proved option. Therefore, the answer is {answer}."
        return "The symbolic verifier did not find exactly one option entailed by the premises, so the safest answer is Unknown."
    key = dbg.get("query_key")
    if dbg.get("entailed"):
        return f"The supplied premises entail the queried statement ({key}). Therefore, the answer is Yes."
    if dbg.get("contradicted"):
        return f"The supplied premises entail the negation of the queried statement ({key}). Therefore, the answer is No."
    if answer == "No":
        return "The queried statement is not entailed by the supplied premises, so for this truth-checking question the answer is No."
    return "The supplied premises do not entail the queried statement or its negation, so the answer is Unknown."
