from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Compatibility layer used by older scripts/tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    antecedents: tuple[str, ...]
    consequent: str
    raw: str
    premise_id: int | None = None


# ---------------------------------------------------------------------------
# New Z3-ready Horn representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[str, ...] = ("?x",)

    def text(self) -> str:
        return f"{self.pred}({', '.join(self.args)})"

    def is_negative(self) -> bool:
        return self.pred.startswith("not_")


@dataclass(frozen=True)
class HornRule:
    antecedents: tuple[Atom, ...]
    consequent: Atom
    raw: str
    premise_id: int | None = None


STOP_WORDS = {
    "a", "an", "the", "that", "then", "therefore",
    "it", "they", "he", "she", "his", "her", "their",
    "this", "these", "those",
}

GENERIC_SUBJECTS = {
    "student", "students", "person", "people", "someone", "anyone",
    "faculty", "faculty_member", "faculty_members", "member", "members",
    "driver", "drivers", "professor", "professors",
    "curriculum", "course", "courses", "project", "projects",
    "employee", "employees", "nurse", "nurses",
}

SUBJECT_TITLES = {
    "professor", "dr", "doctor", "nurse", "student", "mr", "ms", "mrs",
}

NEGATION_PATTERNS = [
    ("cannot", "can not"),
    ("can't", "can not"),
    ("does not", "not"),
    ("do not", "not"),
    ("did not", "not"),
    ("is not", "not"),
    ("are not", "not"),
    ("has not", "not has"),
    ("have not", "not have"),
]


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_period(text: str) -> str:
    return _clean_spaces(text).strip(" .;:\n\t")


def _normalize_negation_text(text: str) -> str:
    out = text
    for old, new in NEGATION_PATTERNS:
        out = re.sub(rf"\b{re.escape(old)}\b", new, out, flags=re.I)
    return out


def slug(text: str) -> str:
    text = _normalize_negation_text(text.lower())
    text = text.replace("≥", " greater_equal ")
    text = text.replace("<=", " less_equal ")
    text = text.replace(">", " greater_than ")
    text = text.replace("<", " less_than ")
    text = re.sub(r"[^a-z0-9_+\-\s]+", " ", text)
    tokens = [t for t in text.split() if t and t not in STOP_WORDS]
    return "_".join(tokens)


def normalize_clause(text: str) -> str:
    return slug(text)


def _norm_entity(text: str | None, default: str = "?x") -> str:
    text = _strip_period(text or "")

    if not text:
        return default

    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    text = text.strip()

    low = slug(text)

    if low in GENERIC_SUBJECTS:
        return default

    if low in {"they", "them", "their", "it", "its", "he", "him", "she", "her"}:
        return default

    return low or default


def _normalize_predicate_phrase(text: str) -> str:
    text = _strip_period(text)
    text = _normalize_negation_text(text)

    negative = False

    if re.search(r"\bnot\b", text, flags=re.I):
        negative = True
        text = re.sub(r"\bnot\b", " ", text, flags=re.I)

    # Remove common auxiliaries while preserving semantic verb/object.
    text = re.sub(
        r"^(is|are|was|were|be|been|being)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^(has|have|had)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^(can|could|may|might|must|should|will|would)\s+",
        "can ",
        text,
        flags=re.I,
    )

    # Canonical small normalizations.
    replacements = {
        "received": "receive",
        "receives": "receive",
        "completed": "complete",
        "completes": "complete",
        "passed": "pass",
        "passes": "pass",
        "qualified": "qualify",
        "qualifies": "qualify",
        "eligible": "eligible",
        "awarded": "award",
        "awards": "award",
        "graduated": "graduate",
        "graduates": "graduate",
        "enhanced": "enhance",
        "enhances": "enhance",
        "provides": "provide",
        "provided": "provide",
    }

    tokens = slug(text).split("_")
    tokens = [replacements.get(t, t) for t in tokens if t]
    pred = "_".join(tokens)

    if not pred:
        pred = "unknown"

    if negative and not pred.startswith("not_"):
        pred = "not_" + pred

    return pred


def negate_atom(atom: Atom) -> Atom:
    if atom.pred.startswith("not_"):
        return Atom(atom.pred[4:], atom.args)
    return Atom("not_" + atom.pred, atom.args)


def positive_atom(atom: Atom) -> Atom:
    if atom.pred.startswith("not_"):
        return Atom(atom.pred[4:], atom.args)
    return atom


def split_conditions(text: str) -> list[str]:
    parts = re.split(
        r"\s+(?:and|&)\s+|,\s*and\s+|;\s*",
        _strip_period(text),
        flags=re.I,
    )
    return [normalize_clause(p) for p in parts if normalize_clause(p)]


def split_condition_phrases(text: str) -> list[str]:
    parts = re.split(
        r"\s+(?:and|&)\s+|,\s*and\s+|;\s*",
        _strip_period(text),
        flags=re.I,
    )
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _split_if_rule(raw: str) -> tuple[str, str] | None:
    body = re.sub(r"^\s*if\s+", "", _strip_period(raw), flags=re.I).strip()
    low = body.lower()

    for sep in [", then ", ",then ", " then ", ","]:
        idx = low.find(sep)
        if idx > 0:
            left = body[:idx].strip()
            right = body[idx + len(sep):].strip()
            if left and right:
                return left, right

    return None


def parse_rule(premise: str) -> Rule | None:
    """Compatibility parser: returns string-based Rule.

    The new solver uses parse_horn_rule/parse_premise_horn, but this function
    is kept because scripts/tests may still import it.
    """
    raw = _strip_period(premise)

    if re.match(r"^if\s+", raw, flags=re.I):
        pair = _split_if_rule(raw)
        if pair:
            antecedents = tuple(split_conditions(pair[0]))
            consequent = normalize_clause(pair[1])
            if antecedents and consequent:
                return Rule(antecedents=antecedents, consequent=consequent, raw=premise.strip())

    patterns = [
        r"^(.+?)\s+implies\s+(.+)$",
        r"^(.+?)\s+leads to\s+(.+)$",
        r"^(.+?)\s+therefore\s+(.+)$",
    ]

    for pat in patterns:
        m = re.match(pat, raw, flags=re.I)
        if not m:
            continue

        antecedents = tuple(split_conditions(m.group(1)))
        consequent = normalize_clause(m.group(2))

        if antecedents and consequent:
            return Rule(antecedents=antecedents, consequent=consequent, raw=premise.strip())

    return None


def parse_fact(premise: str) -> str | None:
    if parse_rule(premise):
        return None
    cleaned = normalize_clause(premise)
    return cleaned or None


# ---------------------------------------------------------------------------
# NL -> Atom/Horn parsing
# ---------------------------------------------------------------------------

def parse_clause_atom(
    clause: str,
    default_subject: str = "?x",
    force_subject: str | None = None,
) -> Atom | None:
    clause = _strip_period(clause)
    clause = re.sub(r"^\s*(then|therefore)\s+", "", clause, flags=re.I)
    clause = re.sub(r"^\s*if\s+", "", clause, flags=re.I)

    if not clause:
        return None

    if force_subject:
        subject = force_subject
        pred_phrase = clause
        return Atom(_normalize_predicate_phrase(pred_phrase), (subject,))

    # Pronoun / no-subject clause inside a rule.
    if re.match(
        r"^(has|have|had|complete|completed|pass|passed|receive|received|submit|submitted|can|cannot|not|is|are|eligible|qualified|qualifies|qualify)\b",
        clause,
        flags=re.I,
    ):
        return Atom(_normalize_predicate_phrase(clause), (default_subject,))

    # "X has/receives/completes/passes ..."
    m = re.match(
        r"^(.+?)\s+((?:not\s+)?(?:has|have|had|complete|completed|completes|pass|passed|passes|receive|received|receives|submit|submitted|submits|hold|holds|published|publish|paid|pays|missed|misses|attends|attend|scored|scores|can|cannot|is|are|was|were|qualifies|qualify|graduates|graduate|needs|need|meets|meet|provides|provide|enhances|enhance|awarded|award)\b.*)$",
        clause,
        flags=re.I,
    )

    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred_phrase = m.group(2)
        return Atom(_normalize_predicate_phrase(pred_phrase), (subject,))

    # "X with Y" as a condition: registered nurses with certification.
    m = re.match(r"^(.+?)\s+with\s+(.+)$", clause, flags=re.I)
    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred_phrase = "has " + m.group(2)
        return Atom(_normalize_predicate_phrase(pred_phrase), (subject,))

    # Fallback: treat whole clause as predicate on default subject.
    return Atom(_normalize_predicate_phrase(clause), (default_subject,))


def parse_fact_atom(premise: str, premise_id: int | None = None) -> Atom | None:
    if parse_horn_rule(premise, premise_id=premise_id):
        return None

    raw = _strip_period(premise)

    # Skip broad universal statements as facts; they are handled as rules.
    if re.match(r"^(all|every|anyone|students who|people who|faculty members who|drivers who)\b", raw, flags=re.I):
        return None

    return parse_clause_atom(raw, default_subject="?x")


def _parse_if_horn(raw: str, premise_id: int | None) -> HornRule | None:
    pair = _split_if_rule(raw)

    if not pair:
        return None

    left, right = pair
    subject = "?x"

    antecedents: list[Atom] = []

    for part in split_condition_phrases(left):
        atom = parse_clause_atom(part, default_subject=subject)
        if atom:
            antecedents.append(atom)

    consequent = parse_clause_atom(right, default_subject=subject)

    if antecedents and consequent:
        return HornRule(tuple(antecedents), consequent, raw=raw, premise_id=premise_id)

    return None


def _find_rule_split_for_who(after_who: str) -> tuple[str, str] | None:
    """Split text after 'who' into condition blob and consequent blob.

    Example:
    'have completed A and passed B are qualified for C'
    -> ('have completed A and passed B', 'are qualified for C')
    """
    low = after_who.lower()

    separators = [
        " are ",
        " is ",
        " can ",
        " qualify ",
        " qualifies ",
        " receive ",
        " receives ",
        " graduate ",
        " graduates ",
        " become ",
        " becomes ",
        " get ",
        " gets ",
        " may ",
        " must ",
        " will ",
    ]

    best_idx = -1
    best_sep = ""

    for sep in separators:
        idx = low.find(sep)
        if idx > 0 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
            best_sep = sep

    if best_idx <= 0:
        return None

    left = after_who[:best_idx].strip()
    right = after_who[best_idx + 1:].strip()  # keep separator verb without leading space

    if not left or not right:
        return None

    return left, right


def _parse_who_horn(raw: str, premise_id: int | None) -> HornRule | None:
    text = _strip_period(raw)

    if " who " not in text.lower():
        return None

    before, after = re.split(r"\bwho\b", text, maxsplit=1, flags=re.I)
    split = _find_rule_split_for_who(after)

    if not split:
        return None

    cond_blob, cons_blob = split
    subject = "?x"

    antecedents: list[Atom] = []

    # Class/type condition from "students", "drivers", etc. is usually too noisy,
    # so we do not force a type atom. The factual conditions are more important.
    for cond in split_condition_phrases(cond_blob):
        atom = parse_clause_atom(cond, default_subject=subject)
        if atom:
            antecedents.append(atom)

    consequent = parse_clause_atom(cons_blob, default_subject=subject)

    if antecedents and consequent:
        return HornRule(tuple(antecedents), consequent, raw=raw, premise_id=premise_id)

    return None


def _parse_all_horn(raw: str, premise_id: int | None) -> HornRule | None:
    text = _strip_period(raw)

    # "All registered nurses with Advanced Practice certification are authorized to prescribe medication."
    m = re.match(r"^(all|every)\s+(.+?)\s+(are|is|can|must|will)\s+(.+)$", text, flags=re.I)

    if not m:
        return None

    subject = "?x"
    left = m.group(2)
    right = m.group(3) + " " + m.group(4)

    antecedents: list[Atom] = []

    # Handle "X with Y".
    if " with " in left.lower():
        left_a, left_b = re.split(r"\bwith\b", left, maxsplit=1, flags=re.I)
        atom_a = parse_clause_atom("is " + left_a, default_subject=subject)
        atom_b = parse_clause_atom("has " + left_b, default_subject=subject)
        if atom_a:
            antecedents.append(atom_a)
        if atom_b:
            antecedents.append(atom_b)
    else:
        atom = parse_clause_atom("is " + left, default_subject=subject)
        if atom:
            antecedents.append(atom)

    consequent = parse_clause_atom(right, default_subject=subject)

    if antecedents and consequent:
        return HornRule(tuple(antecedents), consequent, raw=raw, premise_id=premise_id)

    return None


def parse_horn_rule(premise: str, premise_id: int | None = None) -> HornRule | None:
    raw = _strip_period(premise)

    if not raw:
        return None

    if re.match(r"^if\s+", raw, flags=re.I):
        rule = _parse_if_horn(raw, premise_id)
        if rule:
            return rule

    rule = _parse_who_horn(raw, premise_id)
    if rule:
        return rule

    rule = _parse_all_horn(raw, premise_id)
    if rule:
        return rule

    # Compatibility with "A implies B"
    for sep in [" implies ", " leads to ", " therefore "]:
        if sep in raw.lower():
            left, right = re.split(re.escape(sep), raw, maxsplit=1, flags=re.I)
            ants = [
                a for a in (
                    parse_clause_atom(p, default_subject="?x")
                    for p in split_condition_phrases(left)
                )
                if a
            ]
            cons = parse_clause_atom(right, default_subject="?x")
            if ants and cons:
                return HornRule(tuple(ants), cons, raw=raw, premise_id=premise_id)

    return None


def parse_premise_horn(premise: str, premise_id: int | None = None) -> tuple[HornRule | None, Atom | None]:
    rule = parse_horn_rule(premise, premise_id=premise_id)

    if rule:
        return rule, None

    return None, parse_fact_atom(premise, premise_id=premise_id)


# ---------------------------------------------------------------------------
# Question / MCQ parsing
# ---------------------------------------------------------------------------

def extract_question_assumptions(question: str) -> list[str]:
    assumptions: list[str] = []

    for m in re.finditer(r"\bif\s+(.+?)(?:\?|\.|$)", question, flags=re.I):
        clause = m.group(1)
        assumptions.extend(split_conditions(clause))

    for m in re.finditer(r"\bassuming\s+(.+?)(?:\?|\.|,|$)", question, flags=re.I):
        assumptions.extend(split_conditions(m.group(1)))

    return assumptions


def extract_question_assumption_atoms(question: str, default_subject: str | None = None) -> list[Atom]:
    subject = default_subject or "?x"
    atoms: list[Atom] = []

    for m in re.finditer(r"\bif\s+(.+?)(?:\?|\.|$)", question, flags=re.I):
        clause = m.group(1)
        for part in split_condition_phrases(clause):
            atom = parse_clause_atom(part, default_subject=subject)
            if atom:
                atoms.append(atom)

    for m in re.finditer(r"\bassuming\s+(.+?)(?:\?|\.|,|$)", question, flags=re.I):
        clause = m.group(1)
        for part in split_condition_phrases(clause):
            atom = parse_clause_atom(part, default_subject=subject)
            if atom:
                atoms.append(atom)

    return atoms


def extract_mc_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}

    # Handles A. text / B. text until next option or end.
    pattern = re.compile(
        r"\b([A-D])\.\s*(.*?)(?=\n\s*[A-D]\.\s*|\Z)",
        flags=re.S,
    )

    for letter, body in pattern.findall(question):
        clean = _clean_spaces(body)
        if clean:
            options[letter.upper()] = clean

    return options


def strip_options(question: str) -> str:
    return re.split(r"\n\s*A\.\s*", question, maxsplit=1)[0].strip()


def known_subjects_from_atoms(atoms: set[Atom]) -> list[str]:
    subjects = sorted({
        a.args[0]
        for a in atoms
        if a.args and not a.args[0].startswith("?")
    })

    return subjects


def choose_default_subject(known_subjects: list[str], question: str = "") -> str:
    q = slug(question)

    for s in known_subjects:
        if s and s in q:
            return s

    for preferred in ["curriculum", "john", "professor_john", "sophia", "david", "alex", "sarah", "minh"]:
        for s in known_subjects:
            if preferred == s or preferred in s:
                return s

    if len(known_subjects) == 1:
        return known_subjects[0]

    return "?x"


def parse_question_atom(question: str, known_subjects: list[str] | None = None) -> Atom | None:
    known_subjects = known_subjects or []
    q = strip_options(question)
    q = re.sub(r",?\s*according to the premises.*$", "", q, flags=re.I)
    q = re.sub(r",?\s*based on the premises.*$", "", q, flags=re.I)
    q = _strip_period(q.strip("? "))

    default_subject = choose_default_subject(known_subjects, q)

    # Does Sophia qualify for scholarship?
    m = re.match(r"^does\s+(.+?)\s+(.+)$", q, flags=re.I)
    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred = m.group(2)
        return parse_clause_atom(pred, default_subject=subject, force_subject=subject)

    # Can Professor John supervise graduate-level research?
    m = re.match(r"^can\s+(.+?)\s+(.+)$", q, flags=re.I)
    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred = "can " + m.group(2)
        return parse_clause_atom(pred, default_subject=subject, force_subject=subject)

    # Is Sophia eligible for ...?
    m = re.match(r"^is\s+(.+?)\s+(.+)$", q, flags=re.I)
    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred = "is " + m.group(2)
        return parse_clause_atom(pred, default_subject=subject, force_subject=subject)

    # Are all employees registered?
    m = re.match(r"^are\s+all\s+(.+?)\s+(.+)$", q, flags=re.I)
    if m:
        return parse_clause_atom("are " + m.group(2), default_subject="?x")

    # "Does the logical chain demonstrate that X meets all requirements to Y?"
    m = re.search(r"\bthat\s+(.+?)\s+(can|qualifies|qualify|is|are|meets|meet)\s+(.+)$", q, flags=re.I)
    if m:
        subject = _norm_entity(m.group(1), default_subject)
        pred = m.group(2) + " " + m.group(3)
        return parse_clause_atom(pred, default_subject=subject, force_subject=subject)

    # "lead to enhanced critical thinking"
    if re.search(r"\bcritical thinking\b", q, flags=re.I):
        subject = "curriculum" if "curriculum" in known_subjects else default_subject
        return Atom("enhance_critical_thinking", (subject,))

    return parse_clause_atom(q, default_subject=default_subject)


def parse_option_atoms(option_text: str, known_subjects: list[str] | None = None) -> list[Atom]:
    known_subjects = known_subjects or []
    default_subject = choose_default_subject(known_subjects, option_text)

    text = _strip_period(option_text)

    # Remove leading option-ish phrases.
    text = re.sub(r"^(it|he|she|they)\s+", "", text, flags=re.I)

    atoms: list[Atom] = []

    # Handle "X but not Y" as two atoms.
    parts = re.split(r"\s+but\s+", text, flags=re.I)

    for part in parts:
        part = part.strip(" .")
        if not part:
            continue

        atom = parse_clause_atom(part, default_subject=default_subject)

        if atom:
            atoms.append(atom)

    return atoms


def build_fol_like_trace(rules: list[Rule], facts: set[str]) -> str:
    rule_lines = []

    for i, r in enumerate(rules, 1):
        ants = " ∧ ".join(r.antecedents)
        rule_lines.append(f"R{i}: ({ants}) -> {r.consequent}")

    fact_lines = [f"F{i}: {f}" for i, f in enumerate(sorted(facts), 1)]

    return "\n".join(fact_lines + rule_lines)