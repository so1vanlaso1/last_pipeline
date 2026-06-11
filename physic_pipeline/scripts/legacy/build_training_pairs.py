#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing the package first.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import hashlib
import json
import random
import re
from typing import Any

from exact_fama.router import route_task
from exact_fama.schemas import PredictRequest
from exact_fama.utils.jsonl import read_jsonl, write_jsonl

try:
    from exact_fama.physics.extractor import extract_quantities
    from exact_fama.physics.formulas import choose_formula
except Exception:  # lets this script run before editable install is refreshed
    extract_quantities = None
    choose_formula = None

SYSTEM_ANSWER = (
    "You are an explainable educational QA assistant. Return valid JSON with answer, "
    "explanation, and optional reasoning fields. Never invent premises, formulas, or units."
)

SYSTEM_PARSE = (
    "You are a strict semantic parser for an explainable educational QA system. "
    "Convert the input into compact JSON for a downstream symbolic solver. "
    "Do not solve unless the field asks for known answer metadata. Preserve negation, units, and premise ids."
)

SYSTEM_REWRITE = (
    "Rewrite solver traces into concise EXACT-style educational explanations. "
    "Keep answer and unit unchanged. Do not invent premises, formulas, or numbers. "
    "Use premise/formula references when available, e.g. 'Premise 4 confirms ...; premise 1 then implies ...'. "
    "Do not copy the draft explanation verbatim. Return JSON only with answer, unit, explanation."
)

WEAK_EXPLANATION_PATTERNS = [
    r"the answer follows from the given premises",
    r"solve the physics problem step by step",
    r"based on the derived conclusions",
    r"the system parsed the premises",
    r"no explanation",
    r"label:\s*",
]


def make_user_content(sample: dict[str, Any]) -> str:
    premises = sample.get("premises-NL") or sample.get("premises_nl") or []
    if premises:
        return "Premises:\n" + "\n".join(f"- {p}" for p in premises) + f"\n\nQuestion: {sample['question']}"
    return f"Question: {sample['question']}"


def stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        return [x]
    return []


def is_weak_explanation(text: Any) -> bool:
    s = stringify(text).strip()
    if len(s.split()) < 12:
        return True
    low = s.lower()
    return any(re.search(pat, low) for pat in WEAK_EXPLANATION_PATTERNS)


def make_assistant_answer(sample: dict[str, Any]) -> str:
    explanation = sample.get("explanation", "")
    if is_weak_explanation(explanation):
        explanation = make_exact_style_explanation(sample)
    payload = {
        "answer": sample.get("answer", ""),
        "unit": sample.get("unit"),
        "explanation": explanation,
        "cot": sample.get("cot", []),
        "premises": sample.get("premises", []),
    }
    return json.dumps(payload, ensure_ascii=False)


def split_logic_conditions(text: str) -> list[str]:
    parts = re.split(r"\s+(?:and|&|,)\s+", text, flags=re.I)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def weak_logic_parse(sample: dict[str, Any]) -> dict[str, Any]:
    premises = sample.get("premises-NL") or sample.get("premises_nl") or []
    premise_map = []
    facts = []
    rules = []
    for idx, p in enumerate(premises, 1):
        raw = stringify(p).strip()
        m = re.match(r"^\s*if\s+(.+?)\s*(?:,\s*)?(?:then\s+)?(.+?)\.?$", raw, flags=re.I)
        if m:
            ants = split_logic_conditions(m.group(1))
            cons = [m.group(2).strip(" .")]
            rule = {"id": f"P{idx}", "if": ants, "then": cons, "original": raw}
            rules.append(rule)
            premise_map.append({"id": f"P{idx}", "kind": "rule", "original": raw, "rule": rule})
        else:
            facts.append({"id": f"P{idx}", "fact": raw})
            premise_map.append({"id": f"P{idx}", "kind": "fact", "original": raw})
    q = stringify(sample.get("question"))
    return {
        "task_type": "logic",
        "question_type": "mcq" if re.search(r"\b[A-D]\.\s+", q) else "yes_no_or_open",
        "premise_map": premise_map,
        "facts": facts,
        "rules": rules,
        "query": q,
        "answer_metadata": {"answer": sample.get("answer", "")},
    }


def weak_physics_parse(sample: dict[str, Any]) -> dict[str, Any]:
    question = stringify(sample.get("question"))
    quantities_out: list[dict[str, Any]] = []
    formula_name = None
    formula_expr = None
    target = None
    unit_expected = sample.get("unit")

    if extract_quantities is not None:
        quantities = extract_quantities(question)
        for q in quantities.values():
            quantities_out.append({"symbol": q.symbol, "value_si": q.value, "unit": q.unit, "raw": q.raw})
        if choose_formula is not None:
            values = {k: v.value for k, v in quantities.items()}
            formula = choose_formula(question, values)
            if formula is not None:
                formula_name = formula.name
                formula_expr = formula.expression
                target = formula.target
                unit_expected = unit_expected or formula.unit

    return {
        "task_type": "physics",
        "question": question,
        "target": target,
        "quantities": quantities_out,
        "candidate_formulas": [formula_expr] if formula_expr else [],
        "formula_name": formula_name,
        "unit_expected": unit_expected,
        "answer_metadata": {"answer": sample.get("answer", ""), "unit": sample.get("unit")},
    }


def make_assistant_parse(sample: dict[str, Any]) -> str:
    try:
        req = PredictRequest.model_validate(sample)
        task_type = route_task(req)
    except Exception:
        task_type = str(sample.get("type") or "logic")
    payload = weak_physics_parse(sample) if task_type == "physics" else weak_logic_parse(sample)
    return json.dumps(payload, ensure_ascii=False)


def _infer_task_type(sample: dict[str, Any]) -> str:
    try:
        req = PredictRequest.model_validate(sample)
        return route_task(req)
    except Exception:
        return str(sample.get("type") or "logic")


def _premises(sample: dict[str, Any]) -> list[str]:
    return [stringify(p).strip() for p in (sample.get("premises") or sample.get("premises-NL") or sample.get("premises_nl") or []) if stringify(p).strip()]


def _option_text(question: str, answer: str) -> str:
    ans = stringify(answer).strip().upper().replace("OPTION ", "")[:1]
    if ans not in {"A", "B", "C", "D"}:
        return ""
    m = re.search(rf"\b{ans}\.\s*([^\n]+)", question)
    return m.group(1).strip() if m else ""


def _facts_and_rules(premises: list[str]) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    facts: list[tuple[int, str]] = []
    rules: list[tuple[int, str, str]] = []
    for i, p in enumerate(premises, 1):
        text = p.strip().rstrip(".")
        m = re.match(r"^if\s+(.+?)\s*(?:,\s*)?(?:then\s+)?(.+)$", text, flags=re.I)
        if m:
            rules.append((i, m.group(1).strip(), m.group(2).strip()))
        else:
            facts.append((i, text))
    return facts, rules


def make_logic_style_explanation(sample: dict[str, Any]) -> str:
    answer = stringify(sample.get("answer", "Uncertain")) or "Uncertain"
    q = stringify(sample.get("question"))
    premises = _premises(sample)
    facts, rules = _facts_and_rules(premises)
    opt = _option_text(q, answer)

    pieces: list[str] = []
    if facts:
        first_facts = ", ".join(f"premise {i}" for i, _ in facts[:3])
        pieces.append(f"{first_facts.capitalize()} provide the starting facts.")
    if rules:
        chain_bits = []
        for i, ant, cons in rules[:4]:
            chain_bits.append(f"premise {i} states that if {ant}, then {cons}")
        pieces.append(" Then, " + "; ".join(chain_bits) + ".")
    if opt:
        pieces.append(f"These derived facts support option {answer}: {opt}.")
    elif answer.lower() in {"yes", "true", "entailed"}:
        pieces.append("Therefore, the queried conclusion is supported by the premises, so the answer is Yes.")
    elif answer.lower() in {"no", "false", "contradiction", "contradicted"}:
        pieces.append("Therefore, the queried conclusion is contradicted or ruled out by the premises, so the answer is No.")
    elif answer.lower() in {"unknown", "uncertain", "undetermined", "not enough information"}:
        pieces.append("The available premise chain does not prove the query or its negation, so the answer is Uncertain.")
    else:
        pieces.append(f"Therefore, the final answer is {answer}.")

    return " ".join(pieces).strip()


def make_physics_style_explanation(sample: dict[str, Any]) -> str:
    answer = stringify(sample.get("answer", ""))
    unit = sample.get("unit")
    q = stringify(sample.get("question"))
    cot_items = as_list(sample.get("cot"))
    if cot_items:
        cleaned = [stringify(c).strip().rstrip(".") for c in cot_items if stringify(c).strip()]
        body = " ".join(cleaned[:5])
    else:
        parse = weak_physics_parse(sample)
        formulas = parse.get("candidate_formulas") or []
        quantities = parse.get("quantities") or []
        parts = []
        if quantities:
            parts.append("Identify the given quantities: " + ", ".join(qt.get("raw", "") for qt in quantities[:6] if qt.get("raw")) + ".")
        if formulas:
            parts.append(f"Use the relevant formula {formulas[0]}.")
        body = " ".join(parts) or "Identify the given quantities, select the matching physics formula, substitute the values, and compute the result."
    unit_part = f" {unit}" if unit else ""
    return f"{body} Thus, the final answer is {answer}{unit_part}."


def make_exact_style_explanation(sample: dict[str, Any]) -> str:
    typ = _infer_task_type(sample)
    if typ == "physics":
        return make_physics_style_explanation(sample)
    return make_logic_style_explanation(sample)


def make_rewrite_user_content(sample: dict[str, Any]) -> str:
    answer = sample.get("answer", "")
    unit = sample.get("unit")
    cot = sample.get("cot", [])
    draft = sample.get("explanation", "")
    premises = _premises(sample)
    return (
        "Rewrite the following solver output into an EXACT-style educational explanation.\n"
        "Requirements:\n"
        "- Keep answer and unit unchanged.\n"
        "- Mention only listed premises/formulas/trace steps.\n"
        "- Prefer concise premise-numbered reasoning, not generic solver narration.\n\n"
        f"Question: {sample.get('question', '')}\n"
        f"Fixed answer: {answer}\n"
        f"Fixed unit: {unit}\n"
        f"Premises/formulas: {json.dumps(premises, ensure_ascii=False)}\n"
        f"Solver trace / CoT: {json.dumps(cot, ensure_ascii=False)}\n"
        f"Draft explanation: {draft}\n"
        "Return JSON with answer, unit, and explanation."
    )


def make_assistant_rewrite(sample: dict[str, Any]) -> str:
    gold_expl = sample.get("explanation", "")
    explanation = stringify(gold_expl).strip()
    if is_weak_explanation(explanation):
        explanation = make_exact_style_explanation(sample)
    payload = {
        "answer": sample.get("answer", ""),
        "unit": sample.get("unit"),
        "explanation": explanation,
    }
    return json.dumps(payload, ensure_ascii=False)


def expand_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        if "questions" in row and isinstance(row.get("questions"), list):
            for idx, q in enumerate(row["questions"]):
                sample = {
                    "id": f"{row.get('id', 'logic')}-{idx}",
                    "type": "logic",
                    "premises-NL": row.get("premises-NL", row.get("premises_nl", [])),
                    "question": q,
                    "answer": (row.get("answers") or [""])[idx] if idx < len(row.get("answers") or []) else "",
                    "explanation": (row.get("explanation") or [""])[idx]
                    if isinstance(row.get("explanation"), list) and idx < len(row.get("explanation"))
                    else row.get("explanation", ""),
                }
                samples.append(sample)
        else:
            samples.append(row)
    return samples


def stable_float_key(sample: dict[str, Any]) -> float:
    key = stringify(sample.get("id")) + "\n" + stringify(sample.get("question"))
    h = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return int(h, 16) / float(16 ** 12)


def choose_mode(sample: dict[str, Any], mode: str, parser_ratio: float, rewrite_ratio: float) -> str:
    if mode != "mixed":
        return mode
    rnd = stable_float_key(sample)
    if rnd < parser_ratio:
        return "parser"
    if rnd < parser_ratio + rewrite_ratio:
        return "rewrite"
    return "answer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["answer", "parser", "rewrite", "mixed"], default="mixed")
    parser.add_argument("--parser_ratio", type=float, default=0.10)
    parser.add_argument("--rewrite_ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = expand_records(read_jsonl(args.input))
    out = []
    counts = {"answer": 0, "parser": 0, "rewrite": 0}
    for sample in rows:
        if not sample.get("question") or sample.get("answer", "") in (None, ""):
            continue
        record_mode = choose_mode(sample, args.mode, args.parser_ratio, args.rewrite_ratio)
        if record_mode == "parser":
            messages = [
                {"role": "system", "content": SYSTEM_PARSE},
                {"role": "user", "content": make_user_content(sample)},
                {"role": "assistant", "content": make_assistant_parse(sample)},
            ]
        elif record_mode == "rewrite":
            messages = [
                {"role": "system", "content": SYSTEM_REWRITE},
                {"role": "user", "content": make_rewrite_user_content(sample)},
                {"role": "assistant", "content": make_assistant_rewrite(sample)},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_ANSWER},
                {"role": "user", "content": make_user_content(sample)},
                {"role": "assistant", "content": make_assistant_answer(sample)},
            ]
        counts[record_mode] += 1
        out.append({"messages": messages})

    write_jsonl(args.output, out)
    print(json.dumps({"output": args.output, "records": len(out), "mode_counts": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
