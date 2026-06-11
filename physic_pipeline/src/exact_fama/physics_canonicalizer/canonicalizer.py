from __future__ import annotations
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from ..llm.qwen_client import QwenClient
from ..schemas import PredictRequest, SolverResult
from .lexicon import CANONICAL_FAMILIES, OUTPUT_UNIT_ALIASES
_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:e|E)\s*[-+]?\d+)?"
_UNIT = (
    r"microfarads?|microfarad|μF|µF|uF|mF|nF|pF|F|"
    r"microcoulombs?|microcoulomb|μC|µC|uC|mC|nC|pC|C|"
    r"millihenries|millihenry|microhenries|microhenry|mH|μH|µH|uH|H|"
    r"kΩ|kω|Ω|ω|kohm|ohms?|"
    r"kHz|Hz|"
    r"mA|A|"
    r"kV|mV|V|volts?|"
    r"mJ|microjoules?|microjoule|μJ|µJ|uJ|J|"
    r"cm\^2|cm²|cm2|mm\^2|mm²|mm2|m\^2|m²|m2|"
    r"km|cm|mm|m|"
    r"N/C|V/m|J/m\^3|J/m³|J/m3|Wb|mWb|μWb|µWb|T|mT|μT|µT|W"
)
def _normalize_text(text: str) -> str:
    s = str(text or "")
    s = s.translate(_SUPERSCRIPT_MAP)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("µ", "μ").replace("Ω", "Ω").replace("π", "pi")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def _lower(text: str) -> str:
    return _normalize_text(text).lower()
def _unit_clean(unit: str | None) -> str:
    u = _normalize_text(unit or "").strip()
    low = u.lower().replace("µ", "μ")
    return OUTPUT_UNIT_ALIASES.get(low, u)
def _value_unit(value: str, unit: str) -> str:
    return f"{_normalize_text(value).replace(' ', '')} {_unit_clean(unit)}".strip()
def _find_output_unit(text: str) -> str | None:
    t = _normalize_text(text)
    patterns = [
        rf"(?:answer|give\s+the\s+answer|calculate|compute|find|determine)\s+(?:the\s+)?(?:result\s+)?(?:in|to|using|use)\s+(?P<u>{_UNIT})\b",
        rf"(?:in|using|use)\s+(?P<u>{_UNIT})\s*\??\s*$",
        rf"what\s+is\s+[^?]*\s+in\s+(?P<u>{_UNIT})\b",
        rf"\[(?:expected_unit|unit):\s*(?P<u>[^\]]+)\]",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            return _unit_clean(m.group("u"))
    return None
def _contains_formula_error(result: SolverResult) -> bool:
    text = " ".join(str(w) for w in (result.warnings or []))
    return "PHYSICS_FORMULA_ERROR" in text or "PHYSICS_PARSE_ERROR" in text
def _is_uncertain(result: SolverResult) -> bool:
    return str(result.answer or "").strip().lower() in {"", "uncertain", "unknown"}
def _has_fatal_warning(result: SolverResult) -> bool:
    text = " ".join(str(w) for w in (result.warnings or []))
    return "OUTPUT_SCHEMA_ERROR" in text or "division by zero" in text.lower()
def _safe_float(x: str) -> float | None:
    try:
        return float(str(x).replace(" ", ""))
    except Exception:
        return None
def _numeric_tokens(text: str) -> list[str]:
    s = _normalize_text(text)
    out: list[str] = []
    for m in re.finditer(rf"(?<![A-Za-z_])({_NUM})(?![A-Za-z_])", s):
        out.append(re.sub(r"\s+", "", m.group(1)))
    return out
def _num_equiv(a: str, b: str) -> bool:
    fa, fb = _safe_float(a), _safe_float(b)
    if fa is None or fb is None:
        return a == b
    return math.isclose(fa, fb, rel_tol=1e-10, abs_tol=1e-12)
@dataclass
class CanonicalizationResult:
    accepted: bool = False
    canonical_question: str = ""
    family: str = ""
    method: str = "none"
    target_unit: str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
class PhysicsCanonicalizer:
    def __init__(
        self,
        llm: QwenClient | None = None,
        *,
        enabled: bool = True,
        use_llm: bool = False,
        policy: str = "fallback_only",
        min_candidate_confidence: float = 0.45,
        max_llm_retries: int = 1,
        prompt_path: str = "",
    ) -> None:
        self.llm = llm
        self.enabled = bool(enabled)
        self.use_llm = bool(use_llm)
        self.policy = str(policy or "fallback_only").strip().lower()
        self.min_candidate_confidence = float(min_candidate_confidence)
        self.max_llm_retries = max(0, int(max_llm_retries))
        self.prompt_path = str(prompt_path or "").strip()
        self._loaded_system_prompt: str | None = None
    def should_try(self, request: PredictRequest, baseline: SolverResult) -> bool:
        if not self.enabled:
            return False
        if (request.type or "physics") != "physics":
            return False
        if self.policy == "off":
            return False
        if self.policy == "always":
            return True
        return _is_uncertain(baseline) or _contains_formula_error(baseline) or float(baseline.confidence or 0.0) < 0.35
    def canonicalize(
        self,
        question: str,
        *,
        baseline: SolverResult | None = None,
        force_llm: bool = False,
        prior_warnings: list[str] | None = None,
    ) -> CanonicalizationResult:
        if not self.enabled:
            return CanonicalizationResult(warnings=["PHYSICS_CANONICALIZER_SKIPPED: disabled"])
        if force_llm:
            if self.use_llm and self.llm is not None and self.llm.backend != "none":
                return self._llm_based(question, baseline=baseline, rule_warnings=prior_warnings or [])
            return CanonicalizationResult(
                warnings=list(prior_warnings or []) + [
                    "PHYSICS_CANONICALIZER_LLM_SKIPPED: LLM paraphrase is disabled or backend is none."
                ]
            )
        rule = self._rule_based(question)
        if rule.accepted:
            return rule
        if self.use_llm and self.llm is not None and self.llm.backend != "none":
            return self._llm_based(question, baseline=baseline, rule_warnings=rule.warnings)
        return rule
    def make_request(self, original: PredictRequest, canonical: CanonicalizationResult) -> PredictRequest:
        data = original.model_dump(by_alias=True)
        data["type"] = "physics"
        data["question"] = canonical.canonical_question
        if canonical.target_unit:
            data["unit"] = canonical.target_unit
        meta = dict(data.get("metadata") or {})
        meta["physics_canonicalizer"] = {
            "method": canonical.method,
            "family": canonical.family,
            "original_question": original.question,
            "canonical_question": canonical.canonical_question,
            "target_unit": canonical.target_unit,
        }
        data["metadata"] = meta
        return PredictRequest.model_validate(data)
    def choose_result(
        self,
        *,
        baseline: SolverResult,
        candidate: SolverResult | None,
        canonical: CanonicalizationResult,
    ) -> SolverResult:
        debug_payload = {
            "enabled": self.enabled,
            "attempted": bool(canonical.accepted),
            "method": canonical.method,
            "family": canonical.family,
            "canonical_question": canonical.canonical_question,
            "target_unit": canonical.target_unit,
            "warnings": canonical.warnings,
            "baseline_answer": baseline.answer,
            "candidate_answer": candidate.answer if candidate else None,
            "candidate_confidence": candidate.confidence if candidate else None,
        }
        if not canonical.accepted or candidate is None:
            baseline.debug = dict(baseline.debug or {})
            baseline.debug["physics_canonicalizer"] = debug_payload | {"decision": "not_used"}
            return baseline
        if candidate.answer == "Uncertain" or _has_fatal_warning(candidate):
            baseline.debug = dict(baseline.debug or {})
            baseline.debug["physics_canonicalizer"] = debug_payload | {"decision": "rejected_candidate_uncertain_or_fatal"}
            return baseline
        if float(candidate.confidence or 0.0) < self.min_candidate_confidence:
            baseline.debug = dict(baseline.debug or {})
            baseline.debug["physics_canonicalizer"] = debug_payload | {"decision": "rejected_low_candidate_confidence"}
            return baseline
        if self.policy != "always" and not (_is_uncertain(baseline) or _contains_formula_error(baseline) or float(baseline.confidence or 0.0) < 0.35):
            baseline.debug = dict(baseline.debug or {})
            baseline.debug["physics_canonicalizer"] = debug_payload | {"decision": "rejected_baseline_not_failure"}
            return baseline
        candidate.debug = dict(candidate.debug or {})
        candidate.debug["physics_canonicalizer"] = debug_payload | {"decision": "accepted_canonical_solver_retry"}
        candidate.warnings = list(candidate.warnings or [])
        return candidate
    def _rule_based(self, question: str) -> CanonicalizationResult:
        t = _normalize_text(question)
        q = t.lower()
        rules = [
            self._rule_parallel_plate_field,
            self._rule_lc_design,
            self._rule_lc_frequency_or_resonance_check,
            self._rule_capacitor_voltage,
            self._rule_capacitor_charge,
            self._rule_capacitor_energy,
            self._rule_series_capacitor_voltage,
            self._rule_point_charge_field,
            self._rule_rlc_operating_point,
            self._rule_equivalent_resistance,
            self._rule_solenoid,
            self._rule_induction,
            self._rule_battery_terminal_voltage,
            self._rule_ohm_power,
            self._rule_measurement,
        ]
        for rule in rules:
            try:
                out = rule(t, q)
            except Exception as exc:
                out = CanonicalizationResult(warnings=[f"PHYSICS_CANONICALIZER_RULE_ERROR: {rule.__name__}: {type(exc).__name__}: {exc}"])
            if out.accepted:
                return out
        return CanonicalizationResult(warnings=["PHYSICS_CANONICALIZER_NO_RULE_MATCH"])
    @staticmethod
    def _accepted(question: str, family: str, *, unit: str | None = None, evidence: dict[str, Any] | None = None, confidence: float = 0.9) -> CanonicalizationResult:
        return CanonicalizationResult(
            accepted=True,
            canonical_question=question,
            family=family,
            method="rule",
            target_unit=unit,
            confidence=confidence,
            evidence=evidence or {},
        )
    def _rule_parallel_plate_field(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"parallel\s+plates?\s+are\s+separated\s+by\s+(?P<d>{_NUM})\s*(?P<du>km|cm|mm|m)\s+and\s+connected\s+to\s+(?P<u>{_NUM})\s*(?P<uu>kV|mV|V|volts?)",
            rf"plates?\s+are\s+(?P<d>{_NUM})\s*(?P<du>km|cm|mm|m)\s+apart\s+.*?(?:voltage|potential\s+difference|connected\s+to)\s+(?P<u>{_NUM})\s*(?P<uu>kV|mV|V|volts?)",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m and ("what is e" in q or "electric field" in q or "field" in q):
                u = _value_unit(m.group("u"), m.group("uu"))
                d = _value_unit(m.group("d"), m.group("du"))
                cq = f"The potential difference between two parallel plates is {u} and their separation is {d}. Find the uniform electric field magnitude between them."
                return self._accepted(cq, "parallel_plate_electric_field", unit="V/m", evidence={"U": u, "d": d})
        return CanonicalizationResult()
    def _rule_lc_design(self, t: str, q: str) -> CanonicalizationResult:
        pats_c = [
            rf"circuit\s+needs\s+f\s*0?\s*=\s*(?P<f>{_NUM})\s*(?P<fu>kHz|Hz)\s+with\s+inductance\s+(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\b.*?(?:capacitor\s+value|capacitance).*?(?:in|use)\s+(?P<out>microfarads?|μF|uF|nF|pF|F)",
            rf"(?:an\s+)?LC\s+circuit\s+must\s+resonate\s+at\s+(?P<f>{_NUM})\s*(?P<fu>kHz|Hz)\s+and\s+uses\s+an\s+inductor\s+of\s+(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\b.*?capacitance\s+is\s+required\s+in\s+(?P<out>microfarads?|μF|uF|nF|pF|F)",
            rf"(?:find|calculate|determine).*?capacitance.*?resonat(?:e|es|ing).*?(?P<f>{_NUM})\s*(?P<fu>kHz|Hz).*?induct(?:ance|or).*?(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)",
        ]
        for pat in pats_c:
            m = re.search(pat, t, flags=re.I)
            if m:
                f = _value_unit(m.group("f"), m.group("fu"))
                L = _value_unit(m.group("L"), m.group("Lu"))
                unit = _unit_clean(m.groupdict().get("out") or _find_output_unit(t) or "F")
                cq = f"Given an inductor with inductance L = {L}, the LC circuit needs to resonate at frequency f = {f}. Calculate the required capacitance C in {unit}."
                return self._accepted(cq, "lc_required_capacitance", unit=unit, evidence={"L": L, "f": f})
        pats_l = [
            rf"calculate\s+the\s+inductance\s+needed\s+for\s+an\s+LC\s+oscillator\s+with\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+capacitance\s+at\s+(?P<f>{_NUM})\s*(?P<fu>kHz|Hz)",
            rf"capacitor\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?resonate\s+at\s+f\s*=\s*(?P<f>{_NUM})\s*(?P<fu>kHz|Hz).*?(?:required\s+inductance|what\s+is\s+L|calculate\s+L)",
        ]
        for pat in pats_l:
            m = re.search(pat, t, flags=re.I)
            if m:
                C = _value_unit(m.group("C"), m.group("Cu"))
                f = _value_unit(m.group("f"), m.group("fu"))
                unit = _find_output_unit(t) or "mH"
                cq = f"Given a capacitor with capacitance C = {C}, the LC circuit needs to resonate at frequency f = {f}. Calculate the required inductance L in {unit}."
                return self._accepted(cq, "lc_required_inductance", unit=unit, evidence={"C": C, "f": f})
        return CanonicalizationResult()
    def _rule_lc_frequency_or_resonance_check(self, t: str, q: str) -> CanonicalizationResult:
        m = re.search(
            rf"lossless\s+LC\s+circuit,?\s*L\s*=\s*(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\s+and\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?(?:determine|find|calculate)\s+f\b",
            t,
            flags=re.I,
        )
        if m:
            L = _value_unit(m.group("L"), m.group("Lu"))
            C = _value_unit(m.group("C"), m.group("Cu"))
            cq = f"An LC circuit has inductance L = {L} and capacitance C = {C}. Calculate the resonant frequency f in Hz."
            return self._accepted(cq, "lc_required_capacitance", unit="Hz", evidence={"L": L, "C": C}, confidence=0.82)
        pats = [
            rf"given\s+L\s*=\s*(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\s+and\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?(?:check\s+whether|does|is).*?f\s*=\s*(?P<f>{_NUM})\s*(?P<fu>kHz|Hz).*?resonant",
            rf"series\s+RLC\s+circuit\s+has\s+R\s*=\s*(?P<R>{_NUM})\s*(?P<Ru>kΩ|Ω|ω|ohms?),\s*L\s*=\s*(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\s+and\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?source\s+frequency\s+is\s+(?P<f>{_NUM})\s*(?P<fu>kHz|Hz).*?resonance\s+occur",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                L = _value_unit(m.group("L"), m.group("Lu"))
                C = _value_unit(m.group("C"), m.group("Cu"))
                f = _value_unit(m.group("f"), m.group("fu"))
                cq = f"Given an RLC circuit with inductance L = {L}, capacitance C = {C}, and source frequency f = {f}. Does resonance occur?"
                return self._accepted(cq, "lc_required_capacitance", unit=None, evidence={"L": L, "C": C, "f": f}, confidence=0.82)
        return CanonicalizationResult()
    def _rule_capacitor_voltage(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"determine\s+the\s+potential\s+difference\s+of\s+a\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+capacitor\s+if\s+its\s+charge\s+is\s+(?P<Q>{_NUM})\s*(?P<Qu>mC|μC|uC|nC|C)",
            rf"charge\s+on\s+a\s+capacitor\s+is\s+(?P<Q>{_NUM})\s*(?P<Qu>mC|μC|uC|nC|C).*?capacitance\s+is\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?(?:calculate|find|determine)\s+U",
            rf"capacitor\s+of\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?(?:carries|stores|has\s+charge)\s+(?P<Q>{_NUM})\s*(?P<Qu>mC|μC|uC|nC|C).*?(?:find|determine)\s+U",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                C = _value_unit(m.group("C"), m.group("Cu"))
                Q = _value_unit(m.group("Q"), m.group("Qu"))
                unit = _find_output_unit(t) or "V"
                cq = f"A capacitor has capacitance C = {C} and charge Q = {Q}. Calculate the voltage U across the capacitor in {unit}."
                return self._accepted(cq, "capacitor_voltage_q_over_c", unit=unit, evidence={"C": C, "Q": Q})
        return CanonicalizationResult()
    def _rule_capacitor_charge(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"capacitance\s+is\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+and\s+the\s+applied\s+voltage\s+is\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?).*?(?:compute|calculate|find).*?charge",
            rf"find\s+Q\s+for\s+a\s+capacitor\s+with\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+and\s+U\s*=\s*(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?)",
            rf"(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+capacitor\s+(?:is\s+)?(?:maintained\s+at|connected\s+to|has)\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?).*?(?:charge|Q)",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                C = _value_unit(m.group("C"), m.group("Cu"))
                U = _value_unit(m.group("U"), m.group("Uu"))
                unit = _find_output_unit(t) or "C"
                cq = f"A capacitor has capacitance C = {C} and voltage U = {U}. Calculate the charge Q stored on the capacitor in {unit}."
                return self._accepted(cq, "capacitor_charge_cv", unit=unit, evidence={"C": C, "U": U})
        return CanonicalizationResult()
    def _rule_capacitor_energy(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"for\s+a\s+capacitor\s+with\s+capacitance\s+(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+connected\s+to\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?).*?energy",
            rf"(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+capacitor\s+has\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?)\s+across\s+its\s+terminals.*?(?:energy|W)",
            rf"capacitance\s+C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F).*?(?:voltage|U)\s*=\s*(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?).*?(?:energy|W)",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                C = _value_unit(m.group("C"), m.group("Cu"))
                U = _value_unit(m.group("U"), m.group("Uu"))
                unit = _find_output_unit(t) or "J"
                cq = f"A capacitor has capacitance C = {C} and voltage U = {U}. Calculate the energy stored in the capacitor in {unit}."
                return self._accepted(cq, "capacitor_energy_cv", unit=unit, evidence={"C": C, "U": U})
        return CanonicalizationResult()
    def _rule_series_capacitor_voltage(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"in\s+a\s+series\s+capacitor\s+pair,?\s*C1\s+is\s+(?P<C1>{_NUM})\s*(?P<C1u>microfarads?|μF|uF|nF|pF|F)\s+and\s+C2\s+is\s+(?P<C2>{_NUM})\s*(?P<C2u>microfarads?|μF|uF|nF|pF|F).*?total\s+voltage\s+is\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?).*?what\s+is\s+U\s*(?P<idx>[12])",
            rf"capacitors\s+of\s+(?P<C1>{_NUM})\s*(?P<C1u>microfarads?|μF|uF|nF|pF|F)\s+and\s+(?P<C2>{_NUM})\s*(?P<C2u>microfarads?|μF|uF|nF|pF|F)\s+are\s+in\s+series\s+on\s+a\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?)\s+supply.*?capacitor\s+C\s*(?P<idx>[12])",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                C1 = _value_unit(m.group("C1"), m.group("C1u"))
                C2 = _value_unit(m.group("C2"), m.group("C2u"))
                U = _value_unit(m.group("U"), m.group("Uu"))
                idx = m.group("idx")
                cq = f"Two capacitors C1 = {C1} and C2 = {C2} are connected in series across a total voltage UAB = {U}. Calculate the voltage across capacitor C{idx}."
                return self._accepted(cq, "series_capacitor_voltage_division", unit="V", evidence={"C1": C1, "C2": C2, "UAB": U, "target": f"U{idx}"})
        return CanonicalizationResult()
    def _rule_point_charge_field(self, t: str, q: str) -> CanonicalizationResult:
        pats = [
            rf"what\s+electric\s+field\s+is\s+produced\s+(?P<r>{_NUM})\s*(?P<ru>km|cm|mm|m)\s+from\s+a\s+(?P<Q>{_NUM})\s*(?P<Qu>mC|μC|uC|nC|pC|C)\s+charge\s+when\s+the\s+dielectric\s+constant\s+is\s+(?P<eps>{_NUM})",
            rf"electric\s+field.*?(?P<r>{_NUM})\s*(?P<ru>km|cm|mm|m)\s+from\s+(?:a\s+)?(?:point\s+)?charge\s+(?P<Q>{_NUM})\s*(?P<Qu>mC|μC|uC|nC|pC|C).*?(?:dielectric\s+constant|relative\s+permittivity|εr)\s*(?:is|=)\s*(?P<eps>{_NUM})",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                r = _value_unit(m.group("r"), m.group("ru"))
                Q = _value_unit(m.group("Q"), m.group("Qu"))
                eps = m.group("eps").strip()
                cq = f"A point charge Q = {Q} is in a medium with dielectric constant εr = {eps}. Calculate the electric field magnitude at distance r = {r}."
                return self._accepted(cq, "point_charge_electric_field", unit="N/C", evidence={"Q": Q, "r": r, "eps": eps})
        return CanonicalizationResult()
    def _rule_rlc_operating_point(self, t: str, q: str) -> CanonicalizationResult:
        pat = rf"AC\s+source\s+of\s+(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?)\s+RMS\s+and\s+frequency\s+(?P<f>{_NUM})\s*(?P<fu>kHz|Hz)\s+feeds\s+R\s*=\s*(?P<R>{_NUM})\s*(?P<Ru>kΩ|Ω|ω|ohms?)\s*,\s*L\s*=\s*(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\s*,\s*C\s*=\s*(?P<C>{_NUM})\s*(?P<Cu>microfarads?|μF|uF|nF|pF|F)\s+in\s+series"
        m = re.search(pat, t, flags=re.I)
        if m:
            U = _value_unit(m.group("U"), m.group("Uu"))
            f = _value_unit(m.group("f"), m.group("fu"))
            R = _value_unit(m.group("R"), m.group("Ru"))
            L = _value_unit(m.group("L"), m.group("Lu"))
            C = _value_unit(m.group("C"), m.group("Cu"))
            if "power factor" in q or "cos" in q:
                target = "power factor"
                unit = None
            elif "impedance" in q or re.search(r"\bZ\b", t):
                target = "impedance Z"
                unit = "Ω"
            elif "current" in q:
                target = "RMS current I"
                unit = "A"
            elif "voltage across the resistor" in q:
                target = "voltage across the resistor"
                unit = "V"
            elif "voltage across the inductor" in q:
                target = "voltage across the inductor"
                unit = "V"
            elif "voltage across the capacitor" in q:
                target = "voltage across the capacitor"
                unit = "V"
            else:
                return CanonicalizationResult()
            cq = f"A series RLC circuit has R = {R}, L = {L}, C = {C}. It is supplied by RMS voltage U = {U} at frequency f = {f}. Calculate the {target}."
            return self._accepted(cq, "series_rlc_operating_point", unit=unit, evidence={"R": R, "L": L, "C": C, "U": U, "f": f, "target": target})
        return CanonicalizationResult()
    def _rule_equivalent_resistance(self, t: str, q: str) -> CanonicalizationResult:
        if not any(k in q for k in ["resistors in parallel", "parallel combination", "series combination", "resistors in series"]):
            return CanonicalizationResult()
        vals = re.findall(rf"\bR\s*_?\s*(?P<idx>\d+)\s*=\s*(?P<v>{_NUM})\s*(?P<u>kΩ|Ω|ω|kohm|ohms?)", t, flags=re.I)
        if len(vals) < 2:
            return CanonicalizationResult()
        parts = [f"R{idx} = {_value_unit(v, u)}" for idx, v, u in vals]
        if "parallel" in q:
            cq = "Find the equivalent resistance of resistors connected in parallel: " + ", ".join(parts) + "."
            return self._accepted(cq, "equivalent_resistance_parallel", unit="Ω", evidence={"resistors": parts})
        if "series" in q:
            cq = "Find the equivalent resistance of resistors connected in series: " + ", ".join(parts) + "."
            return self._accepted(cq, "equivalent_resistance_series", unit="Ω", evidence={"resistors": parts})
        return CanonicalizationResult()
    def _rule_solenoid(self, t: str, q: str) -> CanonicalizationResult:
        if "solenoid" not in q:
            return CanonicalizationResult()
        N = None
        for pat in [rf"N\s*=\s*(?P<N>{_NUM})", rf"(?P<N>{_NUM})\s+turns"]:
            m = re.search(pat, t, flags=re.I)
            if m:
                N = m.group("N"); break
        lm = re.search(rf"(?:length|l|over)\s*(?:=|is|of)?\s*(?P<l>{_NUM})\s*(?P<lu>km|cm|mm|m)", t, flags=re.I)
        im = re.search(rf"(?:current|I|carrying)\s*(?:=|is|of)?\s*(?P<I>{_NUM})\s*(?P<Iu>mA|A)", t, flags=re.I)
        am = re.search(rf"(?:area|cross-sectional\s+area|S)\s*(?:=|is|of)?\s*(?P<A>{_NUM})\s*(?P<Au>cm\^2|cm²|cm2|mm\^2|mm²|mm2|m\^2|m²|m2)", t, flags=re.I)
        if N and lm and im and ("magnetic field" in q or re.search(r"\bB\b", t)):
            l = _value_unit(lm.group("l"), lm.group("lu"))
            I = _value_unit(im.group("I"), im.group("Iu"))
            cq = f"A long solenoid has N = {N} turns, length l = {l}, and current I = {I}. Calculate the magnetic field inside the solenoid."
            return self._accepted(cq, "solenoid_magnetic_field", unit="T", evidence={"N": N, "l": l, "I": I})
        if N and lm and am and ("inductance" in q or "self-inductance" in q or re.search(r"\bL\b", t)):
            l = _value_unit(lm.group("l"), lm.group("lu"))
            A = _value_unit(am.group("A"), am.group("Au"))
            unit = _find_output_unit(t) or "H"
            cq = f"A solenoid has N = {N} turns, length l = {l}, and cross-sectional area A = {A}. Calculate the self-inductance L in {unit}."
            return self._accepted(cq, "solenoid_self_inductance", unit=unit, evidence={"N": N, "l": l, "A": A})
        return CanonicalizationResult()
    def _rule_induction(self, t: str, q: str) -> CanonicalizationResult:
        m = re.search(
            rf"coil\s+produces\s+(?P<eps>{_NUM})\s*(?P<epsu>kV|mV|V|volts?)\s+of\s+self-induced\s+EMF\s+for\s+a\s+current\s+change\s+of\s+(?P<di>{_NUM})\s*(?P<diu>mA|A)\s+in\s+(?P<dt>{_NUM})\s*(?P<dtu>ms|s).*?find\s+L",
            t,
            flags=re.I,
        )
        if m:
            eps = _value_unit(m.group("eps"), m.group("epsu"))
            di = _value_unit(m.group("di"), m.group("diu"))
            dt = _value_unit(m.group("dt"), m.group("dtu"))
            unit = _find_output_unit(t) or "H"
            cq = f"A coil has self-induced emf ε = {eps} when the current changes by ΔI = {di} over Δt = {dt}. Calculate the inductance L in {unit}."
            return self._accepted(cq, "solenoid_self_inductance", unit=unit, evidence={"epsilon": eps, "delta_I": di, "delta_t": dt}, confidence=0.78)
        m = re.search(
            rf"inductance\s+L\s*=\s*(?P<L>{_NUM})\s*(?P<Lu>mH|μH|uH|H)\s+has\s+current\s+changing\s+by\s+(?P<di>{_NUM})\s*(?P<diu>mA|A)\s+over\s+(?P<dt>{_NUM})\s*(?P<dtu>ms|s).*?(?:calculate|find)\s*(?:ε|epsilon|emf)",
            t,
            flags=re.I,
        )
        if m:
            L = _value_unit(m.group("L"), m.group("Lu"))
            di = _value_unit(m.group("di"), m.group("diu"))
            dt = _value_unit(m.group("dt"), m.group("dtu"))
            cq = f"An inductor has inductance L = {L}. The current changes by ΔI = {di} over Δt = {dt}. Calculate the induced emf ε."
            return self._accepted(cq, "solenoid_self_inductance", unit="V", evidence={"L": L, "delta_I": di, "delta_t": dt}, confidence=0.78)
        return CanonicalizationResult()
    def _rule_battery_terminal_voltage(self, t: str, q: str) -> CanonicalizationResult:
        if not any(k in q for k in ["terminal voltage", "battery", "internal resistance", "emf"]):
            return CanonicalizationResult()
        em = re.search(rf"(?:emf|electromotive\s+force|E)\s*(?:=|is|of)?\s*(?P<E>{_NUM})\s*(?P<Eu>kV|mV|V|volts?)", t, flags=re.I)
        im = re.search(rf"(?:current|I)\s*(?:=|is|of)?\s*(?P<I>{_NUM})\s*(?P<Iu>mA|A)", t, flags=re.I)
        rm = re.search(rf"(?:internal\s+resistance|r)\s*(?:=|is|of)?\s*(?P<r>{_NUM})\s*(?P<ru>kΩ|Ω|ω|ohms?)", t, flags=re.I)
        if em and im and rm:
            E = _value_unit(em.group("E"), em.group("Eu"))
            I = _value_unit(im.group("I"), im.group("Iu"))
            r = _value_unit(rm.group("r"), rm.group("ru"))
            cq = f"A battery has emf E = {E}, internal resistance r = {r}, and current I = {I}. Calculate the terminal voltage U."
            return self._accepted(cq, "battery_terminal_voltage", unit="V", evidence={"E": E, "r": r, "I": I})
        return CanonicalizationResult()
    def _rule_ohm_power(self, t: str, q: str) -> CanonicalizationResult:
        vm = re.search(rf"(?:voltage|potential\s+difference|U|V)\s*(?:=|is|of)?\s*(?P<U>{_NUM})\s*(?P<Uu>kV|mV|V|volts?)", t, flags=re.I)
        im = re.search(rf"(?:current|I)\s*(?:=|is|of)?\s*(?P<I>{_NUM})\s*(?P<Iu>mA|A)", t, flags=re.I)
        rm = re.search(rf"(?:resistance|resistor|R)\s*(?:=|is|of)?\s*(?P<R>{_NUM})\s*(?P<Ru>kΩ|Ω|ω|ohms?)", t, flags=re.I)
        if "power" in q and ((vm and im) or (vm and rm) or (im and rm)):
            pieces=[]; ev={}
            if vm: U=_value_unit(vm.group("U"), vm.group("Uu")); pieces.append(f"voltage U = {U}"); ev["U"]=U
            if im: I=_value_unit(im.group("I"), im.group("Iu")); pieces.append(f"current I = {I}"); ev["I"]=I
            if rm: R=_value_unit(rm.group("R"), rm.group("Ru")); pieces.append(f"resistance R = {R}"); ev["R"]=R
            cq = "A circuit has " + ", ".join(pieces) + ". Calculate the electric power P."
            return self._accepted(cq, "ohm_power_current_voltage", unit="W", evidence=ev)
        return CanonicalizationResult()
    def _rule_measurement(self, t: str, q: str) -> CanonicalizationResult:
        m = re.search(
            rf"student\s+reports\s+(?P<x>{_NUM})\s*(?P<u>cm|mm|m|g|kg|mA|A|mV|V|s|°C|C)\s+with\s+absolute\s+uncertainty\s+(?P<dx>{_NUM})\s*(?P<du>cm|mm|m|g|kg|mA|A|mV|V|s|°C|C).*?percentage\s+error",
            t,
            flags=re.I,
        )
        if m:
            x = _value_unit(m.group("x"), m.group("u"))
            dx = _value_unit(m.group("dx"), m.group("du"))
            cq = f"A measured value is {x} with absolute uncertainty Δx = {dx}. Find the percentage relative error."
            return self._accepted(cq, "measurement_uncertainty", unit="%", evidence={"x": x, "dx": dx}, confidence=0.8)
        m = re.search(
            rf"measured\s+value\s+is\s+(?P<x>{_NUM})\s*(?P<u>cm|mm|m|g|kg|mA|A|mV|V|s|°C|C)\s+with\s+uncertainty\s*±\s*(?P<dx>{_NUM})\s*(?P<du>cm|mm|m|g|kg|mA|A|mV|V|s|°C|C).*?upper\s+limit",
            t,
            flags=re.I,
        )
        if m:
            x = _value_unit(m.group("x"), m.group("u"))
            dx = _value_unit(m.group("dx"), m.group("du"))
            cq = f"The measured value is {x} ± {dx}. Find the maximum possible value."
            return self._accepted(cq, "measurement_uncertainty", unit=_unit_clean(m.group("u")), evidence={"x": x, "dx": dx}, confidence=0.8)
        if "±" in t and ("relative" in q or "percentage" in q or "maximum" in q or "upper limit" in q):
            unit = _find_output_unit(t) or ("%" if "percent" in q or "percentage" in q else None)
            return self._accepted(t, "measurement_uncertainty", unit=unit, evidence={}, confidence=0.65)
        return CanonicalizationResult()
    def _llm_based(self, question: str, *, baseline: SolverResult | None, rule_warnings: list[str] | None = None) -> CanonicalizationResult:
        schema = self._llm_schema()
        base_messages = [
            {"role": "system", "content": self._llm_system_prompt()},
            {"role": "user", "content": self._llm_user_prompt(question, baseline=baseline)},
        ]
        max_schema_repairs = max(1, self.max_llm_retries)
        messages = list(base_messages)
        last_result: CanonicalizationResult | None = None
        for attempt in range(max_schema_repairs + 1):
            try:
                raw = self.llm.generate_json(messages, schema=schema, max_retries=0)                            
            except Exception as exc:
                return CanonicalizationResult(
                    warnings=list(rule_warnings or []) + [f"PHYSICS_CANONICALIZER_LLM_ERROR: {type(exc).__name__}: {exc}"],
                    raw={"exception": str(exc), "attempt": attempt},
                )
            result = self._validate_llm_output(question, raw, rule_warnings=rule_warnings)
            if result.accepted:
                return result
            last_result = result
            if attempt < max_schema_repairs:
                messages = list(base_messages) + [{
                    "role": "user",
                    "content": json.dumps({
                        "repair_required": True,
                        "previous_output": raw,
                        "validation_warnings": result.warnings,
                        "required_top_level_keys": [
                            "family", "canonical_question", "target", "target_unit",
                            "quantities", "relations", "confidence",
                        ],
                        "allowed_families": sorted(CANONICAL_FAMILIES),
                        "important": [
                            "Return exactly one JSON object, no markdown.",
                            "Do not solve; do not include final answer.",
                            "Use family=lc_required_capacitance for LC resonance required capacitance problems.",
                            "Use canonical_question, not question/rewritten_question/canonical.",
                        ],
                    }, ensure_ascii=False, indent=2),
                }]
        return last_result or CanonicalizationResult(
            warnings=list(rule_warnings or []) + ["PHYSICS_CANONICALIZER_LLM_ERROR: no usable LLM output"],
        )
    @staticmethod
    def _raw_get_any(raw: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in raw and raw.get(key) not in (None, ""):
                return raw.get(key)
        return None
    @staticmethod
    def _infer_family_from_text(text: str) -> str:
        q = _lower(text)
        if ("lc" in q or "resonance" in q or "resonate" in q or "f0" in q) and ("capacitance" in q or re.search(r"\bC\b", text)) and ("inductance" in q or re.search(r"\bL\b", text)):
            if "capacitance" in q or "capacitor value" in q or "required c" in q or "calculate c" in q:
                return "lc_required_capacitance"
            if "inductance" in q or "required l" in q or "calculate l" in q:
                return "lc_required_inductance"
        if "parallel plate" in q and ("electric field" in q or "field magnitude" in q or "between them" in q):
            return "parallel_plate_electric_field"
        if "point charge" in q and "electric field" in q:
            return "point_charge_electric_field"
        if "series rlc" in q or "power factor" in q or "impedance" in q:
            return "series_rlc_operating_point"
        if "capacitor" in q and "charge" in q and "voltage" in q:
            return "capacitor_charge_cv"
        if "capacitor" in q and "charge" in q and ("potential difference" in q or "voltage" in q):
            return "capacitor_voltage_q_over_c"
        if "capacitor" in q and ("energy" in q or "stored" in q):
            return "capacitor_energy_cv"
        if "solenoid" in q and "magnetic field" in q:
            return "solenoid_magnetic_field"
        if "solenoid" in q and "inductance" in q:
            return "solenoid_self_inductance"
        return ""
    def _coerce_llm_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        out = dict(raw or {})
        family = self._raw_get_any(out, ["family", "formula_family", "problem_family", "problem_type", "category", "type"])
        cq = self._raw_get_any(out, ["canonical_question", "rewritten_question", "canonical", "paraphrased_question", "question"])
        target = self._raw_get_any(out, ["target", "target_quantity", "quantity_to_find", "asked_quantity"])
        unit = self._raw_get_any(out, ["target_unit", "unit", "output_unit", "answer_unit"])
        confidence = self._raw_get_any(out, ["confidence", "score"])
        if family is not None:
            fam = str(family).strip().lower().replace(" ", "_").replace("-", "_")
            aliases = {
                "lc_resonance_required_capacitance": "lc_required_capacitance",
                "required_capacitance": "lc_required_capacitance",
                "resonance_capacitance": "lc_required_capacitance",
                "lc_capacitance": "lc_required_capacitance",
                "lc_resonance_required_inductance": "lc_required_inductance",
                "required_inductance": "lc_required_inductance",
                "resonance_inductance": "lc_required_inductance",
                "electric_field_parallel_plates": "parallel_plate_electric_field",
                "point_charge_field": "point_charge_electric_field",
                "rlc": "series_rlc_operating_point",
            }
            out["family"] = aliases.get(fam, fam)
        if not out.get("canonical_question") and cq:
            out["canonical_question"] = str(cq).strip()
        if not out.get("target") and target:
            out["target"] = str(target).strip()
        if "target_unit" not in out and unit is not None:
            out["target_unit"] = unit
        if "confidence" not in out and confidence is not None:
            out["confidence"] = confidence
        if not isinstance(out.get("quantities"), list):
            out["quantities"] = []
        if not isinstance(out.get("relations"), list):
            out["relations"] = []
        if not out.get("family") and out.get("canonical_question"):
            out["family"] = self._infer_family_from_text(str(out["canonical_question"]))
        if not out.get("target_unit"):
            inferred_unit = _find_output_unit(str(out.get("canonical_question") or ""))
            if inferred_unit:
                out["target_unit"] = inferred_unit
        if not out.get("confidence"):
            out["confidence"] = 0.72 if out.get("canonical_question") and out.get("family") else 0.0
        return out
    def _validate_llm_output(self, original_question: str, raw: dict[str, Any], *, rule_warnings: list[str] | None = None) -> CanonicalizationResult:
        warnings = list(rule_warnings or [])
        coerced = self._coerce_llm_raw(raw)
        family = str(coerced.get("family") or "").strip()
        cq = str(coerced.get("canonical_question") or "").strip()
        unit = _unit_clean(str(coerced.get("target_unit") or "").strip()) or None
        try:
            conf = float(coerced.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if family not in CANONICAL_FAMILIES:
            warnings.append(f"PHYSICS_CANONICALIZER_LLM_REJECTED: unsupported family {family!r}")
            return CanonicalizationResult(warnings=warnings, raw={"raw": raw, "coerced": coerced})
        if not cq or len(cq) > 800:
            warnings.append("PHYSICS_CANONICALIZER_LLM_REJECTED: empty or too long canonical question")
            return CanonicalizationResult(warnings=warnings, raw={"raw": raw, "coerced": coerced})
        if conf < 0.55:
            warnings.append("PHYSICS_CANONICALIZER_LLM_REJECTED: low confidence")
            return CanonicalizationResult(warnings=warnings, raw={"raw": raw, "coerced": coerced})
        orig_nums = _numeric_tokens(original_question)
        can_nums = _numeric_tokens(cq)
        for n in can_nums:
            if not any(_num_equiv(n, o) for o in orig_nums):
                warnings.append(f"PHYSICS_CANONICALIZER_LLM_REJECTED: canonical question introduced number {n}")
                return CanonicalizationResult(warnings=warnings, raw={"raw": raw, "coerced": coerced})
        if re.search(r"\b(answer|final answer|equals?)\b", cq, flags=re.I):
            warnings.append("PHYSICS_CANONICALIZER_LLM_REJECTED: canonical question appears to contain answer wording")
            return CanonicalizationResult(warnings=warnings, raw={"raw": raw, "coerced": coerced})
        return CanonicalizationResult(
            accepted=True,
            canonical_question=cq,
            family=family,
            method="llm",
            target_unit=unit,
            confidence=conf,
            warnings=warnings,
            raw={"raw": raw, "coerced": coerced},
            evidence={"llm_quantities": coerced.get("quantities"), "relations": coerced.get("relations")},
        )
    @staticmethod
    def _llm_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["family", "canonical_question", "target", "target_unit", "quantities", "relations", "confidence"],
            "properties": {
                "family": {"type": "string", "enum": sorted(CANONICAL_FAMILIES)},
                "canonical_question": {"type": "string"},
                "target": {"type": "string"},
                "target_unit": {"type": ["string", "null"]},
                "quantities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["symbol", "value", "unit", "source_text"],
                        "properties": {
                            "symbol": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "source_text": {"type": "string"},
                        },
                    },
                },
                "relations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
    def _llm_system_prompt(self) -> str:
        if self._loaded_system_prompt is not None:
            return self._loaded_system_prompt
        if self.prompt_path:
            try:
                path = Path(self.prompt_path)
                if path.exists():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        self._loaded_system_prompt = text
                        return text
            except Exception:
                pass
        self._loaded_system_prompt = (
            "You are a physics question canonicalizer for a deterministic solver. "
            "Do not solve the problem. Do not compute or include the final answer. "
            "Rewrite only into a solver-friendly canonical physics word problem. "
            "Preserve all numeric values and units exactly as written unless only normalizing obvious unit spelling, such as volts -> V or microfarads -> μF. "
            "Do not add any number that is not present in the original question. "
            "Return exactly one JSON object conforming to this schema: "
            + json.dumps(PhysicsCanonicalizer._llm_schema(), ensure_ascii=False)
        )
        return self._loaded_system_prompt
    @staticmethod
    def _llm_user_prompt(question: str, *, baseline: SolverResult | None) -> str:
        schema = PhysicsCanonicalizer._llm_schema()
        payload = {
            "task": "Canonicalize this physics question for a deterministic solver. Do not solve it.",
            "original_question": question,
            "baseline_solver": {
                "answer": baseline.answer if baseline else None,
                "warnings": baseline.warnings if baseline else [],
                "confidence": baseline.confidence if baseline else None,
            },
            "required_json_schema": schema,
            "allowed_families": sorted(CANONICAL_FAMILIES),
            "canonical_style_examples": [
                {
                    "family": "capacitor_energy_cv",
                    "canonical_question": "A capacitor has capacitance C = 10 μF and voltage U = 100 V. Calculate the energy stored in the capacitor in J.",
                    "target": "energy stored",
                    "target_unit": "J",
                },
                {
                    "family": "point_charge_electric_field",
                    "canonical_question": "A point charge Q = 2 μC is in a medium with dielectric constant εr = 2. Calculate the electric field magnitude at distance r = 10 cm.",
                    "target": "electric field magnitude",
                    "target_unit": "N/C",
                },
                {
                    "family": "lc_required_capacitance",
                    "canonical_question": "An LC circuit has inductance L = 0.109 H and resonant frequency f0 = 100 Hz. Calculate the required capacitance C in μF.",
                    "target": "required capacitance C",
                    "target_unit": "μF",
                },
            ],
            "output_template": {
                "family": "one allowed family string",
                "canonical_question": "solver-friendly rewritten question, no answer",
                "target": "quantity being asked",
                "target_unit": "unit requested, or null",
                "quantities": [
                    {"symbol": "L", "value": "0.109", "unit": "H", "source_text": "0.109 H"}
                ],
                "relations": ["resonance"],
                "confidence": 0.85,
            },
            "instructions": [
                "Return exactly one JSON object, no markdown, no prose.",
                "No answer field.",
                "No hidden calculations.",
                "Preserve all visible numbers; do not introduce computed values.",
                "For LC required capacitance, family must be lc_required_capacitance.",
                "For LC required inductance, family must be lc_required_inductance.",
                "Use canonical_question exactly as the key name.",
                "If ambiguous, set confidence below 0.55.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
