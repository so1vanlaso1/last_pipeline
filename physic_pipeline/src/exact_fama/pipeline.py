from __future__ import annotations

from typing import Any

from .config import Settings, load_settings
from .explanation import ExplanationGenerator
from .llm.qwen_client import QwenClient
from .llm.structured_parser import StructuredParseResult, StructuredParser
from .llm.logic_semantic_head import LogicSemanticHead
from .logic.solver import solve_logic
from .physics.solver import solve_physics
from .physics_canonicalizer import PhysicsCanonicalizer
from .router import route_task
from .schemas import PredictRequest, PredictResponse, SolverResult
from .validation import validate_result
from .verification import verify_result


_FATAL_WARNING_PREFIXES = (
    "LOGIC_ENTAILMENT_ERROR",
    "OUTPUT_SCHEMA_ERROR",
)


def _warnings_text(result: SolverResult) -> str:
    return " ".join(str(w) for w in (result.warnings or []))


def _has_fatal_warning(result: SolverResult) -> bool:
    text = _warnings_text(result)
    return any(prefix in text for prefix in _FATAL_WARNING_PREFIXES)


def _has_physics_formula_error(result: SolverResult) -> bool:
    return "PHYSICS_FORMULA_ERROR" in _warnings_text(result)


def _should_try_physics_canonicalizer(result: SolverResult, *, policy: str) -> bool:
    if str(policy or "").strip().lower() == "always":
        return True
    if str(result.answer or "").strip() == "Uncertain":
        return True
    if _has_physics_formula_error(result):
        return True
    try:
        if float(result.confidence or 0.0) < 0.35:
            return True
    except Exception:
        return True
    return False


def _has_explicit_solver_proof(result: SolverResult) -> bool:
    debug = result.debug or {}
    proof_steps = debug.get("proof_steps") or []
    if proof_steps:
        return True

    mcq_debug = debug.get("mcq_debug") or {}
    decision = str(mcq_debug.get("decision") or "")
    if decision.startswith("selected_"):
        return True

    return False


def _candidate_can_override_baseline(baseline: SolverResult, candidate: SolverResult) -> bool:
    if candidate.answer == "Uncertain":
        return False
    if _has_fatal_warning(candidate):
        return False
    if not _has_explicit_solver_proof(candidate):
        return False

    baseline_reason = str((baseline.debug or {}).get("inference_reason") or "")

    # Safest and most useful case: parser helps the solver prove something that
    # the baseline could not prove.
    if baseline.answer == "Uncertain":
        return True

    # Requirement-checking questions sometimes use conservative closed-world No.
    # If the parser supplies an explicit proof for Yes, allow the solver-backed
    # candidate to win. This still does not trust the LLM answer directly.
    if baseline.answer == "No" and candidate.answer == "Yes" and "not_entailed" in baseline_reason:
        return True

    return False


def _choose_logic_result(
    baseline: SolverResult,
    candidate: SolverResult | None,
    parser_result: StructuredParseResult,
    policy: str,
) -> SolverResult:
    if candidate is None or not parser_result.accepted:
        baseline.debug["structured_parser"] = {
            "accepted": False,
            "warnings": parser_result.warnings,
        }
        return baseline

    policy = (policy or "safe_override").strip().lower()

    if policy == "baseline":
        chosen = baseline
        decision = "baseline_policy"
    elif candidate.answer == baseline.answer:
        chosen = candidate
        decision = "same_answer_candidate_trace"
    elif policy == "same_answer_only":
        chosen = baseline
        decision = "rejected_different_answer_same_answer_only"
    elif policy == "safe_override" and _candidate_can_override_baseline(baseline, candidate):
        chosen = candidate
        decision = "safe_solver_backed_override"
    else:
        chosen = baseline
        decision = "rejected_different_answer_not_safe"

    chosen.debug["structured_parser"] = {
        "accepted": parser_result.accepted,
        "decision": decision,
        "warnings": parser_result.warnings,
        "raw": parser_result.raw,
        "baseline_answer": baseline.answer,
        "candidate_answer": candidate.answer,
    }

    if chosen is candidate:
        chosen.warnings.extend(
            str(w) for w in parser_result.warnings
            if not str(w).startswith("STRUCTURED_PARSE_REJECTED")
        )
        chosen.warnings.append(f"LOGIC_INFO: structured parser decision = {decision}.")
    else:
        # Keep diagnostics without polluting the answer too much.
        if parser_result.warnings:
            chosen.warnings.append(
                "LOGIC_INFO: structured parser output was not used; see debug."
            )

    return chosen


class ExactFamaPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.llm = QwenClient(self.settings.model)
        use_llm_expl = bool(self.settings.pipeline.get("use_llm_for_explanation", False))
        use_llm_parse = bool(self.settings.pipeline.get("use_llm_for_structured_parse", False))
        self.parser_policy = str(self.settings.pipeline.get("parser_answer_policy", "safe_override"))
        use_logic_head = bool(self.settings.pipeline.get("use_llm_for_logic_semantic_head", False))
        logic_head_policy = str(self.settings.pipeline.get("logic_semantic_head_policy", "safe"))
        self.parser = StructuredParser(self.llm, use_llm=use_llm_parse)
        # IMPORTANT: the generic LLM structured parser is logic-only.
        # Physics uses PhysicsCanonicalizer, which never answers directly and
        # only rewrites failed physics questions before a deterministic solver retry.
        self.logic_semantic_head = LogicSemanticHead(self.llm, use_llm=use_logic_head, policy=logic_head_policy)
        self.explainer = ExplanationGenerator(self.llm, use_llm=use_llm_expl)

        # Physics-only canonicalization retry.  This is intentionally separate
        # from the logic structured parser: it never answers directly.  It only
        # rewrites a physics question into solver-friendly wording, then calls
        # the deterministic physics solver again under a strict gate.
        physics_can_cfg = self.settings.pipeline.get("physics_canonicalizer", {}) or {}
        self.physics_canonicalizer = PhysicsCanonicalizer(
            self.llm,
            enabled=bool(self.settings.pipeline.get(
                "use_physics_canonicalizer",
                physics_can_cfg.get("enabled", True),
            )),
            use_llm=bool(self.settings.pipeline.get(
                "use_llm_for_physics_canonicalizer",
                physics_can_cfg.get("use_llm", False),
            )),
            policy=str(self.settings.pipeline.get(
                "physics_canonicalizer_policy",
                physics_can_cfg.get("policy", "fallback_only"),
            )),
            min_candidate_confidence=float(physics_can_cfg.get("min_candidate_confidence", 0.45)),
            max_llm_retries=int(physics_can_cfg.get("max_llm_retries", 1)),
            prompt_path=str(physics_can_cfg.get("prompt_path", "") or ""),
        )

    def predict(self, request: PredictRequest) -> PredictResponse:
        task_type = route_task(request)
        used_modules = ["input_normalizer", "task_router"]

        if task_type == "logic":
            parser_result = self.parser.parse_with_diagnostics(request, task_type)
            if parser_result.accepted:
                used_modules.append("llm_structured_parser")
        else:
            # Keep physics fully separate from the generic LLM structured parser.
            # Any physics LLM use happens only inside PhysicsCanonicalizer and is
            # followed by a deterministic solve_physics retry.
            parser_result = StructuredParseResult()

        if task_type == "logic":
            baseline = solve_logic(request, structured_parse={})
            used_modules.extend(["logic_rule_parser", "logic_forward_chainer", "logic_verifier"])

            candidate: SolverResult | None = None
            if parser_result.accepted:
                candidate = solve_logic(request, structured_parse=parser_result.data)

            result = _choose_logic_result(
                baseline=baseline,
                candidate=candidate,
                parser_result=parser_result,
                policy=self.parser_policy,
            )

            before_head = result.answer
            result = self.logic_semantic_head.refine(
                request,
                result,
                baseline=baseline,
                candidate=candidate,
                parser_result=parser_result,
            )
            if (result.debug or {}).get("logic_semantic_head", {}).get("enabled"):
                used_modules.append("llm_logic_semantic_head")
                if result.answer != before_head:
                    used_modules.append("llm_logic_semantic_override")
        else:
            # Do not feed zero-shot LLM physics parses to a solver contract that
            # currently owns quantity/formula extraction itself.
            result = solve_physics(request, structured_parse={})
            used_modules.extend(["physics_quantity_extractor", "physics_formula_solver", "physics_unit_verifier"])
            if (
                self.physics_canonicalizer.enabled
                and _should_try_physics_canonicalizer(result, policy=self.physics_canonicalizer.policy)
            ):
                attempts: list[dict[str, Any]] = []

                def _try_physics_canonical_retry(*, force_llm: bool = False) -> tuple[SolverResult, bool, Any]:
                    canonical = self.physics_canonicalizer.canonicalize(
                        request.question,
                        baseline=result,
                        force_llm=force_llm,
                        prior_warnings=[a.get("decision", "") for a in attempts],
                    )
                    if not canonical.accepted:
                        attempts.append({
                            "method": "llm" if force_llm else canonical.method,
                            "accepted": False,
                            "family": canonical.family,
                            "canonical_question": canonical.canonical_question,
                            "warnings": canonical.warnings,
                            "decision": "no_canonical_rewrite",
                        })
                        return result, False, canonical

                    used_modules.append("physics_llm_paraphrase" if canonical.method == "llm" else "physics_canonicalizer")
                    canonical_request = self.physics_canonicalizer.make_request(request, canonical)
                    canonical_candidate = solve_physics(canonical_request, structured_parse={})
                    chosen = self.physics_canonicalizer.choose_result(
                        baseline=result,
                        candidate=canonical_candidate,
                        canonical=canonical,
                    )
                    accepted = chosen is canonical_candidate
                    decision = (chosen.debug or {}).get("physics_canonicalizer", {}).get("decision")
                    attempts.append({
                        "method": canonical.method,
                        "accepted": canonical.accepted,
                        "family": canonical.family,
                        "canonical_question": canonical.canonical_question,
                        "target_unit": canonical.target_unit,
                        "candidate_answer": canonical_candidate.answer,
                        "candidate_confidence": canonical_candidate.confidence,
                        "llm_raw": canonical.raw if canonical.method == "llm" else None,
                        "decision": decision or ("accepted" if accepted else "rejected"),
                    })
                    return chosen, accepted, canonical

                # Attempt 1: rule-based canonicalization, fast and deterministic.
                chosen, accepted, first_canonical = _try_physics_canonical_retry(force_llm=False)
                if accepted:
                    result = chosen
                    used_modules.append("physics_canonical_solver_retry")
                else:
                    # Attempt 2: if rule-based paraphrase existed but the solver still
                    # returned Uncertain, force the physics-specific LLM paraphraser.
                    # This is the path you expected to see for cases like PRT005027.
                    # It remains separate from StructuredParser and still does not answer.
                    if (
                        self.physics_canonicalizer.use_llm
                        and self.physics_canonicalizer.llm is not None
                        and self.physics_canonicalizer.llm.backend != "none"
                    ):
                        chosen2, accepted2, _second_canonical = _try_physics_canonical_retry(force_llm=True)
                        if accepted2:
                            result = chosen2
                            used_modules.append("physics_llm_paraphrase_solver_retry")
                        else:
                            used_modules.append("physics_canonicalizer_rejected")
                    else:
                        used_modules.append("physics_canonicalizer_rejected")

                result.debug = dict(result.debug or {})
                result.debug["physics_canonicalizer_attempts"] = attempts
                if "physics_canonicalizer" not in result.debug:
                    result.debug["physics_canonicalizer"] = {
                        "enabled": self.physics_canonicalizer.enabled,
                        "attempted": bool(attempts),
                        "decision": attempts[-1]["decision"] if attempts else "not_attempted",
                        "attempts": attempts,
                    }

        result = verify_result(request, result, task_type=task_type)
        used_modules.append("answer_verifier")

        result = self.explainer.rewrite(request, result)
        used_modules.append("explanation_generator")
        used_modules.append("output_validator")
        return validate_result(result, task_type=task_type, used_modules=used_modules)


_pipeline: ExactFamaPipeline | None = None


def get_pipeline() -> ExactFamaPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ExactFamaPipeline()
    return _pipeline
