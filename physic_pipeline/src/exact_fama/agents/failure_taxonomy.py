from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    ROUTING_ERROR = "ROUTING_ERROR"
    LOGIC_PARSE_ERROR = "LOGIC_PARSE_ERROR"
    LOGIC_ENTAILMENT_ERROR = "LOGIC_ENTAILMENT_ERROR"
    PHYSICS_PARSE_ERROR = "PHYSICS_PARSE_ERROR"
    PHYSICS_UNIT_ERROR = "PHYSICS_UNIT_ERROR"
    PHYSICS_FORMULA_ERROR = "PHYSICS_FORMULA_ERROR"
    EXPLANATION_WEAK = "EXPLANATION_WEAK"
    OUTPUT_SCHEMA_ERROR = "OUTPUT_SCHEMA_ERROR"
    ANSWER_WRONG = "ANSWER_WRONG"


MITIGATION_MAP = {
    FailureType.ROUTING_ERROR: ["router_verifier", "schema_normalizer"],
    FailureType.LOGIC_PARSE_ERROR: ["logic_parser", "nl_to_rule_prompt"],
    FailureType.LOGIC_ENTAILMENT_ERROR: ["logic_verifier", "premise_tracker"],
    FailureType.PHYSICS_PARSE_ERROR: ["physics_quantity_extractor", "unit_normalizer"],
    FailureType.PHYSICS_UNIT_ERROR: ["physics_unit_verifier"],
    FailureType.PHYSICS_FORMULA_ERROR: ["formula_selector", "formula_bank"],
    FailureType.EXPLANATION_WEAK: ["explanation_rewriter", "cot_formatter"],
    FailureType.OUTPUT_SCHEMA_ERROR: ["output_validator"],
    FailureType.ANSWER_WRONG: ["solver_verifier", "regression_tests"],
}
