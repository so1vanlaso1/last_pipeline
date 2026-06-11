"""Data structures shared across the cascade pipeline.

Deliberately uses plain dataclasses (no pydantic) so the data/prompt/logging
layers import with zero heavy dependencies — the model layer is the only part
that needs torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerType(str, Enum):
    # The dataset's non-MCQ label set is {Yes, No, Unknown}. "Unknown" is what
    # the prompt surfaces to the user as "Not Given".
    YES_NO_UNKNOWN = "yes_no_unknown"
    MCQ = "mcq"
    OPEN_ENDED = "open_ended"


@dataclass
class Record:
    """One question to answer. `answer`/`options` come from the dataset; only
    `premises_nl` + `question_nl` (+ parsed MCQ `options`) are ever shown to a
    model — the gold `answer` is held back for optional scoring."""

    id: str
    premises_nl: list[str]
    question_nl: str
    answer_type: AnswerType
    answer: str | None = None          # gold, canonicalized; NEVER fed to a model
    options: list[str] | None = None   # MCQ option texts, in A,B,C… order
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelReply:
    """Exactly what one model did on one record — kept verbatim for the logs."""

    model_label: str       # e.g. "Qwen-4B" / "Gemma-E2B" / "Gemma-E4B(8B)"
    model_id: str          # HF repo id (or "stub:…")
    prompt: str            # the exact text fed to the model (rendered template)
    raw: str               # the exact raw completion text
    answer: str | None     # canonical: "A"/"B"/… (MCQ) or "Yes"/"No"/"Unknown"
    answer_display: str     # human readable, e.g. "A. Sophia qualifies…"
    explanation: str
    elapsed_s: float = 0.0
    # Soft-vote bookkeeping: the weight this model contributes to its label, and
    # its size class ("4b" → 1.0, "8b" → 1.5 by default). Set from the model when
    # the reply is produced (see cascade.query).
    weight: float = 1.0
    model_class: str = "4b"
    # Judge-flow bookkeeping: what this reply was in the flow ("voter" for the
    # weighted vote, "generator"/"judge" for the generate→judge flow), and the
    # 0-based indices of the premises the model said it used.
    role: str = "voter"
    premises_used: list[int] = field(default_factory=list)


@dataclass
class FinalAnswer:
    id: str
    answer_type: AnswerType
    answer: str | None         # canonical final answer
    answer_display: str
    explanation: str
    decider: str               # which path produced it (see cascade.DECIDER_*)
    agreed: bool               # True iff every model that voted picked this label
    confidence: float          # winning weight / total weight (the vote margin)
    replies: list[ModelReply] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # label → summed weight
    gold: str | None = None    # only set when --show-gold
    elapsed_s: float = 0.0
    premises_used: list[int] = field(default_factory=list)  # 0-based, judge flow
