#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr

from exact_fama.pipeline import ExactFamaPipeline
from exact_fama.schemas import PredictRequest


PIPELINE: ExactFamaPipeline | None = None


def get_pipeline() -> ExactFamaPipeline:
    global PIPELINE

    if PIPELINE is None:
        PIPELINE = ExactFamaPipeline()

    return PIPELINE


def parse_premises(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def run_predict(
    task_type: str,
    question: str,
    premises_text: str,
    show_debug: bool,
) -> tuple[str, str, str, str]:
    start = time.perf_counter()

    if task_type == "logic":
        row = {
            "type": "logic",
            "premises-NL": parse_premises(premises_text),
            "question": question.strip(),
        }
    else:
        row = {
            "type": "physics",
            "question": question.strip(),
        }

    pipeline = get_pipeline()
    req = PredictRequest.model_validate(row)
    pred = pipeline.predict(req).model_dump(mode="json")

    elapsed = time.perf_counter() - start

    answer_block = json.dumps(
        {
            "answer": pred.get("answer"),
            "unit": pred.get("unit"),
            "confidence": pred.get("confidence"),
            "task_type": pred.get("task_type"),
            "prediction_time_seconds": round(elapsed, 3),
        },
        indent=2,
        ensure_ascii=False,
    )

    explanation = pred.get("explanation") or ""

    trace = {
        "cot": pred.get("cot") or [],
        "premises": pred.get("premises") or [],
        "warnings": pred.get("warnings") or [],
        "llm_status": {
            "llm_rewrite_enabled": (pred.get("debug") or {}).get("llm_rewrite_enabled"),
            "llm_rewrite_changed": (pred.get("debug") or {}).get("llm_rewrite_changed"),
            "llm_rewrite_failed": (pred.get("debug") or {}).get("llm_rewrite_failed"),
        },
        "proof_steps": (pred.get("debug") or {}).get("proof_steps"),
        "formula": (pred.get("debug") or {}).get("formula"),
        "quantities": (pred.get("debug") or {}).get("quantities"),
    }

    trace_block = json.dumps(trace, indent=2, ensure_ascii=False)

    if show_debug:
        raw_json = json.dumps(pred, indent=2, ensure_ascii=False)
    else:
        raw_json = json.dumps(
            {
                "answer": pred.get("answer"),
                "unit": pred.get("unit"),
                "explanation": pred.get("explanation"),
                "warnings": pred.get("warnings"),
            },
            indent=2,
            ensure_ascii=False,
        )

    return answer_block, explanation, trace_block, raw_json


def pipeline_info() -> str:
    pipeline = get_pipeline()

    info = {
        "EXACT_FAMA_CONFIG": os.environ.get("EXACT_FAMA_CONFIG"),
        "LLM_BACKEND": os.environ.get("LLM_BACKEND"),
        "MODEL_NAME": os.environ.get("MODEL_NAME"),
        "LORA_ADAPTER_PATH": os.environ.get("LORA_ADAPTER_PATH"),
        "LLM_TEMPERATURE": os.environ.get("LLM_TEMPERATURE"),
        "LLM_MAX_NEW_TOKENS": os.environ.get("LLM_MAX_NEW_TOKENS"),
        "resolved_backend": pipeline.llm.backend,
        "resolved_model": pipeline.llm.model_name,
        "use_llm_parse": pipeline.parser.use_llm,
        "use_llm_rewrite": pipeline.explainer.use_llm,
    }

    return json.dumps(info, indent=2, ensure_ascii=False)


with gr.Blocks(title="EXACT-FAMA Local Tester") as demo:
    gr.Markdown("# EXACT-FAMA Local Tester")
    gr.Markdown(
        "Full pipeline: input → solver → optional Qwen rewrite → output. "
        "Trace bên dưới là public reasoning/debug trace, không phải hidden chain-of-thought."
    )

    with gr.Row():
        task_type = gr.Radio(
            choices=["logic", "physics"],
            value="physics",
            label="Task type",
        )

        show_debug = gr.Checkbox(
            value=False,
            label="Show full debug JSON",
        )

    premises = gr.Textbox(
        label="Premises-NL, mỗi dòng một premise. Chỉ dùng cho logic.",
        lines=8,
        placeholder="If a curriculum is well-structured and has exercises, it enhances student engagement.\nThe curriculum has practical exercises.",
    )

    question = gr.Textbox(
        label="Question",
        lines=6,
        placeholder="Nhập câu hỏi ở đây...",
    )

    with gr.Row():
        run_btn = gr.Button("Predict", variant="primary")
        info_btn = gr.Button("Show pipeline info")

    answer_out = gr.Code(label="Answer summary", language="json")
    explanation_out = gr.Textbox(label="Explanation", lines=8)
    trace_out = gr.Code(label="Public reasoning trace", language="json")
    raw_out = gr.Code(label="Raw output JSON", language="json")

    info_out = gr.Code(label="Pipeline info", language="json")

    run_btn.click(
        fn=run_predict,
        inputs=[task_type, question, premises, show_debug],
        outputs=[answer_out, explanation_out, trace_out, raw_out],
    )

    info_btn.click(
        fn=pipeline_info,
        inputs=[],
        outputs=[info_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
