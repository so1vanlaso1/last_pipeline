#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact_fama.pipeline import ExactFamaPipeline
from exact_fama.schemas import PredictRequest


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def print_pipeline_info(pipeline: ExactFamaPipeline) -> None:
    print("\n=== PIPELINE SETTINGS ===")
    print("EXACT_FAMA_CONFIG:", os.environ.get("EXACT_FAMA_CONFIG"))
    print("LLM_BACKEND:", os.environ.get("LLM_BACKEND"))
    print("MODEL_NAME:", os.environ.get("MODEL_NAME"))
    print("LORA_ADAPTER_PATH:", os.environ.get("LORA_ADAPTER_PATH"))
    print("LLM_TEMPERATURE:", os.environ.get("LLM_TEMPERATURE"))
    print("LLM_MAX_NEW_TOKENS:", os.environ.get("LLM_MAX_NEW_TOKENS"))

    print("Resolved backend:", pipeline.llm.backend)
    print("Resolved model:", pipeline.llm.model_name)
    print("Use LLM parse:", pipeline.parser.use_llm)
    print("Use LLM rewrite:", pipeline.explainer.use_llm)
    print("=========================\n")


def parse_premises_smart(text: str) -> list[str]:
    """
    Hàm phân tích premise thông minh. Hỗ trợ:
    1. Dạng list JSON: ["A", "B"]
    2. Dạng chuỗi dataset: "A", "B", "C"
    3. Dạng text thông thường (mỗi câu 1 dòng)
    """
    text = text.strip()
    if not text:
        return []

    # 1. Thử parse dạng JSON array: ["A", "B"]
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
        except Exception:
            pass

    # 2. Thử parse dạng Python strings (như dataset của bạn): "A", "B", "C"
    try:
        # Ép kiểu thành dạng list của Python
        parsed = ast.literal_eval(f"[{text}]")
        if isinstance(parsed, list):
            return [str(i) for i in parsed]
    except Exception:
        pass

    # 3. Fallback dùng Regex: Bắt mọi chuỗi nằm trong ngoặc kép (phòng hờ format lỗi)
    if '"' in text:
        matches = re.findall(r'"(.*?)"', text)
        if matches:
            return [m.strip() for m in matches if m.strip()]

    # 4. Fallback cuối cùng: Tách theo từng dòng như cũ
    return [line.strip() for line in text.splitlines() if line.strip()]


def multiline_input(prompt: str, end_token: str = "DONE") -> str:
    print(prompt)
    print(f"(Nhấn Enter rồi gõ chữ '{end_token}' trên một dòng riêng biệt để xác nhận hoàn tất)")
    lines: list[str] = []

    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == end_token.upper():
            break
        lines.append(line)

    return "\n".join(lines).strip()


def build_logic_request() -> dict[str, Any]:
    print("\n--- NHẬP PREMISES ---")
    print("Bạn có thể copy-paste toàn bộ format từ dataset vào đây (VD: \"Câu 1\", \"Câu 2\").")
    premises_text = multiline_input("Nhập premises:", end_token="DONE")
    
    premises = parse_premises_smart(premises_text)
    
    print(f"\n[OK] Đã nhận diện được {len(premises)} premises hợp lệ.")

    print("\n--- NHẬP QUESTION ---")
    question = multiline_input("Nhập câu hỏi logic:", end_token="DONE")

    return {
        "type": "logic",
        "premises-NL": premises,
        "question": question,
    }


def build_physics_request() -> dict[str, Any]:
    print("\n--- NHẬP PHYSICS QUESTION ---")
    question = multiline_input("Nhập câu hỏi physics:", end_token="DONE")

    return {
        "type": "physics",
        "question": question,
    }


def print_public_trace(pred: dict[str, Any]) -> None:
    print("\n=== PUBLIC TRACE ===")
    print("\nAnswer:", pred.get("answer"))
    print("Unit:", pred.get("unit"))
    print("Confidence:", pred.get("confidence"))

    print("\nExplanation:")
    print(pred.get("explanation") or "")

    print("\nCOT / Solver trace:")
    cot = pred.get("cot") or []
    if cot:
        for i, step in enumerate(cot, 1):
            print(f"{i}. {step}")
    else:
        print("(empty)")

    debug = pred.get("debug") or {}

    if debug.get("proof_steps"):
        print("\nProof steps:")
        print(json.dumps(debug["proof_steps"], indent=2, ensure_ascii=False))

    warnings = pred.get("warnings") or []
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("-", w)
    print("====================")


def run_once(pipeline: ExactFamaPipeline) -> None:
    print("\n" + "="*40)
    print("CHỌN TASK BẠN MUỐN CHẠY:")
    print("1. Logic (Premises + Question)")
    print("2. Physics (Question only)")
    print("3. Nhập raw JSON")
    print("q. Thoát (Quit)")
    print("="*40)

    choice = input("Lựa chọn của bạn > ").strip().lower()

    if choice in {"q", "quit", "exit"}:
        raise KeyboardInterrupt

    if choice == "1":
        row = build_logic_request()
    elif choice == "2":
        row = build_physics_request()
    elif choice == "3":
        raw = multiline_input("\nPaste JSON object:", end_token="DONE")
        row = json.loads(raw)
    else:
        print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
        return

    print("\n[Đang chạy suy luận...]")
    start = time.perf_counter()
    req = PredictRequest.model_validate(row)
    pred = pipeline.predict(req).model_dump(mode="json")
    elapsed = time.perf_counter() - start

    print(f"\n✅ Suy luận hoàn tất trong: {format_duration(elapsed)}")

    print_public_trace(pred)


def main() -> None:
    print("="*50)
    print("KHỞI ĐỘNG EXACT-FAMA PIPELINE")
    print("Đang nạp Model Weights vào RAM/GPU. Quá trình này có thể mất vài phút...")
    print("Vui lòng không tắt chương trình!")
    print("="*50)
    
    start = time.perf_counter()
    pipeline = ExactFamaPipeline()
    elapsed = time.perf_counter() - start
    
    print(f"✅ Model đã sẵn sàng! (Thời gian nạp: {format_duration(elapsed)})")
    print_pipeline_info(pipeline)

    while True:
        try:
            run_once(pipeline)
        except KeyboardInterrupt:
            print("\nĐang thoát chương trình. Tạm biệt!")
            break
        except Exception as exc:
            print(f"\n❌ LỖI ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    main()