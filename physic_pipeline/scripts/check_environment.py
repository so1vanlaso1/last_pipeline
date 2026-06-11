#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing first.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import platform


def main() -> None:
    print("=== EXACT-FAMA environment check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            print(f"GPU: {props.name}")
            print(f"VRAM: {props.total_memory / (1024 ** 3):.2f} GB")
            print(f"CUDA runtime in torch: {torch.version.cuda}")
        else:
            print("GPU not visible to PyTorch. Solver-only mode still works.")
    except Exception as exc:
        print(f"PyTorch not available or failed to import: {exc}")

    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except Exception as exc:
        print(f"Transformers not available: {exc}")

    try:
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
    except Exception as exc:
        print(f"FastAPI not available: {exc}")

    print("\nRecommended first validation commands:")
    print("  pytest -q")
    print("  python scripts/run_inference.py --input data/sample/dev.jsonl --output artifacts/predictions.jsonl")


if __name__ == "__main__":
    main()
