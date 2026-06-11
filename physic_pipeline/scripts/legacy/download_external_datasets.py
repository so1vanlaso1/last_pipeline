from datasets import load_dataset
from pathlib import Path

OUT = Path("data/external")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # Logic / FOL
    "tasksource/proofwriter",
    "tasksource/folio",
    "yale-nlp/P-FOLIO",

    # Physics / STEM
    "lmms-lab/SciBench_Physics",
    "TIGER-Lab/TheoremQA",
    "zhibei1204/PhysReason",
    "afdsafas/EEE-Bench",
    "xw27/scibench",
]

for name in DATASETS:
    safe_name = name.replace("/", "__")
    out_dir = OUT / safe_name

    print(f"\n=== Downloading {name} ===")
    try:
        ds = load_dataset(name)
        ds.save_to_disk(str(out_dir))
        print(f"Saved to {out_dir}")
        print(ds)
    except Exception as e:
        print(f"FAILED: {name}")
        print(e)
