#!/usr/bin/env python
from __future__ import annotations
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
changed = sum(1 for r in rows if (r.get('debug') or {}).get('llm_rewrite_changed') is True)
enabled = sum(1 for r in rows if (r.get('debug') or {}).get('llm_rewrite_enabled') is True)
failed = sum(1 for r in rows if (r.get('debug') or {}).get('llm_rewrite_failed'))
print(json.dumps({
    'rows': len(rows),
    'llm_rewrite_enabled': enabled,
    'llm_rewrite_changed': changed,
    'llm_rewrite_failed': failed,
    'changed_rate': changed / len(rows) if rows else 0,
}, indent=2, ensure_ascii=False))
