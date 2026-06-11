#!/usr/bin/env bash
# Convenience wrapper around run_cascade.py. Activates the venv setup.sh made and
# forwards all arguments. Examples:
#   ./run.sh --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold --limit 20
#   ./run.sh --stages 4b,gemma8b --precision 8bit --only mcq
#   ./run.sh --stages liquid8b --precision bf16 --think --show-gold
#   ./run.sh --backend stub --stages 4b,gemma8b,liquid8b --show-gold  # no-GPU wiring test
set -euo pipefail
cd "$(dirname "$0")"
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
exec python run_cascade.py "$@"
