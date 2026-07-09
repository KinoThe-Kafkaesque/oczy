#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/eval_guard.py
exec uv run python -m oczy.experiments.consolidation_distillation
