#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/eval_guard.py c9ad5e74c999..HEAD
exec uv run python -m oczy.experiments.consolidation_distillation
