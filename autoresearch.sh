#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/eval_guard.py c9ad5e74c999..HEAD
exec env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m oczy.experiments.consolidation_distillation --stage stage_1_transfer --seeds 2
