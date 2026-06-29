#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m oczy.experiments.correction_competence_v2 --driver real --seeds 1 --levels 1
