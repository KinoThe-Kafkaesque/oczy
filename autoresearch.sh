#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m oczy.experiments.scope_selectivity_stressor --driver real
