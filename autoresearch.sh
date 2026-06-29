#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m oczy.experiments.bounded_growth.bounded_growth_eval
