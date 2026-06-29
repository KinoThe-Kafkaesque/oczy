#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m oczy.experiments.layer_l_probe --driver real
