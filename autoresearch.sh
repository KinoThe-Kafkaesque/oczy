#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m oczy.experiments.conversation_world_model --driver real
