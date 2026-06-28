#!/usr/bin/env bash
# Autoresearch benchmark entrypoint for "orchestrate the remaining research lanes".
#
# Runs lanes/orchestrator.py which imports lanes/lane_01.py ... lanes/lane_07.py,
# each measuring one research lane's primary metric. Emits:
#   METRIC lanes_with_signal=<count>     (primary)
#   METRIC lane_NN_<name>=<value>         (one per lane, may be nan)
#
# Exits 0 on success (always — partial failures at the lane level surface as
# nan secondaries, not as a non-zero exit). Exits non-zero only if the
# orchestrator itself fails to start.

set -u

cd "$(dirname "$0")/.."

# Run from the repo root so `from lanes...` and `from src.oczy...` imports work.
exec uv run python -m lanes.orchestrator