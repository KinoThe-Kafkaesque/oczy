#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv run python autoresearch_metabolism.py
