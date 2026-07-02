"""Pytest configuration for experiments tests.

The ``eval`` package lives at the repo root, outside the ``src/`` import
tree.  Insert the repo root into ``sys.path`` so that
``from eval.v2 import ...`` (pulled in transitively by
``oczy.experiments.minimal_loop_forgetting``) resolves in test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
