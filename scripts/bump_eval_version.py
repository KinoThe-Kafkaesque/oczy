"""Recompute and write eval/v2/MANIFEST.json hashes.

Run this after a deliberate, human-approved change to the frozen eval
assets (stage JSONs or ``eval/v2/__init__.py``).  It regenerates
``MANIFEST.json`` from the current file contents so that
``verify_manifest()`` passes again.

Usage:
    uv run python scripts/bump_eval_version.py
"""

import sys
from pathlib import Path

# Ensure the repo root is importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.v2 import EvalIntegrityError, recompute_manifest, write_manifest


def main() -> int:
    """Recompute hashes and write MANIFEST.json.

    Returns 0 on success, 1 on error (message printed to stderr).
    """
    try:
        manifest = recompute_manifest()
        write_manifest()
    except EvalIntegrityError as exc:
        print(f"eval integrity error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"could not write MANIFEST.json: {exc}", file=sys.stderr)
        return 1

    version = manifest.get("version", "unknown")
    n_files = len(manifest.get("files", {}))
    print(
        f"MANIFEST.json updated: version={version}, files hashed={n_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
