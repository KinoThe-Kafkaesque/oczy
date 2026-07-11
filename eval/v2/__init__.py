"""eval/v2 — Frozen, versioned evaluation data for the Oczy curriculum.

Current data version: v2.2 (2026-07-11 protocol repair — Stage 1 is probe-only,
Stage 3 probes are episode-interleaved, Stage 4 consolidation precedes its
post-test, and the default split is category-stratified; see
experiments_logs/2026-07-11_eval_v2_2_protocol_repair.md). Existing episode
and probe text remains unchanged.

This package contains the canonical stage JSONs, scoring, and validation code.
The MANIFEST.json file holds SHA-256 hashes of every asset; `verify_manifest()`
recomputes them and raises `EvalIntegrityError` on mismatch.  Set
``EVAL_CHANGE_APPROVED=1`` to bypass the check (for deliberate, human-approved
eval changes).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class EvalIntegrityError(RuntimeError):
    """Raised when a protected eval file does not match its MANIFEST hash."""


_DATA_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _DATA_DIR / "MANIFEST.json"


def get_data_dir() -> Path:
    """Return the directory containing the frozen stage JSON files."""
    return _DATA_DIR


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    """Load and return the MANIFEST.json dict."""
    with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_manifest() -> None:
    """Recompute file hashes against MANIFEST.json.

    Raises `EvalIntegrityError` if any file's hash differs from the manifest,
    UNLESS ``EVAL_CHANGE_APPROVED=1`` is set in the environment.
    """
    if os.environ.get("EVAL_CHANGE_APPROVED") == "1":
        return
    manifest = _load_manifest()
    files = manifest.get("files", {})
    for relpath, expected_hash in files.items():
        fpath = _DATA_DIR / relpath
        if not fpath.exists():
            raise EvalIntegrityError(
                f"Manifest entry {relpath!r} not found at {fpath}"
            )
        actual = _sha256(fpath)
        if actual != expected_hash:
            raise EvalIntegrityError(
                f"Hash mismatch for {relpath!r}: "
                f"expected {expected_hash}, got {actual}"
            )


def recompute_manifest() -> dict:
    """Compute a fresh MANIFEST.json dict from current file contents.

    Used by `scripts/bump_eval_version.py` to regenerate the manifest after
    deliberate, human-approved eval changes.
    """
    files = {}
    for fpath in sorted(_DATA_DIR.iterdir()):
        if fpath.name == "MANIFEST.json" or fpath.name.startswith("."):
            continue
        if fpath.is_file():
            files[fpath.name] = _sha256(fpath)
    return {"version": "v2.2", "files": files}


def write_manifest() -> None:
    """Overwrite MANIFEST.json with a freshly computed hash set."""
    manifest = recompute_manifest()
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
