"""Versioned replay inventory and exact comparison contracts."""

import json
from collections import Counter
from pathlib import Path

from scripts.capability_validation_contract import canonical, sha, write_json

REPO = Path(__file__).resolve().parents[2]
CAP = "experiments_logs/artifacts/2026-09-13_capability_validation_v1"
PILOT = "experiments_logs/artifacts/2026-09-12_r23_5_pilot_v1"
DIVERSITY = "experiments/scoped-diversity-dev-v3"
REPLAYS = {
    "capability_reference": (f"{CAP}/run/reference/results.json", 240),
    "capability_restore": (f"{CAP}/run/restore/results.json", 336),
    "capability_actions": (f"{CAP}/run/actions/results.json", 20),
    "pilot_restore_soft": (f"{PILOT}/execution/restore_soft/results.json", 96),
    "pilot_restore_text": (f"{PILOT}/execution/restore_text/results.json", 16),
}


def read(path):
    return json.loads(Path(path).read_text())


def compare_rows(actual, expected):
    """Count complete row objects, including prompts and action outcomes."""
    def encode(rows):
        return Counter(canonical(row) for row in rows)
    return bool(expected) and encode(actual) == encode(expected)


def verify(root):
    manifest = read(root / "MANIFEST.json")
    digest = sha(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}))
    if manifest["manifest_sha256"] != digest:
        raise ValueError("Regression manifest changed")
    for path, expected in manifest["files"].items():
        if sha((REPO / path).read_bytes()) != expected:
            raise ValueError(f"Regression input changed: {path}")
    return manifest


__all__ = ["REPO", "CAP", "PILOT", "DIVERSITY", "REPLAYS", "read", "verify", "compare_rows", "canonical", "sha", "write_json"]
