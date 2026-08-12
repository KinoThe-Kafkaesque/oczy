"""Tests for the eval/v2 MANIFEST.json integrity system.

These tests defend the concrete contracts of the frozen-eval freeze (S0.1):

* ``verify_manifest()`` succeeds on the real, unmodified bundled files
* ``verify_manifest()`` raises ``EvalIntegrityError`` on any tampered asset
* ``EVAL_CHANGE_APPROVED=1`` bypasses the check (for human-approved changes)
* ``recompute_manifest()`` produces a structurally valid manifest whose every
  entry points at a file that actually exists in the data dir
* ``scripts/bump_eval_version.py`` is idempotent — running it twice yields a
  byte-identical ``MANIFEST.json``
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import eval.v2 as eval_v2
from eval.v2 import (
    EvalIntegrityError,
    get_data_dir,
    recompute_manifest,
    verify_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _bump_module():
    """Load ``scripts/bump_eval_version.py`` as an importable module by path."""
    script = _REPO_ROOT / "scripts" / "bump_eval_version.py"
    assert script.exists(), f"bump_eval_version.py missing at {script}"
    spec = importlib.util.spec_from_file_location("bump_eval_version", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_eval_v2(tmp_path: Path) -> Path:
    """Copy the bundled eval/v2 tree (manifest + assets) into ``tmp_path``."""
    dest = tmp_path / "v2"
    shutil.copytree(get_data_dir(), dest)
    return dest


def test_manifest_verifies_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """``verify_manifest()`` raises nothing on the real bundled files."""
    monkeypatch.delenv("EVAL_CHANGE_APPROVED", raising=False)
    # Must not raise.
    verify_manifest()


def test_manifest_tampering_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered stage JSON must trigger ``EvalIntegrityError``."""
    monkeypatch.delenv("EVAL_CHANGE_APPROVED", raising=False)

    data_dir = _copy_eval_v2(tmp_path)
    manifest_path = data_dir / "MANIFEST.json"

    # Tamper with a stage JSON so its on-disk hash no longer matches the
    # manifest's recorded hash (which was copied verbatim from the bundle).
    target = data_dir / "stage_0_grounding.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Mutate a value deep enough to change the serialized bytes.
    if isinstance(payload, dict):
        payload["_tampered"] = True
    else:
        payload = {"_tampered": True}
    target.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(eval_v2, "_DATA_DIR", data_dir)
    monkeypatch.setattr(eval_v2, "_MANIFEST_PATH", manifest_path)

    with pytest.raises(EvalIntegrityError):
        verify_manifest()


def test_eval_change_approved_bypasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``EVAL_CHANGE_APPROVED=1`` a tampered copy must NOT raise."""
    monkeypatch.setenv("EVAL_CHANGE_APPROVED", "1")

    data_dir = _copy_eval_v2(tmp_path)
    manifest_path = data_dir / "MANIFEST.json"

    target = data_dir / "stage_1_transfer.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_tampered"] = True
    else:
        payload = {"_tampered": True}
    target.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(eval_v2, "_DATA_DIR", data_dir)
    monkeypatch.setattr(eval_v2, "_MANIFEST_PATH", manifest_path)

    # Must NOT raise — the env bypass short-circuits before any hashing.
    verify_manifest()


def test_recompute_manifest_produces_valid() -> None:
    """``recompute_manifest()`` returns a well-formed manifest of real files."""
    manifest = recompute_manifest()
    assert manifest["version"] == "v2.2"
    files = manifest["files"]
    assert files, "manifest should hash at least one file"
    data_dir = get_data_dir()
    for relpath in files:
        assert (data_dir / relpath).exists(), f"missing manifest entry: {relpath}"


def test_bump_script_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running ``bump_eval_version.py`` twice yields a byte-identical manifest."""
    monkeypatch.delenv("EVAL_CHANGE_APPROVED", raising=False)

    data_dir = _copy_eval_v2(tmp_path)
    manifest_path = data_dir / "MANIFEST.json"
    monkeypatch.setattr(eval_v2, "_DATA_DIR", data_dir)
    monkeypatch.setattr(eval_v2, "_MANIFEST_PATH", manifest_path)

    bump = _bump_module()
    assert bump.main() == 0
    first = manifest_path.read_bytes()
    assert bump.main() == 0
    second = manifest_path.read_bytes()

    assert first == second, "bump_eval_version.py is not idempotent"
    # And the regenerated manifest is itself structurally valid.
    parsed = json.loads(second)
    assert parsed["version"] == "v2.2"
    assert parsed["files"]
