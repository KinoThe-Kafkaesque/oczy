"""Pytest tests for infrastructure/kaggle/runtime_manifest.py and execution_report.py.

Covers: strict JSON, exact keys/types, safe paths, sorted/unique files/roles,
model/no-model branches, component hashes, deterministic self-hash, tampering,
streaming observation, exact file set, report sentinel/direct loading, and exact
mismatch reporting.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_KAGGLE_DIR = str(Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle")
if _KAGGLE_DIR not in sys.path:
    sys.path.insert(0, _KAGGLE_DIR)

from runtime_manifest import (  # type: ignore[import-not-found]
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    RuntimeManifestError,
    _validate_artifact_path,
    compare_runtime_manifests,
    compute_component_sha256,
    compute_manifest_sha256,
    observe_runtime_manifest,
    strict_canonical_json,
    strict_json_loads,
    validate_runtime_manifest,
)
from execution_report import (  # type: ignore[import-not-found]
    EXECUTION_REPORT_SCHEMA_VERSION,
    EXECUTION_REPORT_SENTINEL_PREFIX,
    SentinelError,
    _extract_kaggle_log_sentinel,
    _extract_sentinel_report,
    load_execution_report,
    validate_execution_report_runtime,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_model_manifest(**overrides: Any) -> dict[str, Any]:
    """Return a valid no-model manifest dict (no self-hash)."""
    base: dict[str, Any] = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "python_version": "3.12.0",
        "packages": {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        },
        "model": {
            "logical_model_id": None,
            "resolved_model_convention": "none",
            "artifact_files": [],
            "model_weights_sha256": None,
            "model_config_sha256": None,
            "tokenizer_sha256": None,
            "chat_template_sha256": None,
        },
        "greedy_generation": None,
    }
    base.update(overrides)
    return base


def _valid_no_model(**overrides: Any) -> dict[str, Any]:
    """Return a fully valid no-model manifest (self-hash attached)."""
    m = _no_model_manifest(**overrides)
    m["manifest_sha256"] = compute_manifest_sha256(m)
    return m


def _model_artifacts() -> list[dict[str, Any]]:
    """Return sorted artifact files for a model-bearing manifest."""
    return sorted([
        {
            "path": "chat_template.jinja",
            "size_bytes": 200,
            "sha256": "b" * 64,
            "roles": ["chat_template"],
        },
        {
            "path": "config.json",
            "size_bytes": 100,
            "sha256": "a" * 64,
            "roles": ["config"],
        },
        {
            "path": "model.safetensors",
            "size_bytes": 300,
            "sha256": "c" * 64,
            "roles": ["weights"],
        },
        {
            "path": "tokenizer.json",
            "size_bytes": 400,
            "sha256": "d" * 64,
            "roles": ["tokenizer"],
        },
    ], key=lambda f: f["path"])


def _model_block(**overrides: Any) -> dict[str, Any]:
    """Return a valid model block with computed component hashes."""
    artifacts = _model_artifacts()
    block: dict[str, Any] = {
        "logical_model_id": "test/model",
        "resolved_model_convention": "transformers-pretrained-directory",
        "artifact_files": artifacts,
    }
    block["model_weights_sha256"] = compute_component_sha256(
        [f for f in artifacts if "weights" in f["roles"]]
    )
    block["model_config_sha256"] = compute_component_sha256(
        [f for f in artifacts if "config" in f["roles"]]
    )
    block["tokenizer_sha256"] = compute_component_sha256(
        [f for f in artifacts if "tokenizer" in f["roles"]]
    )
    block["chat_template_sha256"] = compute_component_sha256(
        [f for f in artifacts if "chat_template" in f["roles"]]
    )
    block.update(overrides)
    return block


def _default_greedy(**overrides: Any) -> dict[str, Any]:
    """Return default valid greedy generation config."""
    cfg: dict[str, Any] = {
        "max_new_tokens": 16,
        "min_new_tokens": 0,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "eos_token_ids": [2],
        "pad_token_id": 2,
        "stop_strings": [],
    }
    cfg.update(overrides)
    return cfg


def _valid_model(**overrides: Any) -> dict[str, Any]:
    """Return a fully valid model-bearing manifest (self-hash attached)."""
    m = _no_model_manifest(
        model=_model_block(),
        greedy_generation=_default_greedy(),
    )
    m.update(overrides)
    m["manifest_sha256"] = compute_manifest_sha256(m)
    return m


def _write_file(path: Path, content: bytes) -> str:
    """Write *content* to *path* and return its hex SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


# ======================================================================
# strict_json_loads
# ======================================================================


class TestStrictJsonLoads:
    """Strict JSON parsing: duplicate keys, NaN/Inf, non-object rejection."""

    def test_rejects_duplicate_keys(self) -> None:
        with pytest.raises(RuntimeManifestError, match="Duplicate JSON key"):
            strict_json_loads('{"a": 1, "a": 2}')

    def test_rejects_nan(self) -> None:
        with pytest.raises(RuntimeManifestError, match="NaN"):
            strict_json_loads('{"val": NaN}')

    def test_rejects_infinity(self) -> None:
        with pytest.raises(RuntimeManifestError, match="Infinity"):
            strict_json_loads('{"val": Infinity}')

    def test_rejects_negative_infinity(self) -> None:
        with pytest.raises(RuntimeManifestError, match="Infinity"):
            strict_json_loads('{"val": -Infinity}')

    def test_rejects_non_object(self) -> None:
        with pytest.raises(RuntimeManifestError, match="Expected JSON object"):
            strict_json_loads("[1, 2, 3]")

    def test_accepts_valid_object(self) -> None:
        result = strict_json_loads('{"a": 1, "b": "hello"}')
        assert result == {"a": 1, "b": "hello"}

    def test_accepts_bytes_input(self) -> None:
        result = strict_json_loads(b'{"a": 1}')
        assert result == {"a": 1}

    def test_accepts_nested_object(self) -> None:
        result = strict_json_loads('{"a": {"b": [1, 2]}, "c": null}')
        assert result == {"a": {"b": [1, 2]}, "c": None}


# ======================================================================
# strict_canonical_json
# ======================================================================


class TestStrictCanonicalJson:
    """Canonical JSON serialization: deterministic order, NaN rejection."""

    def test_sorts_keys(self) -> None:
        raw = strict_canonical_json({"z": 1, "a": 2, "m": 3})
        assert raw == b'{"a":2,"m":3,"z":1}'

    def test_deterministic_across_key_order(self) -> None:
        a = strict_canonical_json({"b": 1, "a": 2})
        b = strict_canonical_json({"a": 2, "b": 1})
        assert a == b

    def test_compact_separators(self) -> None:
        raw = strict_canonical_json({"a": 1, "b": 2})
        assert raw == b'{"a":1,"b":2}'

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            strict_canonical_json({"val": float("nan")})

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValueError):
            strict_canonical_json({"val": float("inf")})

    def test_ensure_ascii_false_allows_unicode(self) -> None:
        raw = strict_canonical_json({"greeting": "\u00e9"})
        assert "\u00e9".encode("utf-8") in raw

    def test_nested_stable_ordering(self) -> None:
        a = strict_canonical_json({"z": {"c": 3, "a": 1}, "x": 2})
        b = strict_canonical_json({"x": 2, "z": {"a": 1, "c": 3}})
        assert a == b

    def test_empty_object(self) -> None:
        assert strict_canonical_json({}) == b"{}"

    def test_empty_list(self) -> None:
        assert strict_canonical_json([]) == b"[]"


# ======================================================================
# compute_manifest_sha256
# ======================================================================


class TestComputeManifestSha256:
    """Self-hash computation: strips manifest_sha256 field."""

    def test_strips_self_hash(self) -> None:
        m = _valid_no_model()
        h = compute_manifest_sha256(m)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_same_input(self) -> None:
        m1 = _valid_no_model()
        m2 = _valid_no_model()
        assert compute_manifest_sha256(m1) == compute_manifest_sha256(m2)

    def test_different_payload_different_hash(self) -> None:
        m1 = _valid_no_model()
        m2 = _valid_no_model(python_version="3.13.0")
        assert compute_manifest_sha256(m1) != compute_manifest_sha256(m2)

    def test_manifest_sha256_field_value_does_not_affect_hash(self) -> None:
        m = _valid_no_model()
        h1 = compute_manifest_sha256(m)
        m["manifest_sha256"] = "f" * 64
        h2 = compute_manifest_sha256(m)
        assert h1 == h2

    def test_absent_manifest_sha256_okay(self) -> None:
        m = _valid_no_model()
        del m["manifest_sha256"]
        h = compute_manifest_sha256(m)
        assert len(h) == 64

    def test_with_model_manifest(self) -> None:
        m = _valid_model()
        h = compute_manifest_sha256(m)
        assert len(h) == 64


# ======================================================================
# compute_component_sha256
# ======================================================================


class TestComputeComponentSha256:
    """Component hash: projects to {path, size_bytes, sha256}, ignores roles."""

    def test_projects_correct_keys(self) -> None:
        files = [
            {"path": "a.txt", "size_bytes": 10, "sha256": "a" * 64, "roles": ["config"]},
            {"path": "b.bin", "size_bytes": 20, "sha256": "b" * 64, "roles": ["weights"]},
        ]
        h = compute_component_sha256(files)
        assert len(h) == 64
        # Same files with different roles -> same hash (roles excluded).
        files2 = [
            {"path": "a.txt", "size_bytes": 10, "sha256": "a" * 64, "roles": ["other"]},
            {"path": "b.bin", "size_bytes": 20, "sha256": "b" * 64, "roles": ["chat_template"]},
        ]
        assert h == compute_component_sha256(files2)

    def test_deterministic(self) -> None:
        files = [
            {"path": "x", "size_bytes": 1, "sha256": "c" * 64, "roles": ["weights"]},
        ]
        assert compute_component_sha256(files) == compute_component_sha256(files)

    def test_different_paths_produce_different_hash(self) -> None:
        h1 = compute_component_sha256(
            [{"path": "a", "size_bytes": 1, "sha256": "c" * 64, "roles": ["weights"]}]
        )
        h2 = compute_component_sha256(
            [{"path": "b", "size_bytes": 1, "sha256": "c" * 64, "roles": ["weights"]}]
        )
        assert h1 != h2

    def test_different_sizes_produce_different_hash(self) -> None:
        h1 = compute_component_sha256(
            [{"path": "a", "size_bytes": 1, "sha256": "c" * 64, "roles": ["weights"]}]
        )
        h2 = compute_component_sha256(
            [{"path": "a", "size_bytes": 2, "sha256": "c" * 64, "roles": ["weights"]}]
        )
        assert h1 != h2

    def test_empty_files_list(self) -> None:
        h = compute_component_sha256([])
        assert len(h) == 64


# ======================================================================
# Artifact path validation
# ======================================================================


class TestArtifactPathValidation:
    """Artifact path safety rules."""

    def test_rejects_non_string(self) -> None:
        with pytest.raises(RuntimeManifestError, match="must be a string"):
            _validate_artifact_path(42)

    def test_rejects_empty(self) -> None:
        with pytest.raises(RuntimeManifestError, match="must not be empty"):
            _validate_artifact_path("")

    def test_rejects_leading_whitespace(self) -> None:
        with pytest.raises(RuntimeManifestError, match="leading/trailing whitespace"):
            _validate_artifact_path(" foo.txt")

    def test_rejects_trailing_whitespace(self) -> None:
        with pytest.raises(RuntimeManifestError, match="leading/trailing whitespace"):
            _validate_artifact_path("foo.txt ")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(RuntimeManifestError, match="contains backslash"):
            _validate_artifact_path("foo\\bar.txt")

    def test_rejects_nul(self) -> None:
        with pytest.raises(RuntimeManifestError, match="NUL byte"):
            _validate_artifact_path("foo\x00.txt")

    def test_rejects_absolute(self) -> None:
        with pytest.raises(RuntimeManifestError, match="is absolute"):
            _validate_artifact_path("/etc/passwd")

    def test_rejects_dot_component(self) -> None:
        with pytest.raises(RuntimeManifestError, match="reserved component"):
            _validate_artifact_path("./foo.txt")

    def test_rejects_dotdot_component(self) -> None:
        with pytest.raises(RuntimeManifestError, match="reserved component"):
            _validate_artifact_path("../foo.txt")

    def test_rejects_dotdot_midpath(self) -> None:
        with pytest.raises(RuntimeManifestError, match="reserved component"):
            _validate_artifact_path("foo/../bar.txt")

    def test_accepts_valid_relative_path(self) -> None:
        _validate_artifact_path("foo/bar/baz.txt")  # does not raise

    def test_accepts_single_component(self) -> None:
        _validate_artifact_path("model.safetensors")  # does not raise

    def test_accepts_nested_subdir(self) -> None:
        _validate_artifact_path("subdir/other/model.bin")  # does not raise


# ======================================================================
# validate_runtime_manifest — top-level and general
# ======================================================================


class TestValidateRuntimeManifestTopLevel:
    """validate_runtime_manifest top-level key set, schema version, fields."""

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(RuntimeManifestError, match="must be a JSON object"):
            validate_runtime_manifest([])  # type: ignore[arg-type]

    def test_rejects_extra_top_level_key(self) -> None:
        m = _valid_no_model()
        m["extra_field"] = True  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            validate_runtime_manifest(m)

    def test_rejects_missing_top_level_key(self) -> None:
        m = _valid_no_model()
        del m["python_version"]
        with pytest.raises(RuntimeManifestError, match="missing key"):
            validate_runtime_manifest(m)

    def test_rejects_wrong_schema_version(self) -> None:
        m = _valid_no_model(schema_version="wrong/v1")
        with pytest.raises(RuntimeManifestError, match="schema_version"):
            validate_runtime_manifest(m)

    def test_rejects_non_string_python_version(self) -> None:
        m = _valid_no_model(python_version=3.12)  # type: ignore[arg-type]
        with pytest.raises(RuntimeManifestError, match="python_version"):
            validate_runtime_manifest(m)

    def test_rejects_empty_python_version(self) -> None:
        m = _valid_no_model(python_version="")
        with pytest.raises(RuntimeManifestError, match="python_version"):
            validate_runtime_manifest(m)

    def test_accepts_valid_no_model(self) -> None:
        m = _valid_no_model()
        result = validate_runtime_manifest(m)
        assert result is m

    def test_accepts_valid_model(self) -> None:
        m = _valid_model()
        result = validate_runtime_manifest(m)
        assert result is m


# ======================================================================
# validate_runtime_manifest — packages
# ======================================================================


class TestValidateRuntimeManifestPackages:
    """Package block validation."""

    def test_rejects_non_dict_packages(self) -> None:
        m = _valid_no_model(packages="bad")  # type: ignore[arg-type]
        with pytest.raises(RuntimeManifestError, match="`packages` must be an object"):
            validate_runtime_manifest(m)

    def test_rejects_extra_package_key(self) -> None:
        m = _no_model_manifest()
        m["packages"]["numpy"] = "1.26.0"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            validate_runtime_manifest(m)

    def test_rejects_missing_package_key(self) -> None:
        m = _no_model_manifest()
        del m["packages"]["torch"]  # type: ignore[arg-type]
        with pytest.raises(RuntimeManifestError, match="missing key"):
            validate_runtime_manifest(m)

    def test_rejects_non_string_package_version(self) -> None:
        m = _no_model_manifest()
        m["packages"]["torch"] = 2.6  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be a string"):
            validate_runtime_manifest(m)


# ======================================================================
# validate_runtime_manifest — model block
# ======================================================================


class TestValidateRuntimeManifestModel:
    """Model block validation."""

    def test_rejects_non_dict_model(self) -> None:
        m = _valid_no_model(model="bad")  # type: ignore[arg-type]
        with pytest.raises(RuntimeManifestError, match="`model` must be an object"):
            validate_runtime_manifest(m)

    def test_rejects_extra_model_key(self) -> None:
        m = _valid_model()
        m["model"]["extra"] = True  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            validate_runtime_manifest(m)

    def test_rejects_missing_model_key(self) -> None:
        m = _valid_model()
        del m["model"]["logical_model_id"]  # type: ignore[arg-type]
        with pytest.raises(RuntimeManifestError, match="missing key"):
            validate_runtime_manifest(m)

    def test_rejects_invalid_convention(self) -> None:
        m = _valid_model()
        m["model"]["resolved_model_convention"] = "bad-convention"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="resolved_model_convention.*unknown"):
            validate_runtime_manifest(m)


# ======================================================================
# validate_runtime_manifest — artifact files
# ======================================================================


class TestValidateRuntimeManifestArtifactFiles:
    """Artifact file list validation."""

    def test_rejects_non_list(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"] = "bad"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="`artifact_files` must be a list"):
            validate_runtime_manifest(m)

    def test_rejects_non_dict_entry(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"] = ["not_a_dict"]  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="artifact_files\\[0\\]` must be an object"):
            validate_runtime_manifest(m)

    def test_rejects_missing_required_key(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"] = [  # type: ignore[index]
            {"path": "a.bin", "size_bytes": 1, "sha256": "a" * 64}  # missing roles
        ]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="missing required key"):
            validate_runtime_manifest(m)

    def test_rejects_extra_key(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["extra"] = True  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            validate_runtime_manifest(m)

    def test_rejects_invalid_sha256_length(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["sha256"] = "too_short"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="64 lowercase hex"):
            validate_runtime_manifest(m)

    def test_rejects_uppercase_sha256(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["sha256"] = "A" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="64 lowercase hex"):
            validate_runtime_manifest(m)

    def test_rejects_bool_for_size_bytes(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["size_bytes"] = True  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be an int"):
            validate_runtime_manifest(m)

    def test_rejects_zero_size_bytes(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["size_bytes"] = 0  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be positive"):
            validate_runtime_manifest(m)

    def test_rejects_non_list_roles(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["roles"] = "config"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be a nonempty list"):
            validate_runtime_manifest(m)

    def test_rejects_empty_roles(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["roles"] = []  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be a nonempty list"):
            validate_runtime_manifest(m)

    def test_rejects_invalid_role(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["roles"] = ["not_a_role"]  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="roles.*invalid"):
            validate_runtime_manifest(m)

    def test_rejects_duplicate_roles_in_same_file(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["roles"] = ["config", "config"]  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="roles.*duplicate"):
            validate_runtime_manifest(m)

    def test_rejects_non_string_role(self) -> None:
        m = _valid_model()
        m["model"]["artifact_files"][0]["roles"] = [42]  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="contains non-string"):
            validate_runtime_manifest(m)

    def test_rejects_unsorted_paths(self) -> None:
        m = _valid_model()
        # Reverse the list to make it unsorted.
        m["model"]["artifact_files"] = list(reversed(m["model"]["artifact_files"]))  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="must be sorted by `path`"):
            validate_runtime_manifest(m)

    def test_rejects_duplicate_paths(self) -> None:
        m = _valid_model()
        af: list[dict[str, Any]] = list(m["model"]["artifact_files"])  # type: ignore[arg-type]
        af.append(dict(af[0]))
        m["model"]["artifact_files"] = af  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="paths must be unique"):
            validate_runtime_manifest(m)


# ======================================================================
# validate_runtime_manifest — component hashes
# ======================================================================


class TestValidateRuntimeManifestComponentHashes:
    """Component hash integrity checks."""

    def test_rejects_bad_component_hash(self) -> None:
        m = _valid_model()
        m["model"]["model_weights_sha256"] = "f" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="mismatch"):
            validate_runtime_manifest(m)

    def test_accepts_correct_component_hashes(self) -> None:
        m = _valid_model()
        validate_runtime_manifest(m)  # does not raise

    def test_component_hash_uses_only_specific_role_files(self) -> None:
        """Files without the matching role are not included in component hash."""
        artifacts = _model_artifacts()
        block = {
            "logical_model_id": "test/model",
            "resolved_model_convention": "transformers-pretrained-directory",
            "artifact_files": artifacts,
        }
        block["model_weights_sha256"] = compute_component_sha256(
            [f for f in artifacts if "weights" in f["roles"]]
        )
        block["model_config_sha256"] = compute_component_sha256(
            [f for f in artifacts if "config" in f["roles"]]
        )
        block["tokenizer_sha256"] = compute_component_sha256(
            [f for f in artifacts if "tokenizer" in f["roles"]]
        )
        block["chat_template_sha256"] = compute_component_sha256(
            [f for f in artifacts if "chat_template" in f["roles"]]
        )
        m = _valid_model(model=block)
        validate_runtime_manifest(m)  # does not raise

    def test_rejects_missing_required_roles(self) -> None:
        """Model-bearing manifest must cover weights/config/tokenizer/chat_template."""
        # Create artifacts without tokenizer role.
        artifacts = [f for f in _model_artifacts() if "tokenizer" not in f["roles"]]
        block = {
            "logical_model_id": "test/model",
            "resolved_model_convention": "transformers-pretrained-directory",
            "artifact_files": artifacts,
        }
        block["model_weights_sha256"] = compute_component_sha256(
            [f for f in artifacts if "weights" in f["roles"]]
        )
        block["model_config_sha256"] = compute_component_sha256(
            [f for f in artifacts if "config" in f["roles"]]
        )
        block["tokenizer_sha256"] = None
        block["chat_template_sha256"] = compute_component_sha256(
            [f for f in artifacts if "chat_template" in f["roles"]]
        )
        m = _valid_model(model=block)
        with pytest.raises(RuntimeManifestError, match="missing required roles"):
            validate_runtime_manifest(m)


# ======================================================================
# validate_runtime_manifest — no-model branch
# ======================================================================


class TestValidateRuntimeManifestNoModel:
    """No-model branch constraints."""

    def test_accepts_valid_no_model(self) -> None:
        m = _valid_no_model()
        validate_runtime_manifest(m)  # does not raise

    def test_rejects_no_model_with_logical_id(self) -> None:
        m = _valid_no_model()
        m["model"]["logical_model_id"] = "some/model"  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*logical_model_id.*null"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_artifacts(self) -> None:
        m = _valid_no_model()
        m["model"]["artifact_files"] = [  # type: ignore[index]
            {"path": "x.bin", "size_bytes": 1, "sha256": "a" * 64, "roles": ["weights"]}
        ]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*artifact_files.*empty"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_component_hash(self) -> None:
        m = _valid_no_model()
        m["model"]["model_weights_sha256"] = "a" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*model_weights_sha256.*null"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_config_hash(self) -> None:
        m = _valid_no_model()
        m["model"]["model_config_sha256"] = "a" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*model_config_sha256.*null"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_tokenizer_hash(self) -> None:
        m = _valid_no_model()
        m["model"]["tokenizer_sha256"] = "a" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*tokenizer_sha256.*null"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_chat_template_hash(self) -> None:
        m = _valid_no_model()
        m["model"]["chat_template_sha256"] = "a" * 64  # type: ignore[index]
        m["manifest_sha256"] = compute_manifest_sha256(m)
        with pytest.raises(RuntimeManifestError, match="no-model.*chat_template_sha256.*null"):
            validate_runtime_manifest(m)

    def test_rejects_no_model_with_greedy_generation(self) -> None:
        m = _valid_no_model(greedy_generation=_default_greedy())
        with pytest.raises(RuntimeManifestError, match="no-model.*greedy_generation.*null"):
            validate_runtime_manifest(m)


# ======================================================================
# validate_runtime_manifest — greedy generation
# ======================================================================


class TestValidateRuntimeManifestGreedyGeneration:
    """Greedy generation fail-closed constraints."""

    def test_rejects_do_sample_true(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(do_sample=True))
        with pytest.raises(RuntimeManifestError, match="do_sample.*false"):
            validate_runtime_manifest(m)

    def test_rejects_num_beams_not_one(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(num_beams=4))
        with pytest.raises(RuntimeManifestError, match="num_beams.*1"):
            validate_runtime_manifest(m)

    def test_rejects_temperature_not_zero(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(temperature=0.7))
        with pytest.raises(RuntimeManifestError, match="temperature.*0"):
            validate_runtime_manifest(m)

    def test_rejects_zero_max_new_tokens(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(max_new_tokens=0))
        with pytest.raises(RuntimeManifestError, match="max_new_tokens.*positive"):
            validate_runtime_manifest(m)

    def test_rejects_non_int_eos_ids(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(eos_token_ids=["a"]))
        with pytest.raises(RuntimeManifestError, match="eos_token_ids.*integers"):
            validate_runtime_manifest(m)

    def test_rejects_duplicate_eos_ids(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(eos_token_ids=[2, 2]))
        with pytest.raises(RuntimeManifestError, match="eos_token_ids.*unique"):
            validate_runtime_manifest(m)

    def test_rejects_extra_greedy_key(self) -> None:
        cfg = _default_greedy()
        cfg["extra"] = True  # type: ignore[index]
        m = _valid_model(greedy_generation=cfg)
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            validate_runtime_manifest(m)

    def test_rejects_missing_greedy_key(self) -> None:
        cfg = _default_greedy()
        del cfg["use_cache"]  # type: ignore[arg-type]
        m = _valid_model(greedy_generation=cfg)
        with pytest.raises(RuntimeManifestError, match="missing key"):
            validate_runtime_manifest(m)

    def test_rejects_non_list_stop_strings(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(stop_strings="bad"))
        with pytest.raises(RuntimeManifestError, match="stop_strings.*list"):
            validate_runtime_manifest(m)

    def test_rejects_non_string_in_stop_strings(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(stop_strings=[42]))
        with pytest.raises(RuntimeManifestError, match="stop_strings.*strings"):
            validate_runtime_manifest(m)

    def test_model_bearing_requires_greedy(self) -> None:
        m = _valid_model(greedy_generation=None)
        with pytest.raises(RuntimeManifestError, match="model-bearing branch.*greedy_generation.*null"):
            validate_runtime_manifest(m)

    def test_rejects_bool_in_numeric_field(self) -> None:
        m = _valid_model(greedy_generation=_default_greedy(max_new_tokens=True))
        with pytest.raises(RuntimeManifestError):
            validate_runtime_manifest(m)

    def test_accepts_valid_greedy(self) -> None:
        m = _valid_model()
        validate_runtime_manifest(m)  # does not raise


# ======================================================================
# validate_runtime_manifest — self-hash
# ======================================================================


class TestValidateRuntimeManifestSelfHash:
    """Self-hash integrity and tampering detection."""

    def test_rejects_non_string_manifest_sha256(self) -> None:
        m = _no_model_manifest(manifest_sha256=42)
        with pytest.raises(RuntimeManifestError, match="64 lowercase hex"):
            validate_runtime_manifest(m)

    def test_rejects_wrong_length_sha256(self) -> None:
        m = _no_model_manifest(manifest_sha256="abc")
        with pytest.raises(RuntimeManifestError, match="64 lowercase hex"):
            validate_runtime_manifest(m)

    def test_rejects_uppercase_sha256(self) -> None:
        m = _no_model_manifest(manifest_sha256="A" * 64)
        with pytest.raises(RuntimeManifestError, match="64 lowercase hex"):
            validate_runtime_manifest(m)

    def test_accepts_correct_self_hash(self) -> None:
        m = _valid_no_model()
        validate_runtime_manifest(m)  # does not raise

    def test_tampering_model_block_detected(self) -> None:
        m = _valid_model()
        m["model"]["logical_model_id"] = "tampered/model"  # type: ignore[index]
        with pytest.raises(RuntimeManifestError, match="manifest_sha256.*mismatch"):
            validate_runtime_manifest(m)

    def test_tampering_artifact_detected(self) -> None:
        """Tampering artifact data invalidates self-hash."""
        m = _valid_model()
        # Change size in artifact AND recompute component hash to avoid
        # tripping component-hash validation before self-hash check.
        m["model"]["artifact_files"][0]["size_bytes"] = 99999  # type: ignore[index]
        af = m["model"]["artifact_files"]
        m["model"]["chat_template_sha256"] = compute_component_sha256(  # type: ignore[index]
            [f for f in af if "chat_template" in f["roles"]]
        )
        # Now component-hash integrity passes but self-hash is stale.
        with pytest.raises(RuntimeManifestError, match="manifest_sha256.*mismatch"):
            validate_runtime_manifest(m)


# ======================================================================
# compare_runtime_manifests
# ======================================================================


class TestCompareRuntimeManifests:
    """Exact manifest comparison with structured mismatch paths."""

    def test_equal_no_model(self) -> None:
        m1 = _valid_no_model()
        m2 = _valid_no_model()
        assert compare_runtime_manifests(m1, m2) == []

    def test_equal_model(self) -> None:
        m1 = _valid_model()
        m2 = _valid_model()
        assert compare_runtime_manifests(m1, m2) == []

    def test_value_mismatch_top_level(self) -> None:
        m1 = _valid_no_model()
        m2 = _valid_no_model(python_version="3.13.0")
        mismatches = compare_runtime_manifests(m1, m2)
        assert len(mismatches) >= 1
        assert any("python_version" in m for m in mismatches)

    def test_value_mismatch_deep(self) -> None:
        m1 = _valid_model()
        m2 = _valid_model(model=_model_block(logical_model_id="test/different-model"))
        mismatches = compare_runtime_manifests(m1, m2)
        assert len(mismatches) > 0
        assert any("logical_model_id" in m for m in mismatches)

    def test_type_mismatch_null_vs_dict(self) -> None:
        m1 = _valid_no_model()  # greedy_generation = None
        m2 = _valid_model()   # greedy_generation = dict
        mismatches = compare_runtime_manifests(m1, m2)
        assert len(mismatches) > 0
        # The initial mismatch will be model or greedy_generation.
        assert any("NoneType" in m or "model" in m for m in mismatches)

    def test_both_validated_before_comparison(self) -> None:
        """If one manifest fails validation, comparison raises immediately."""
        m1 = _valid_no_model()
        m1["extra"] = True  # type: ignore[index]
        m2 = _valid_no_model()
        with pytest.raises(RuntimeManifestError, match="unexpected key"):
            compare_runtime_manifests(m1, m2)


# ======================================================================
# observe_runtime_manifest — no-model
# ======================================================================


class TestObserveRuntimeManifestNoModel:
    """Observation of no-model manifests."""

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_builds_valid_no_model(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        m = observe_runtime_manifest()
        assert m["schema_version"] == RUNTIME_MANIFEST_SCHEMA_VERSION
        assert m["model"]["logical_model_id"] is None
        assert m["model"]["resolved_model_convention"] == "none"
        assert m["model"]["artifact_files"] == []
        assert m["greedy_generation"] is None
        validate_runtime_manifest(m)

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_model_root_when_no_model(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="no-model.*model_root.*None"):
            observe_runtime_manifest(model_root=Path("/tmp/fake"))

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_logical_id_when_no_model(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="no-model.*logical_model_id.*None"):
            observe_runtime_manifest(logical_model_id="test/model")

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_generation_when_no_model(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="no-model.*generation_config.*None"):
            observe_runtime_manifest(generation_config=_default_greedy())


# ======================================================================
# observe_runtime_manifest — model-bearing HF directory
# ======================================================================


class TestObserveRuntimeManifestHFDirectory:
    """Observation of transformers-pretrained-directory model roots."""

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_observes_hf_directory(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        sha_config = _write_file(model_dir / "config.json", b'{"arch": "test"}')
        sha_weights = _write_file(model_dir / "model.safetensors", b"\x00" * 100)
        sha_tokenizer = _write_file(model_dir / "tokenizer.json", b"{}")
        sha_template = _write_file(model_dir / "chat_template.jinja", b"{{ messages }}")

        m = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/hf-model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        assert m["model"]["logical_model_id"] == "test/hf-model"
        assert m["model"]["resolved_model_convention"] == "transformers-pretrained-directory"

        af = m["model"]["artifact_files"]
        assert len(af) == 4
        paths = [f["path"] for f in af]
        assert paths == sorted(paths)
        assert "config.json" in paths
        assert "model.safetensors" in paths
        assert "tokenizer.json" in paths
        assert "chat_template.jinja" in paths

        for f in af:
            if f["path"] == "config.json":
                assert f["sha256"] == sha_config
            elif f["path"] == "model.safetensors":
                assert f["sha256"] == sha_weights
            elif f["path"] == "tokenizer.json":
                assert f["sha256"] == sha_tokenizer
            elif f["path"] == "chat_template.jinja":
                assert f["sha256"] == sha_template

        validate_runtime_manifest(m)

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_observes_subdirectory_files(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        subdir = model_dir / "sub"
        _write_file(subdir / "vocab.json", b"vocab")
        _write_file(model_dir / "config.json", b"config")
        _write_file(model_dir / "model.safetensors", b"weights")
        _write_file(model_dir / "tokenizer.json", b"tok")
        _write_file(model_dir / "chat_template.jinja", b"template")

        m = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/hf-model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        paths = [f["path"] for f in m["model"]["artifact_files"]]
        assert "sub/vocab.json" in paths

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_symlink_in_directory(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"w")
        (model_dir / "tokenizer.json").write_text("t")
        (model_dir / "chat_template.jinja").write_text("c")
        (model_dir / "real_extra.txt").write_text("real")
        os.symlink(model_dir / "real_extra.txt", model_dir / "link.txt")

        with pytest.raises(RuntimeManifestError, match="symlinks not allowed"):
            observe_runtime_manifest(
                model_root=model_dir,
                logical_model_id="test/hf-model",
                resolved_model_convention="transformers-pretrained-directory",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_empty_directory(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(RuntimeManifestError, match="no regular files found"):
            observe_runtime_manifest(
                model_root=model_dir,
                logical_model_id="test/hf-model",
                resolved_model_convention="transformers-pretrained-directory",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_missing_root(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="does not exist"):
            observe_runtime_manifest(
                model_root=Path("/tmp/nonexistent_xyz_123"),
                logical_model_id="test/hf-model",
                resolved_model_convention="transformers-pretrained-directory",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_component_hashes_computed(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        _write_file(model_dir / "config.json", b"config")
        _write_file(model_dir / "model.safetensors", b"weights")
        _write_file(model_dir / "tokenizer.json", b"tok")
        _write_file(model_dir / "chat_template.jinja", b"template")

        m = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/hf-model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        for key in ("model_weights_sha256", "model_config_sha256",
                     "tokenizer_sha256", "chat_template_sha256"):
            val = m["model"][key]
            assert isinstance(val, str)
            assert len(val) == 64
            assert all(c in "0123456789abcdef" for c in val)


# ======================================================================
# observe_runtime_manifest — GGUF file
# ======================================================================


class TestObserveRuntimeManifestGGUF:
    """Observation of llama-cpp-gguf-file model roots."""

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_observes_gguf_file(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        gguf_path = tmp_path / "model.gguf"
        sha = _write_file(gguf_path, b"\x00" * 1024)

        m = observe_runtime_manifest(
            model_root=gguf_path,
            logical_model_id="test/gguf-model",
            resolved_model_convention="llama-cpp-gguf-file",
            generation_config=_default_greedy(),
        )
        assert m["model"]["resolved_model_convention"] == "llama-cpp-gguf-file"
        af = m["model"]["artifact_files"]
        assert len(af) == 1
        assert af[0]["path"] == "model.gguf"
        assert af[0]["sha256"] == sha
        assert set(af[0]["roles"]) >= {"weights", "config", "tokenizer", "chat_template"}

        validate_runtime_manifest(m)

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_directory_for_gguf(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(RuntimeManifestError, match="expected regular file"):
            observe_runtime_manifest(
                model_root=model_dir,
                logical_model_id="test/gguf-model",
                resolved_model_convention="llama-cpp-gguf-file",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_symlink_for_gguf(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        real_file = tmp_path / "real.gguf"
        real_file.write_bytes(b"data")
        link_path = tmp_path / "link.gguf"
        os.symlink(real_file, link_path)
        with pytest.raises(RuntimeManifestError, match="symlinks not allowed"):
            observe_runtime_manifest(
                model_root=link_path,
                logical_model_id="test/gguf-model",
                resolved_model_convention="llama-cpp-gguf-file",
                generation_config=_default_greedy(),
            )


# ======================================================================
# observe_runtime_manifest — invalid convention / missing requirements
# ======================================================================


class TestObserveRuntimeManifestInvalid:
    """Invalid/missing arguments for observation."""

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_unknown_convention(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="unknown convention"):
            observe_runtime_manifest(
                resolved_model_convention="bad-convention",  # type: ignore[arg-type]
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_model_bearing_without_model_root(self, mock_pkgs: Any, _mock_py: Any) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        with pytest.raises(RuntimeManifestError, match="model-bearing.*model_root.*required"):
            observe_runtime_manifest(
                logical_model_id="test/model",
                resolved_model_convention="transformers-pretrained-directory",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_model_bearing_without_logical_id(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_file(model_dir / "config.json", b"{}")
        _write_file(model_dir / "model.safetensors", b"w")
        _write_file(model_dir / "tokenizer.json", b"t")
        _write_file(model_dir / "chat_template.jinja", b"c")

        with pytest.raises(RuntimeManifestError, match="model-bearing.*logical_model_id.*required"):
            observe_runtime_manifest(
                model_root=model_dir,
                resolved_model_convention="transformers-pretrained-directory",
                generation_config=_default_greedy(),
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_rejects_model_bearing_without_generation(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_file(model_dir / "config.json", b"{}")
        _write_file(model_dir / "model.safetensors", b"w")
        _write_file(model_dir / "tokenizer.json", b"t")
        _write_file(model_dir / "chat_template.jinja", b"c")

        with pytest.raises(RuntimeManifestError, match="model-bearing.*generation_config.*required"):
            observe_runtime_manifest(
                model_root=model_dir,
                logical_model_id="test/model",
                resolved_model_convention="transformers-pretrained-directory",
            )

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    def test_missing_package_raises(self, _mock_py: Any) -> None:
        with patch("runtime_manifest._try_package_version", return_value=None):
            with pytest.raises(RuntimeManifestError, match="not found"):
                observe_runtime_manifest()


# ======================================================================
# observe_runtime_manifest — deterministic canonical output
# ======================================================================


class TestObserveRuntimeManifestDeterminism:
    """Observed manifests are deterministic and canonical."""

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_repeated_observation_same_hash(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        _write_file(model_dir / "config.json", b"config")
        _write_file(model_dir / "model.safetensors", b"weights")
        _write_file(model_dir / "tokenizer.json", b"tok")
        _write_file(model_dir / "chat_template.jinja", b"template")

        m1 = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        m2 = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        assert m1["manifest_sha256"] == m2["manifest_sha256"]
        assert strict_canonical_json(m1) == strict_canonical_json(m2)

    @patch("runtime_manifest.platform.python_version", return_value="3.12.0")
    @patch("runtime_manifest._observe_packages")
    def test_different_content_different_hash(self, mock_pkgs: Any, _mock_py: Any, tmp_path: Path) -> None:
        mock_pkgs.return_value = {
            "torch": "2.6.0",
            "transformers": "4.55.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
        }
        model_dir = tmp_path / "model"
        _write_file(model_dir / "config.json", b"config_v1")
        _write_file(model_dir / "model.safetensors", b"weights")
        _write_file(model_dir / "tokenizer.json", b"tok")
        _write_file(model_dir / "chat_template.jinja", b"template")

        m1 = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )

        _write_file(model_dir / "config.json", b"config_v2_DIFFERENT")

        m2 = observe_runtime_manifest(
            model_root=model_dir,
            logical_model_id="test/model",
            resolved_model_convention="transformers-pretrained-directory",
            generation_config=_default_greedy(),
        )
        assert m1["manifest_sha256"] != m2["manifest_sha256"]


# ======================================================================
# _extract_sentinel_report
# ======================================================================


class TestExtractSentinelReport:
    """Sentinel extraction from stdout text."""

    def _make_sentinel_line(self, report: dict[str, Any]) -> str:
        return EXECUTION_REPORT_SENTINEL_PREFIX + json.dumps(report, separators=(",", ":"))

    def test_extracts_valid_sentinel(self) -> None:
        report = {"schema_version": "v1", "value": 42}
        line = self._make_sentinel_line(report)
        result = _extract_sentinel_report(line)
        assert result == report

    def test_raises_on_missing_sentinel(self) -> None:
        with pytest.raises(SentinelError, match="no OCZY_EXECUTION_REPORT_JSON"):
            _extract_sentinel_report("some output without sentinel")

    def test_raises_on_multiple_sentinels(self) -> None:
        report = {"a": 1}
        line = self._make_sentinel_line(report)
        text = line + "\n" + line
        with pytest.raises(SentinelError, match="multiple"):
            _extract_sentinel_report(text)

    def test_raises_on_unparseable_json(self) -> None:
        text = EXECUTION_REPORT_SENTINEL_PREFIX + "not valid json{{{"
        with pytest.raises(SentinelError, match="unparseable"):
            _extract_sentinel_report(text)

    def test_raises_on_non_object(self) -> None:
        text = EXECUTION_REPORT_SENTINEL_PREFIX + "[1, 2, 3]"
        with pytest.raises(SentinelError, match="not an object"):
            _extract_sentinel_report(text)

    def test_ignores_other_lines(self) -> None:
        report = {"result": "ok"}
        line = self._make_sentinel_line(report)
        text = "INFO: starting\n" + line + "\nINFO: done\n"
        result = _extract_sentinel_report(text)
        assert result == report

    def test_sentinel_must_be_column_zero(self) -> None:
        report = {"a": 1}
        line = self._make_sentinel_line(report)
        text = "  " + line  # indented
        with pytest.raises(SentinelError, match="no OCZY_EXECUTION_REPORT_JSON"):
            _extract_sentinel_report(text)


# ======================================================================
# _extract_kaggle_log_sentinel
# ======================================================================


class TestExtractKaggleLogSentinel:
    """Kaggle kernel log JSON stream extraction."""

    def _make_sentinel_line(self, report: dict[str, Any]) -> str:
        return EXECUTION_REPORT_SENTINEL_PREFIX + json.dumps(report, separators=(",", ":"))

    def _make_kaggle_log(self, stdout_parts: list[str]) -> str:
        entries = [
            {"stream_name": "stdout", "data": part}
            for part in stdout_parts
        ]
        return json.dumps(entries)

    def test_extracts_from_kaggle_log(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "score": 0.95}
        line = self._make_sentinel_line(report)
        log_path = tmp_path / "kernel.log"
        log_path.write_text(self._make_kaggle_log([line]))
        result = _extract_kaggle_log_sentinel(log_path)
        assert result == report

    def test_concatenates_stdout_fragments(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "data": "long"}
        json_text = json.dumps(report, separators=(",", ":"))
        prefix = EXECUTION_REPORT_SENTINEL_PREFIX
        split_point = len(prefix) + 5
        full_line = prefix + json_text
        part1 = full_line[:split_point]
        part2 = full_line[split_point:]
        log_path = tmp_path / "kernel.log"
        log_path.write_text(self._make_kaggle_log([part1, part2]))
        result = _extract_kaggle_log_sentinel(log_path)
        assert result == report

    def test_raises_on_missing_log_file(self, tmp_path: Path) -> None:
        with pytest.raises(SentinelError, match="not found"):
            _extract_kaggle_log_sentinel(tmp_path / "nonexistent.log")

    def test_raises_on_non_json_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "kernel.log"
        log_path.write_text("not valid json at all")
        with pytest.raises(SentinelError, match="not valid JSON"):
            _extract_kaggle_log_sentinel(log_path)

    def test_raises_on_non_array_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "kernel.log"
        log_path.write_text('{"not": "an array"}')
        with pytest.raises(SentinelError, match="not a JSON array"):
            _extract_kaggle_log_sentinel(log_path)

    def test_skips_non_stdout_streams(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "result": "ok"}
        line = self._make_sentinel_line(report)
        entries = [
            {"stream_name": "stderr", "data": "some error"},
            {"stream_name": "stdout", "data": line},
        ]
        log_path = tmp_path / "kernel.log"
        log_path.write_text(json.dumps(entries))
        result = _extract_kaggle_log_sentinel(log_path)
        assert result == report

    def test_raises_on_no_sentinel_in_stdout(self, tmp_path: Path) -> None:
        log_path = tmp_path / "kernel.log"
        log_path.write_text(self._make_kaggle_log(["just some output, no sentinel"]))
        with pytest.raises(SentinelError, match="no OCZY_EXECUTION_REPORT_JSON"):
            _extract_kaggle_log_sentinel(log_path)


# ======================================================================
# load_execution_report
# ======================================================================


class TestLoadExecutionReport:
    """Provider-dispatched report loading."""

    def _make_sentinel_line(self, report: dict[str, Any]) -> str:
        return EXECUTION_REPORT_SENTINEL_PREFIX + json.dumps(report, separators=(",", ":"))

    def test_loads_direct_kaggle_report(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "name": "test"}
        (tmp_path / "execution_report.json").write_text(json.dumps(report))
        result = load_execution_report("kaggle", tmp_path)
        assert result is not None
        assert result.get("name") == "test"

    def test_loads_kaggle_sentinel_from_log(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "result": "ok"}
        line = self._make_sentinel_line(report)
        log_path = tmp_path / "kernel.log"
        log_path.write_text(json.dumps([{"stream_name": "stdout", "data": line}]))
        result = load_execution_report("kaggle", tmp_path)
        assert result is not None
        assert result.get("_source") == "kaggle_log_sentinel"
        assert result.get("result") == "ok"

    def test_loads_kaggle_provenance_fallback(self, tmp_path: Path) -> None:
        provenance = {"schema_version": "v1", "model": "test"}
        (tmp_path / "remote_run_provenance.json").write_text(json.dumps(provenance))
        result = load_execution_report("kaggle", tmp_path)
        assert result is not None
        assert result.get("_source") == "provenance_fallback"

    def test_returns_none_when_nothing_found(self, tmp_path: Path) -> None:
        result = load_execution_report("kaggle", tmp_path)
        assert result is None

    def test_loads_direct_colab_report(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "name": "colab_test"}
        (tmp_path / "execution_report.json").write_text(json.dumps(report))
        result = load_execution_report("colab", tmp_path)
        assert result is not None
        assert result.get("name") == "colab_test"

    def test_loads_colab_sentinel_from_stdout(self, tmp_path: Path) -> None:
        report = {"schema_version": "v1", "result": "colab_ok"}
        line = self._make_sentinel_line(report)
        (tmp_path / "stdout.log").write_text(line)
        result = load_execution_report("colab", tmp_path)
        assert result is not None
        assert result.get("_source") == "stdout_sentinel"

    def test_loads_colab_result_fallback(self, tmp_path: Path) -> None:
        result_json = {"ok": True, "message": "done"}
        (tmp_path / "result.json").write_text(json.dumps(result_json))
        result = load_execution_report("colab", tmp_path)
        assert result is not None
        assert result.get("_source") == "result_fallback"

    def test_returns_none_colab_empty(self, tmp_path: Path) -> None:
        result = load_execution_report("colab", tmp_path)
        assert result is None


# ======================================================================
# validate_execution_report_runtime
# ======================================================================


class TestValidateExecutionReportRuntime:
    """Runtime manifest validation on loaded execution reports."""

    def test_rejects_fallback_source(self) -> None:
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "_source": "provenance_fallback",
        }
        manifest = _valid_no_model()
        with pytest.raises(SentinelError, match="diagnostic fallback"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_result_fallback(self) -> None:
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "_source": "result_fallback",
        }
        manifest = _valid_no_model()
        with pytest.raises(SentinelError, match="diagnostic fallback"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_wrong_schema(self) -> None:
        manifest = _valid_no_model()
        report = {
            "schema_version": "oczy/execution-report/v1",
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": _valid_no_model(),
        }
        with pytest.raises(SentinelError, match="expected report schema"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_missing_expected_hash(self) -> None:
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "observed_runtime_manifest": _valid_no_model(),
        }
        manifest = _valid_no_model()
        with pytest.raises(SentinelError, match="expected_runtime_manifest_sha256"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_wrong_expected_hash(self) -> None:
        manifest = _valid_no_model()
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": "f" * 64,
            "observed_runtime_manifest": _valid_no_model(),
        }
        with pytest.raises(RuntimeManifestError, match="expected_runtime_manifest_sha256 mismatch"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_missing_observed_manifest(self) -> None:
        manifest = _valid_no_model()
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
        }
        with pytest.raises(SentinelError, match="observed_runtime_manifest"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_non_dict_observed(self) -> None:
        manifest = _valid_no_model()
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": "not_an_object",
        }
        with pytest.raises(SentinelError, match="observed_runtime_manifest"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_invalid_observed_manifest(self) -> None:
        manifest = _valid_no_model()
        bad_observed = _valid_no_model()
        bad_observed["extra"] = True  # type: ignore[index]
        bad_observed["manifest_sha256"] = compute_manifest_sha256(bad_observed)
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": bad_observed,
        }
        with pytest.raises(SentinelError, match="observed_runtime_manifest fails validation"):
            validate_execution_report_runtime(report, manifest)

    def test_rejects_manifest_mismatch(self) -> None:
        manifest = _valid_no_model()
        different_observed = _valid_no_model(python_version="3.13.0")
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": different_observed,
        }
        with pytest.raises(RuntimeManifestError, match="runtime manifest mismatch"):
            validate_execution_report_runtime(report, manifest)

    def test_accepts_matching_manifest(self) -> None:
        manifest = _valid_no_model()
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": _valid_no_model(),
        }
        validate_execution_report_runtime(report, manifest)  # does not raise

    def test_mismatch_message_includes_detail(self) -> None:
        manifest = _valid_no_model()
        different = _valid_no_model(python_version="3.99.0")
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": different,
        }
        with pytest.raises(RuntimeManifestError) as exc_info:
            validate_execution_report_runtime(report, manifest)
        error_msg = str(exc_info.value)
        assert "python_version" in error_msg or "mismatch" in error_msg.lower()

    def test_accepts_matching_model_manifest(self) -> None:
        manifest = _valid_model()
        report = {
            "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
            "expected_runtime_manifest_sha256": compute_manifest_sha256(manifest),
            "observed_runtime_manifest": _valid_model(),
        }
        validate_execution_report_runtime(report, manifest)  # does not raise
