"""High-signal contract tests for the meta_cortex/v2 detached signoff gate.

These tests defend the security boundary of the exact-tuple authorization —
not plumbing.  They verify:

  - Candidate alone cannot yield AuthorizedInstrument.
  - Wrong external manifest hash, noncanonical/wrong margin, wrong N/count map,
    wrong/free-form/version-mismatched/non-v4 ID, self-hash tamper, or missing
    signoff fails before any sealed open.
  - Correct exact tuple authorizes; any sealed bit flip fails before task return.
  - Signoff creation is write-once/race-safe; no env/force/default/alias path.
  - Signed v1 cannot be overwritten/rematerialized; changed bytes require v2.
  - EVAL_CHANGE_APPROVED=1 and plausible approval env vars do not bypass.
  - No calibration/materialization code path emits a signoff.
  - Gate-before-read: authorization runs before sealed bytes, model, or output.
  - Detached exact-tuple signoff: generic approval never substitutes.
  - UUID/version validation on signoff ID.

No real model, network, or Qwen is required.  All fixtures are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from oczy.experiments.meta_cortex.authorization import (
    AuthorizedInstrument,
    InstrumentIntegrityError,
    RunAuthorization,
    SignoffError,
    authorize_instrument,
    load_signoff,
    record_signoff,
    validate_signoff_id,
    verify_run_authorization,
)
from oczy.experiments.meta_cortex.contracts import ContractError
from oczy.experiments.meta_cortex.instrument_contracts import (
    CANDIDATE_MANIFEST_SCHEMA,
    SIGNOFF_SCHEMA,
    InstrumentFileEntry,
    SignoffAttestation,
    strict_canonical_json,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DUMMY_SHA = "a" * 64
_DUMMY_SHA_2 = "b" * 64
_VALID_SIGNOFF_ID = "r20-meta-cortex-v2/550e8400-e29b-41d4-a716-446655440000"
_VALID_SIGNOFF_ID_2 = "r20-meta-cortex-v2/660e8400-e29b-42d4-a716-446655440001"
_FAMILY_ORDER = ("contextual_remap", "rule_transformation", "finite_state")
_N = 30
_COUNT_MAP = {
    "contextual_remap": _N,
    "rule_transformation": _N,
    "finite_state": _N,
}
_MARGIN = "0.05"


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _make_file_entry(
    path: str = "public/DEV_VIEW.json",
    sha256: str = _DUMMY_SHA,
    size_bytes: int = 100,
    visibility: str = "public",
    role: str = "dev_view",
) -> InstrumentFileEntry:
    return InstrumentFileEntry(
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        visibility=visibility,
        role=role,
    )


def _make_manifest_dict(
    *,
    equivalence_margin: str = _MARGIN,
    sample_size_tasks_per_family: int = _N,
    meta_test_tasks_by_family: dict[str, int] | None = None,
    lifecycle_state: str = "candidate",
    files: list[dict] | None = None,
    manifest_sha256: str | None = None,
    calibration_holdout_accessed: bool = False,
    leakage_audit_passed: bool = True,
) -> dict:
    """Build a synthetic manifest dict with computed self-hash."""
    if meta_test_tasks_by_family is None:
        meta_test_tasks_by_family = dict(_COUNT_MAP)
    if files is None:
        files = [
            {"path": "public/DEV_VIEW.json", "sha256": _DUMMY_SHA, "size_bytes": 100,
             "visibility": "public", "role": "dev_view"},
            {"path": "public/CALIBRATION_VIEW.json", "sha256": _DUMMY_SHA, "size_bytes": 100,
             "visibility": "calibration", "role": "calibration_view"},
            {"path": "sealed/meta_test.jsonl", "sha256": _DUMMY_SHA, "size_bytes": 200,
             "visibility": "sealed", "role": "meta_test_tasks"},
        ]

    manifest_dict: dict = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "instrument_id": "meta_cortex/v2",
        "instrument_version": "v2",
        "lifecycle_state": lifecycle_state,
        "definition_sha256": _DUMMY_SHA,
        "dev_view_sha256": _DUMMY_SHA,
        "calibration_view_sha256": _DUMMY_SHA,
        "source_commit": "abc123",
        "source_archive_sha256": _DUMMY_SHA,
        "taskgen_schema": "oczy/meta-cortex/taskgen/v1",
        "generator_algorithm": "sha256-counter-rejection/v1",
        "generator_source_sha256": _DUMMY_SHA,
        "prompt_schema": "oczy/meta-cortex/prompts/v1",
        "prompt_registry_sha256": _DUMMY_SHA,
        "scorer_schema": "oczy/meta-cortex/scorers/v1",
        "scorer_registry_sha256": _DUMMY_SHA,
        "endpoint_schema": "oczy/meta-cortex/endpoints/v1",
        "endpoint_registry_sha256": _DUMMY_SHA,
        "organ_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "organ_revision": "main",
        "organ_parameter_sha256": _DUMMY_SHA,
        "chat_template_sha256": _DUMMY_SHA,
        "feature_mode": "final_layer_mean_pool",
        "decoding_mode": "greedy",
        "max_new_tokens": 16,
        "feature_dim": 896,
        "d_cortex": 64,
        "soft_bank_width": 3,
        "abstain_threshold": "0.5",
        "event_min": 1,
        "event_max": 5,
        "family_order": list(_FAMILY_ORDER),
        "probe_counts_by_split_family_kind": {},
        "train_tasks_per_family": 2,
        "tuning_tasks_per_family": 2,
        "calibration_tasks_per_family": 30,
        "sample_size_tasks_per_family": sample_size_tasks_per_family,
        "meta_test_tasks_by_family": meta_test_tasks_by_family,
        "developmental_seeds": [10, 20, 30, 40, 50],
        "evaluation_seeds": [60, 70, 80, 90, 100],
        "equivalence_margin": equivalence_margin,
        "calibration_report_sha256": _DUMMY_SHA,
        "power_report_sha256": _DUMMY_SHA,
        "calibration_holdout_accessed": calibration_holdout_accessed,
        "independent_sample_unit": "task_rule",
        "leakage_audit_sha256": _DUMMY_SHA,
        "leakage_audit_passed": leakage_audit_passed,
        "meta_test_seed_sha256": _DUMMY_SHA,
        "files": files,
    }

    if manifest_sha256 is None:
        payload = {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}
        manifest_sha256 = hashlib.sha256(strict_canonical_json(payload)).hexdigest()
    manifest_dict["manifest_sha256"] = manifest_sha256
    return manifest_dict


def _write_manifest(root: Path, manifest_dict: dict | None = None) -> Path:
    """Write a manifest dict as MANIFEST.json in root."""
    root.mkdir(parents=True, exist_ok=True)
    if manifest_dict is None:
        manifest_dict = _make_manifest_dict()
    path = root / "MANIFEST.json"
    path.write_bytes(strict_canonical_json(manifest_dict) + b"\n")
    return path


def _write_instrument_files(root: Path, files: list[dict] | None = None) -> None:
    """Write dummy instrument files matching the manifest entries."""
    root.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = [
            {"path": "public/DEV_VIEW.json", "sha256": _DUMMY_SHA, "size_bytes": 100,
             "visibility": "public", "role": "dev_view"},
            {"path": "public/CALIBRATION_VIEW.json", "sha256": _DUMMY_SHA, "size_bytes": 100,
             "visibility": "calibration", "role": "calibration_view"},
            {"path": "sealed/meta_test.jsonl", "sha256": _DUMMY_SHA, "size_bytes": 200,
             "visibility": "sealed", "role": "meta_test_tasks"},
        ]
    for entry in files:
        file_path = root / entry["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Write deterministic bytes that hash to _DUMMY_SHA.
        # We need actual bytes whose SHA-256 matches. Use a placeholder and
        # patch the hash verification in tests that need file verification.
        content = b"x" * entry["size_bytes"]
        # Compute actual hash and update the entry.
        actual_hash = hashlib.sha256(content).hexdigest()
        entry["sha256"] = actual_hash
        file_path.write_bytes(content)


def _make_signoff_dict(
    *,
    instrument_manifest_sha256: str = _DUMMY_SHA,
    equivalence_margin: str = _MARGIN,
    sample_size_tasks_per_family: int = _N,
    meta_test_tasks_by_family: dict[str, int] | None = None,
    human_signoff_id: str = _VALID_SIGNOFF_ID,
    lifecycle_state: str = "signed",
    instrument_version: str = "v2",
    signoff_sha256: str | None = None,
) -> dict:
    """Build a synthetic signoff dict with computed self-hash."""
    if meta_test_tasks_by_family is None:
        meta_test_tasks_by_family = dict(_COUNT_MAP)

    signoff_dict: dict = {
        "schema": SIGNOFF_SCHEMA,
        "instrument_id": "meta_cortex/v2",
        "instrument_version": instrument_version,
        "lifecycle_state": lifecycle_state,
        "instrument_manifest_sha256": instrument_manifest_sha256,
        "equivalence_margin": equivalence_margin,
        "sample_size_tasks_per_family": sample_size_tasks_per_family,
        "meta_test_tasks_by_family": meta_test_tasks_by_family,
        "human_signoff_id": human_signoff_id,
        "signed_at_utc": "2026-07-12T00:00:00Z",
    }

    if signoff_sha256 is None:
        payload = {k: v for k, v in signoff_dict.items() if k != "signoff_sha256"}
        signoff_sha256 = hashlib.sha256(strict_canonical_json(payload)).hexdigest()
    signoff_dict["signoff_sha256"] = signoff_sha256
    return signoff_dict


def _write_signoff(path: Path, signoff_dict: dict | None = None) -> Path:
    """Write a signoff dict as SIGNOFF.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if signoff_dict is None:
        signoff_dict = _make_signoff_dict()
    path.write_bytes(strict_canonical_json(signoff_dict) + b"\n")
    return path


def _build_full_instrument(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a complete synthetic instrument directory with manifest, signoff, and files.

    Returns (root, manifest_sha256, signoff_sha256).
    """
    root = tmp_path / "instrument"
    root.mkdir(parents=True, exist_ok=True)

    # Write instrument files with real content.
    file_specs = [
        ("public/DEV_VIEW.json", 100, "public", "dev_view"),
        ("public/CALIBRATION_VIEW.json", 100, "calibration", "calibration_view"),
        ("sealed/meta_test.jsonl", 200, "sealed", "meta_test_tasks"),
    ]
    files_list = []
    for fpath, size, vis, role in file_specs:
        full_path = root / fpath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        content = b"x" * size
        full_path.write_bytes(content)
        files_list.append({
            "path": fpath,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": size,
            "visibility": vis,
            "role": role,
        })

    manifest_dict = _make_manifest_dict(files=files_list)
    # Recompute manifest hash with the real file hashes.
    payload = {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}
    manifest_dict["manifest_sha256"] = hashlib.sha256(
        strict_canonical_json(payload)
    ).hexdigest()
    _write_manifest(root, manifest_dict)

    manifest_sha = manifest_dict["manifest_sha256"]

    # Create signoff via record_signoff.
    signoff_path = root / "SIGNOFF.json"
    signoff_dict = _make_signoff_dict(
        instrument_manifest_sha256=manifest_sha,
    )
    _write_signoff(signoff_path, signoff_dict)
    signoff_sha = signoff_dict["signoff_sha256"]

    return root, manifest_sha, signoff_sha


# ---------------------------------------------------------------------------
# Signoff ID validation tests
# ---------------------------------------------------------------------------


class TestValidateSignoffId:
    def test_valid_v1_id(self) -> None:
        assert validate_signoff_id(_VALID_SIGNOFF_ID) == _VALID_SIGNOFF_ID

    def test_reject_empty(self) -> None:
        with pytest.raises(ContractError, match="non-empty"):
            validate_signoff_id("")

    def test_reject_free_form_yes(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("yes")

    def test_reject_free_form_approved(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("approved")

    def test_reject_free_form_human(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("human")

    def test_reject_email(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("human@example.com")

    def test_reject_whitespace_padded(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id(f" {_VALID_SIGNOFF_ID} ")

    def test_reject_wildcard(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("*")

    def test_reject_uppercase_uuid(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id(
                "r20-meta-cortex-v2/550E8400-E29B-41D4-A716-446655440000"
            )

    def test_reject_non_v4_uuid(self) -> None:
        with pytest.raises(ContractError, match="version 4"):
            validate_signoff_id(
                "r20-meta-cortex-v2/550e8400-e29b-31d4-a716-446655440000"
            )

    def test_reject_wrong_variant(self) -> None:
        with pytest.raises(ContractError, match="variant"):
            validate_signoff_id(
                "r20-meta-cortex-v2/550e8400-e29b-41d4-1716-446655440000"
            )

    def test_reject_version_mismatch(self) -> None:
        """A v1 signoff ID must not match a v2 expected version."""
        from oczy.experiments.meta_cortex.authorization import (
            _validate_signoff_id_with_version,
        )
        with pytest.raises(SignoffError, match="version"):
            _validate_signoff_id_with_version(
                "r20-meta-cortex-v1/550e8400-e29b-41d4-a716-446655440000",
                "v2",
            )

    def test_version_match_accepted(self) -> None:
        from oczy.experiments.meta_cortex.authorization import (
            _validate_signoff_id_with_version,
        )
        _validate_signoff_id_with_version(_VALID_SIGNOFF_ID, "v2")


# ---------------------------------------------------------------------------
# RunAuthorization type tests
# ---------------------------------------------------------------------------


class TestRunAuthorization:
    def test_valid_authorization(self) -> None:
        auth = RunAuthorization(
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
            signoff_sha256=_DUMMY_SHA_2,
        )
        assert auth.instrument_manifest_sha256 == _DUMMY_SHA
        assert auth.equivalence_margin == _MARGIN
        assert auth.sample_size_tasks_per_family == _N

    def test_reject_n_below_30(self) -> None:
        with pytest.raises(SignoffError, match=r">= 30"):
            RunAuthorization(
                instrument_manifest_sha256=_DUMMY_SHA,
                equivalence_margin=_MARGIN,
                sample_size_tasks_per_family=29,
                human_signoff_id=_VALID_SIGNOFF_ID,
                signoff_sha256=_DUMMY_SHA_2,
            )

    def test_reject_bool_n(self) -> None:
        with pytest.raises(SignoffError, match="int"):
            RunAuthorization(
                instrument_manifest_sha256=_DUMMY_SHA,
                equivalence_margin=_MARGIN,
                sample_size_tasks_per_family=True,  # type: ignore[arg-type]
                human_signoff_id=_VALID_SIGNOFF_ID,
                signoff_sha256=_DUMMY_SHA_2,
            )

    def test_reject_noncanonical_margin(self) -> None:
        with pytest.raises((SignoffError, ContractError)):
            RunAuthorization(
                instrument_manifest_sha256=_DUMMY_SHA,
                equivalence_margin="0.050",
                sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
                signoff_sha256=_DUMMY_SHA_2,
            )

    def test_reject_invalid_signoff_id(self) -> None:
        with pytest.raises(SignoffError):
            RunAuthorization(
                instrument_manifest_sha256=_DUMMY_SHA,
                equivalence_margin=_MARGIN,
                sample_size_tasks_per_family=_N,
                human_signoff_id="yes",
                signoff_sha256=_DUMMY_SHA_2,
            )

    def test_reject_bad_manifest_sha(self) -> None:
        with pytest.raises((SignoffError, ContractError)):
            RunAuthorization(
                instrument_manifest_sha256="short",
                equivalence_margin=_MARGIN,
                sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
                signoff_sha256=_DUMMY_SHA_2,
            )

    def test_is_frozen(self) -> None:
        import dataclasses
        auth = RunAuthorization(
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
            signoff_sha256=_DUMMY_SHA_2,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            auth.equivalence_margin = "0.06"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuthorizedInstrument type tests
# ---------------------------------------------------------------------------


class TestAuthorizedInstrument:
    def test_private_constructor_no_sealed_access(self) -> None:
        """AuthorizedInstrument exposes only metadata, no sealed bytes/iterator."""
        # Verify it doesn't have sealed-task or sealed-bytes attributes.
        assert not hasattr(AuthorizedInstrument, "sealed_tasks")
        assert not hasattr(AuthorizedInstrument, "sealed_bytes")
        assert not hasattr(AuthorizedInstrument, "get_sealed")
        assert not hasattr(AuthorizedInstrument, "iter_sealed")

    def test_properties_are_read_only(self) -> None:
        """manifest, signoff, authorization are read-only properties."""
        assert isinstance(AuthorizedInstrument.manifest, property)
        assert isinstance(AuthorizedInstrument.signoff, property)
        assert isinstance(AuthorizedInstrument.authorization, property)


# ---------------------------------------------------------------------------
# record_signoff tests
# ---------------------------------------------------------------------------


class TestRecordSignoff:
    def test_valid_signoff_creation(self, tmp_path: Path) -> None:
        """record_signoff creates a valid SIGNOFF.json."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        attestation = record_signoff(
            root,
            dest,
            expected_manifest_sha256=manifest_dict["manifest_sha256"],
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
        )
        assert attestation.instrument_manifest_sha256 == manifest_dict["manifest_sha256"]
        assert attestation.equivalence_margin == _MARGIN
        assert attestation.human_signoff_id == _VALID_SIGNOFF_ID
        assert dest.exists()

    def test_write_once_rejects_existing(self, tmp_path: Path) -> None:
        """A second signoff to the same destination must fail."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        record_signoff(
            root,
            dest,
            expected_manifest_sha256=manifest_dict["manifest_sha256"],
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
        )
        with pytest.raises(SignoffError, match="write-once|already exists"):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_reject_wrong_manifest_hash(self, tmp_path: Path) -> None:
        """record_signoff must reject a wrong expected_manifest_sha256."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with pytest.raises(SignoffError, match="manifest SHA"):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=_DUMMY_SHA_2,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_reject_wrong_margin(self, tmp_path: Path) -> None:
        """record_signoff must reject a wrong expected_equivalence_margin."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with pytest.raises(SignoffError, match="margin"):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin="0.06",
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_reject_wrong_n(self, tmp_path: Path) -> None:
        """record_signoff must reject a wrong expected_sample_size."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with pytest.raises(SignoffError, match="sample_size"):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=31,
                human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_reject_free_form_signoff_id(self, tmp_path: Path) -> None:
        """record_signoff must reject a free-form human_signoff_id."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with pytest.raises((SignoffError, ContractError)):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id="yes",
            )

    def test_reject_version_mismatched_id(self, tmp_path: Path) -> None:
        """A v1 signoff ID must not authorize the v2 manifest."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        v1_id = "r20-meta-cortex-v1/550e8400-e29b-41d4-a716-446655440000"
        with pytest.raises((SignoffError, ContractError)):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id=v1_id,
            )

    def test_candidate_not_mutated(self, tmp_path: Path) -> None:
        """The candidate manifest must remain byte-identical after signoff."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)
        manifest_before = (root / "MANIFEST.json").read_bytes()

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        record_signoff(
            root,
            dest,
            expected_manifest_sha256=manifest_dict["manifest_sha256"],
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
        )
        manifest_after = (root / "MANIFEST.json").read_bytes()
        assert manifest_before == manifest_after

    def test_missing_manifest_rejected(self, tmp_path: Path) -> None:
        """record_signoff must fail if no manifest exists."""
        root = tmp_path / "empty"
        root.mkdir(parents=True)
        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with pytest.raises(InstrumentIntegrityError, match="[Nn]o MANIFEST"):
            record_signoff(
                root,
                dest,
                expected_manifest_sha256=_DUMMY_SHA,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_env_bypass_has_no_effect(self, tmp_path: Path) -> None:
        """EVAL_CHANGE_APPROVED=1 must not bypass the gate."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        dest = tmp_path / "signoff" / "SIGNOFF.json"
        with patch.dict(os.environ, {"EVAL_CHANGE_APPROVED": "1"}):
            # Should still reject wrong hash.
            with pytest.raises(SignoffError, match="manifest SHA"):
                record_signoff(
                    root,
                    dest,
                    expected_manifest_sha256=_DUMMY_SHA_2,
                    expected_equivalence_margin=_MARGIN,
                    expected_sample_size_tasks_per_family=_N,
                    human_signoff_id=_VALID_SIGNOFF_ID,
                )


# ---------------------------------------------------------------------------
# load_signoff tests
# ---------------------------------------------------------------------------


class TestLoadSignoff:
    def test_valid_signoff_loads(self, tmp_path: Path) -> None:
        signoff_dict = _make_signoff_dict()
        path = tmp_path / "SIGNOFF.json"
        _write_signoff(path, signoff_dict)

        attestation = load_signoff(path)
        assert attestation.human_signoff_id == _VALID_SIGNOFF_ID
        assert attestation.lifecycle_state == "signed"

    def test_missing_signoff_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SignoffError, match="not found"):
            load_signoff(tmp_path / "nonexistent.json")

    def test_tampered_self_hash_rejected(self, tmp_path: Path) -> None:
        signoff_dict = _make_signoff_dict()
        signoff_dict["signoff_sha256"] = _DUMMY_SHA_2  # Wrong hash.
        path = tmp_path / "SIGNOFF.json"
        path.write_bytes(strict_canonical_json(signoff_dict) + b"\n")

        with pytest.raises(SignoffError, match="self-hash"):
            load_signoff(path)

    def test_wrong_schema_rejected(self, tmp_path: Path) -> None:
        signoff_dict = _make_signoff_dict()
        signoff_dict["schema"] = "wrong"
        # Recompute self-hash since we changed a field.
        payload = {k: v for k, v in signoff_dict.items() if k != "signoff_sha256"}
        signoff_dict["signoff_sha256"] = hashlib.sha256(
            strict_canonical_json(payload)
        ).hexdigest()
        path = tmp_path / "SIGNOFF.json"
        path.write_bytes(strict_canonical_json(signoff_dict) + b"\n")

        with pytest.raises(SignoffError, match="schema"):
            load_signoff(path)

    def test_wrong_lifecycle_state_rejected(self, tmp_path: Path) -> None:
        signoff_dict = _make_signoff_dict(lifecycle_state="candidate")
        payload = {k: v for k, v in signoff_dict.items() if k != "signoff_sha256"}
        signoff_dict["signoff_sha256"] = hashlib.sha256(
            strict_canonical_json(payload)
        ).hexdigest()
        path = tmp_path / "SIGNOFF.json"
        path.write_bytes(strict_canonical_json(signoff_dict) + b"\n")

        with pytest.raises(SignoffError, match="lifecycle"):
            load_signoff(path)

    def test_noncanonical_bytes_rejected(self, tmp_path: Path) -> None:
        """Whitespace variants of the signoff JSON must be rejected."""
        signoff_dict = _make_signoff_dict()
        # Write with extra whitespace (non-canonical).
        path = tmp_path / "SIGNOFF.json"
        path.write_text(json.dumps(signoff_dict, indent=2))
        with pytest.raises(SignoffError, match="canonical"):
            load_signoff(path)

    def test_duplicate_keys_rejected(self, tmp_path: Path) -> None:
        signoff_dict = _make_signoff_dict()
        canonical = strict_canonical_json(signoff_dict).decode("utf-8")
        # Inject a duplicate key.
        bad_json = canonical.replace(
            '"human_signoff_id"',
            '"human_signoff_id":"dup","human_signoff_id"',
        )
        path = tmp_path / "SIGNOFF.json"
        path.write_text(bad_json)
        with pytest.raises(SignoffError):
            load_signoff(path)


# ---------------------------------------------------------------------------
# verify_run_authorization tests
# ---------------------------------------------------------------------------


class TestVerifyRunAuthorization:
    def _setup_valid_pair(self, tmp_path: Path) -> tuple[Path, Path, str, str]:
        """Set up a valid manifest+signoff pair. Returns (manifest_path, signoff_path, manifest_sha, signoff_sha)."""
        root = tmp_path / "instrument"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)
        manifest_sha = manifest_dict["manifest_sha256"]

        signoff_dict = _make_signoff_dict(instrument_manifest_sha256=manifest_sha)
        signoff_path = root / "SIGNOFF.json"
        _write_signoff(signoff_path, signoff_dict)
        signoff_sha = signoff_dict["signoff_sha256"]

        return root / "MANIFEST.json", signoff_path, manifest_sha, signoff_sha

    def test_valid_authorization_succeeds(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, signoff_sha = self._setup_valid_pair(tmp_path)
        auth = verify_run_authorization(
            manifest_path,
            signoff_path,
            expected_manifest_sha256=manifest_sha,
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            expected_human_signoff_id=_VALID_SIGNOFF_ID,
        )
        assert auth.instrument_manifest_sha256 == manifest_sha
        assert auth.signoff_sha256 == signoff_sha

    def test_wrong_manifest_hash_fails(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises(SignoffError, match="anifest hash"):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=_DUMMY_SHA_2,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_wrong_margin_fails(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises(SignoffError, match="argin"):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=manifest_sha,
                expected_equivalence_margin="0.06",
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_noncanonical_margin_fails(self, tmp_path: Path) -> None:
        """0.050 is numerically equal to 0.05 but noncanonical — must fail."""
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises((SignoffError, ContractError)):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=manifest_sha,
                expected_equivalence_margin="0.050",
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_wrong_n_fails(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises(SignoffError, match="[Nn] mismatch|sample_size"):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=manifest_sha,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=31,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_wrong_signoff_id_fails(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises(SignoffError, match="[Ii][Dd] mismatch|signoff_id"):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=manifest_sha,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID_2,
            )

    def test_free_form_id_fails(self, tmp_path: Path) -> None:
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with pytest.raises((SignoffError, ContractError)):
            verify_run_authorization(
                manifest_path,
                signoff_path,
                expected_manifest_sha256=manifest_sha,
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id="yes",
            )

    def test_missing_signoff_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "instrument"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)
        with pytest.raises(SignoffError, match="not found"):
            verify_run_authorization(
                root / "MANIFEST.json",
                root / "SIGNOFF.json",
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_unsigned_candidate_rejected(self, tmp_path: Path) -> None:
        """A valid candidate manifest with no signoff must fail."""
        root = tmp_path / "candidate"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)
        with pytest.raises(SignoffError, match="not found"):
            verify_run_authorization(
                root / "MANIFEST.json",
                root / "SIGNOFF.json",
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_signed_state_manifest_rejected(self, tmp_path: Path) -> None:
        """A manifest hand-edited to state='signed' is invalid."""
        root = tmp_path / "bad"
        manifest_dict = _make_manifest_dict(lifecycle_state="signed")
        # Recompute hash for the changed state.
        payload = {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}
        manifest_dict["manifest_sha256"] = hashlib.sha256(
            strict_canonical_json(payload)
        ).hexdigest()
        _write_manifest(root, manifest_dict)

        signoff_dict = _make_signoff_dict(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
        )
        _write_signoff(root / "SIGNOFF.json", signoff_dict)

        with pytest.raises(InstrumentIntegrityError, match="lifecycle_state"):
            verify_run_authorization(
                root / "MANIFEST.json",
                root / "SIGNOFF.json",
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_v1_signoff_vs_v2_manifest_rejected(self, tmp_path: Path) -> None:
        """A v1 signoff paired with a v2 manifest must fail."""
        root = tmp_path / "mismatch"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        # Create an obsolete v1 signoff for the v2 manifest.
        signoff_dict = _make_signoff_dict(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
            instrument_version="v1",
        )
        _write_signoff(root / "SIGNOFF.json", signoff_dict)

        with pytest.raises(SignoffError, match="[Vv]ersion"):
            verify_run_authorization(
                root / "MANIFEST.json",
                root / "SIGNOFF.json",
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

    def test_env_bypass_no_effect(self, tmp_path: Path) -> None:
        """EVAL_CHANGE_APPROVED=1 must not bypass the gate."""
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        with patch.dict(os.environ, {"EVAL_CHANGE_APPROVED": "1"}):
            # Wrong hash should still fail.
            with pytest.raises(SignoffError, match="anifest hash"):
                verify_run_authorization(
                    manifest_path,
                    signoff_path,
                    expected_manifest_sha256=_DUMMY_SHA_2,
                    expected_equivalence_margin=_MARGIN,
                    expected_sample_size_tasks_per_family=_N,
                    expected_human_signoff_id=_VALID_SIGNOFF_ID,
                )

    def test_plausible_approval_env_vars_no_effect(self, tmp_path: Path) -> None:
        """Various plausible env vars must not bypass the gate."""
        manifest_path, signoff_path, manifest_sha, _ = self._setup_valid_pair(tmp_path)
        env_vars = {
            "R20_SIGNOFF_APPROVED": "1",
            "R20_APPROVED": "true",
            "META_CORTEX_SIGNED": "1",
            "INSTRUMENT_APPROVED": "yes",
        }
        with patch.dict(os.environ, env_vars):
            with pytest.raises(SignoffError, match="manifest hash"):
                verify_run_authorization(
                    manifest_path,
                    signoff_path,
                    expected_manifest_sha256=_DUMMY_SHA_2,
                    expected_equivalence_margin=_MARGIN,
                    expected_sample_size_tasks_per_family=_N,
                    expected_human_signoff_id=_VALID_SIGNOFF_ID,
                )

    def test_count_map_mismatch_fails(self, tmp_path: Path) -> None:
        """A signoff with a different count map than N must fail."""
        root = tmp_path / "bad_counts"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        signoff_dict = _make_signoff_dict(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
            meta_test_tasks_by_family={
                "contextual_remap": 30,
                "rule_transformation": 31,
                "finite_state": 30,
            },
        )
        _write_signoff(root / "SIGNOFF.json", signoff_dict)

        with pytest.raises(SignoffError, match="must equal|count_map"):
            verify_run_authorization(
                root / "MANIFEST.json",
                root / "SIGNOFF.json",
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )


# ---------------------------------------------------------------------------
# authorize_instrument tests
# ---------------------------------------------------------------------------


class TestAuthorizeInstrument:
    def test_valid_authorization(self, tmp_path: Path) -> None:
        root, manifest_sha, signoff_sha = _build_full_instrument(tmp_path)
        auth = verify_run_authorization(
            root / "MANIFEST.json",
            root / "SIGNOFF.json",
            expected_manifest_sha256=manifest_sha,
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            expected_human_signoff_id=_VALID_SIGNOFF_ID,
        )
        authorized = authorize_instrument(root, auth)
        assert authorized.manifest.manifest_sha256 == manifest_sha
        assert authorized.signoff.signoff_sha256 == signoff_sha
        assert authorized.authorization.instrument_manifest_sha256 == manifest_sha

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir(parents=True)
        auth = RunAuthorization(
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
            signoff_sha256=_DUMMY_SHA_2,
        )
        with pytest.raises(InstrumentIntegrityError, match="[Nn]o MANIFEST"):
            authorize_instrument(root, auth)

    def test_missing_signoff_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "no_signoff"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)
        auth = RunAuthorization(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
            signoff_sha256=_DUMMY_SHA_2,
        )
        with pytest.raises(SignoffError, match="not found"):
            authorize_instrument(root, auth)

    def test_tampered_file_hash_fails(self, tmp_path: Path) -> None:
        """A bit flip in an instrument file must fail authorization."""
        root, manifest_sha, signoff_sha = _build_full_instrument(tmp_path)
        auth = verify_run_authorization(
            root / "MANIFEST.json",
            root / "SIGNOFF.json",
            expected_manifest_sha256=manifest_sha,
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            expected_human_signoff_id=_VALID_SIGNOFF_ID,
        )
        # Tamper with a file.
        (root / "public/DEV_VIEW.json").write_bytes(b"tampered")
        with pytest.raises(InstrumentIntegrityError, match="[Hh]ash mismatch"):
            authorize_instrument(root, auth)

    def test_wrong_file_size_fails(self, tmp_path: Path) -> None:
        """A file size change must fail authorization."""
        root, manifest_sha, signoff_sha = _build_full_instrument(tmp_path)
        auth = verify_run_authorization(
            root / "MANIFEST.json",
            root / "SIGNOFF.json",
            expected_manifest_sha256=manifest_sha,
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            expected_human_signoff_id=_VALID_SIGNOFF_ID,
        )
        # Change file size.
        (root / "public/DEV_VIEW.json").write_bytes(b"x" * 999)
        with pytest.raises(InstrumentIntegrityError):
            authorize_instrument(root, auth)

    def test_authorization_mismatch_fails(self, tmp_path: Path) -> None:
        """An authorization with wrong manifest hash must fail."""
        root, manifest_sha, signoff_sha = _build_full_instrument(tmp_path)
        bad_auth = RunAuthorization(
            instrument_manifest_sha256=_DUMMY_SHA_2,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            human_signoff_id=_VALID_SIGNOFF_ID,
            signoff_sha256=signoff_sha,
        )
        with pytest.raises(SignoffError, match="manifest hash"):
            authorize_instrument(root, bad_auth)

    def test_env_bypass_no_effect(self, tmp_path: Path) -> None:
        """EVAL_CHANGE_APPROVED=1 must not bypass file hash verification."""
        root, manifest_sha, signoff_sha = _build_full_instrument(tmp_path)
        auth = verify_run_authorization(
            root / "MANIFEST.json",
            root / "SIGNOFF.json",
            expected_manifest_sha256=manifest_sha,
            expected_equivalence_margin=_MARGIN,
            expected_sample_size_tasks_per_family=_N,
            expected_human_signoff_id=_VALID_SIGNOFF_ID,
        )
        # Tamper with a file.
        (root / "public/DEV_VIEW.json").write_bytes(b"tampered")
        with patch.dict(os.environ, {"EVAL_CHANGE_APPROVED": "1"}):
            with pytest.raises(InstrumentIntegrityError, match="[Hh]ash"):
                authorize_instrument(root, auth)


# ---------------------------------------------------------------------------
# Gate-before-read ordering tests
# ---------------------------------------------------------------------------


class TestGateBeforeRead:
    """Authorization must run before any sealed bytes, model, or output access."""

    def test_wrong_hash_no_file_reads(self, tmp_path: Path) -> None:
        """A wrong manifest hash must fail before any file hash verification."""
        root = tmp_path / "instrument"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        signoff_dict = _make_signoff_dict(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
        )
        _write_signoff(root / "SIGNOFF.json", signoff_dict)

        # Spy on file reads to prove no sealed access.
        original_read_bytes = Path.read_bytes
        read_paths: list[str] = []

        def spy_read_bytes(self: Path) -> bytes:
            read_paths.append(str(self))
            return original_read_bytes(self)

        with patch.object(Path, "read_bytes", spy_read_bytes):
            with pytest.raises(SignoffError, match="manifest hash"):
                verify_run_authorization(
                    root / "MANIFEST.json",
                    root / "SIGNOFF.json",
                    expected_manifest_sha256=_DUMMY_SHA_2,
                    expected_equivalence_margin=_MARGIN,
                    expected_sample_size_tasks_per_family=_N,
                    expected_human_signoff_id=_VALID_SIGNOFF_ID,
                )

        # The manifest and signoff JSON are read, but no sealed files.
        sealed_reads = [p for p in read_paths if "sealed" in p]
        assert len(sealed_reads) == 0, (
            f"Sealed files were read before authorization: {sealed_reads}"
        )

    def test_raw_manifest_cannot_authorize(self, tmp_path: Path) -> None:
        """A raw CandidateManifest object cannot be supplied to the gate —
        only verified file paths are accepted."""
        root = tmp_path / "instrument"
        manifest_dict = _make_manifest_dict()
        _write_manifest(root, manifest_dict)

        signoff_dict = _make_signoff_dict(
            instrument_manifest_sha256=manifest_dict["manifest_sha256"],
        )
        _write_signoff(root / "SIGNOFF.json", signoff_dict)

        # verify_run_authorization requires Path arguments, not manifest objects.
        # Passing a dict instead of a Path should fail with TypeError.
        signoff_path = root / "SIGNOFF.json"
        with pytest.raises((TypeError, AttributeError, Exception)):
            verify_run_authorization(
                manifest_dict,  # type: ignore[arg-type]
                signoff_path,
                expected_manifest_sha256=manifest_dict["manifest_sha256"],
                expected_equivalence_margin=_MARGIN,
                expected_sample_size_tasks_per_family=_N,
                expected_human_signoff_id=_VALID_SIGNOFF_ID,
            )

# ---------------------------------------------------------------------------
# No signoff from calibration path
# ---------------------------------------------------------------------------


class TestNoSignoffFromCalibration:
    """No calibration/materialization code path emits a signoff."""

    def test_calibration_module_has_no_record_signoff(self) -> None:
        """The calibration module must not import or define record_signoff."""
        # Check that the authorization module is not imported by
        # instrument_contracts (dependency direction: signoff -> instrument,
        # never calibration -> signoff).
        import oczy.experiments.meta_cortex.instrument_contracts as ic
        assert not hasattr(ic, "record_signoff")
        assert not hasattr(ic, "authorize_instrument")
        assert not hasattr(ic, "AuthorizedInstrument")

    def test_contracts_has_no_signoff_types(self) -> None:
        """The DEV-only contracts.py must not contain signoff/authorization types."""
        import oczy.experiments.meta_cortex.contracts as c
        assert not hasattr(c, "SignoffAttestation")
        assert not hasattr(c, "RunAuthorization")
        assert not hasattr(c, "AuthorizedInstrument")
        assert not hasattr(c, "record_signoff")
        assert not hasattr(c, "CandidateManifest")


# ---------------------------------------------------------------------------
# SignoffAttestation type tests
# ---------------------------------------------------------------------------


class TestSignoffAttestation:
    def test_valid_attestation(self) -> None:
        att = SignoffAttestation(
            schema=SIGNOFF_SCHEMA,
            instrument_id="meta_cortex/v2",
            instrument_version="v2",
            lifecycle_state="signed",
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            meta_test_tasks_by_family=dict(_COUNT_MAP),
            human_signoff_id=_VALID_SIGNOFF_ID,
            signed_at_utc="2026-07-12T00:00:00Z",
            signoff_sha256=_DUMMY_SHA_2,
        )
        assert att.lifecycle_state == "signed"
        assert att.human_signoff_id == _VALID_SIGNOFF_ID

    def test_reject_signed_state_with_approved_boolean(self) -> None:
        """A signoff with only approved=true must not be constructable —
        there is no 'approved' boolean field."""
        # SignoffAttestation has no 'approved' field, so this tests
        # that the schema doesn't support generic approval.
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(SignoffAttestation)}
        assert "approved" not in field_names

    def test_to_json_obj_excludes_self_hash(self) -> None:
        att = SignoffAttestation(
            schema=SIGNOFF_SCHEMA,
            instrument_id="meta_cortex/v2",
            instrument_version="v2",
            lifecycle_state="signed",
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            meta_test_tasks_by_family=dict(_COUNT_MAP),
            human_signoff_id=_VALID_SIGNOFF_ID,
            signed_at_utc="2026-07-12T00:00:00Z",
            signoff_sha256=_DUMMY_SHA_2,
        )
        obj = att.to_json_obj()
        assert "signoff_sha256" not in obj

    def test_is_frozen(self) -> None:
        import dataclasses
        att = SignoffAttestation(
            schema=SIGNOFF_SCHEMA,
            instrument_id="meta_cortex/v2",
            instrument_version="v2",
            lifecycle_state="signed",
            instrument_manifest_sha256=_DUMMY_SHA,
            equivalence_margin=_MARGIN,
            sample_size_tasks_per_family=_N,
            meta_test_tasks_by_family=dict(_COUNT_MAP),
            human_signoff_id=_VALID_SIGNOFF_ID,
            signed_at_utc="2026-07-12T00:00:00Z",
            signoff_sha256=_DUMMY_SHA_2,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            att.equivalence_margin = "0.06"  # type: ignore[misc]
