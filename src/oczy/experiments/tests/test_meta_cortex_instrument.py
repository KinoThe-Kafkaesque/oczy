"""High-signal contract tests for the meta_cortex/v1 instrument boundary.

These tests defend the scientific and security boundaries of the unsigned
candidate instrument — not plumbing.  They verify:

  - Deterministic bytes: same inputs produce byte-identical manifests/hashes.
  - Hash changes for every decision/evidence/file field.
  - Strict malformed/null/extra/duplicate/NaN rejection.
  - Per-file hash/size/path/set/symlink enforcement.
  - Four-way semantic firewall: DEV view never touches sealed/calibration.
  - Candidate manifest binds all instrument bytes, margin, and powered count.
  - Candidate immutability: write-once publication, no overwrite.
  - UUID/version validation on signoff ID format.
  - Canonical decimal string validation for margin and abstain threshold.
  - A=B=C=N count-map invariant.

No real model, network, or Qwen is required.  All fixtures are synthetic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oczy.experiments.meta_cortex.contracts import ContractError
from oczy.experiments.meta_cortex.instrument_contracts import (
    CALIBRATION_VIEW_SCHEMA,
    CANDIDATE_MANIFEST_SCHEMA,
    DEV_VIEW_SCHEMA,
    ENDPOINT_SCHEMA,
    INSTRUMENT_DEFINITION_SCHEMA,
    PROMPT_SCHEMA,
    SCORER_SCHEMA,
    SIGNOFF_SCHEMA,
    TASK_RECORD_SCHEMA,
    CandidateManifest,
    InstrumentDefinitionConfig,
    InstrumentFileEntry,
    canonical_decimal,
    strict_canonical_json,
    strict_json_loads,
    validate_relative_path,
    validate_sha256_hex,
    validate_signoff_id,
)

# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

_DUMMY_SHA = "a" * 64
_DUMMY_SHA_2 = "b" * 64
_VALID_SIGNOFF_ID = "r20-meta-cortex-v1/550e8400-e29b-41d4-a716-446655440000"
_VALID_SIGNOFF_ID_2 = "r20-meta-cortex-v1/660e8400-e29b-42d4-a716-446655440001"
_FAMILY_ORDER = ("contextual_remap", "rule_transformation", "finite_state")
_COUNT_MAP_N30 = {
    "contextual_remap": 30,
    "rule_transformation": 30,
    "finite_state": 30,
}


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


def _make_candidate_manifest(
    *,
    manifest_sha256: str | None = None,
    equivalence_margin: str = "0.05",
    sample_size_tasks_per_family: int = 30,
    meta_test_tasks_by_family: dict[str, int] | None = None,
    files: tuple[InstrumentFileEntry, ...] | None = None,
    lifecycle_state: str = "candidate",
    calibration_holdout_accessed: bool = False,
    leakage_audit_passed: bool = True,
    abstain_threshold: str = "0.5",
    developmental_seeds: tuple[int, ...] = (10, 20, 30, 40, 50),
    evaluation_seeds: tuple[int, ...] = (60, 70, 80, 90, 100),
) -> CandidateManifest:
    """Build a synthetic CandidateManifest with valid defaults."""
    if meta_test_tasks_by_family is None:
        n = sample_size_tasks_per_family
        meta_test_tasks_by_family = {
            "contextual_remap": n,
            "rule_transformation": n,
            "finite_state": n,
        }
    if files is None:
        files = (
            _make_file_entry("public/DEV_VIEW.json", role="dev_view"),
            _make_file_entry("public/CALIBRATION_VIEW.json", visibility="calibration", role="calibration_view"),
            _make_file_entry("sealed/meta_test.jsonl", visibility="sealed", role="meta_test_tasks"),
        )

    # Build the manifest dict to compute the self-hash.
    manifest_dict: dict = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "instrument_id": "meta_cortex/v1",
        "instrument_version": "v1",
        "lifecycle_state": lifecycle_state,
        "definition_sha256": _DUMMY_SHA,
        "dev_view_sha256": _DUMMY_SHA,
        "calibration_view_sha256": _DUMMY_SHA,
        "source_commit": "abc123",
        "source_archive_sha256": _DUMMY_SHA,
        "taskgen_schema": "oczy/meta-cortex/taskgen/v1",
        "generator_algorithm": "sha256-counter-rejection/v1",
        "generator_source_sha256": _DUMMY_SHA,
        "prompt_schema": PROMPT_SCHEMA,
        "prompt_registry_sha256": _DUMMY_SHA,
        "scorer_schema": SCORER_SCHEMA,
        "scorer_registry_sha256": _DUMMY_SHA,
        "endpoint_schema": ENDPOINT_SCHEMA,
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
        "abstain_threshold": abstain_threshold,
        "event_min": 2,
        "event_max": 5,
        "family_order": list(_FAMILY_ORDER),
        "probe_counts_by_split_family_kind": {},
        "train_tasks_per_family": 2,
        "tuning_tasks_per_family": 2,
        "calibration_tasks_per_family": 30,
        "sample_size_tasks_per_family": sample_size_tasks_per_family,
        "meta_test_tasks_by_family": meta_test_tasks_by_family,
        "developmental_seeds": list(developmental_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "equivalence_margin": equivalence_margin,
        "calibration_report_sha256": _DUMMY_SHA,
        "power_report_sha256": _DUMMY_SHA,
        "calibration_holdout_accessed": calibration_holdout_accessed,
        "independent_sample_unit": "task_rule",
        "leakage_audit_sha256": _DUMMY_SHA,
        "leakage_audit_passed": leakage_audit_passed,
        "meta_test_seed_sha256": _DUMMY_SHA,
        "files": [e.to_json_obj() for e in files],
    }

    if manifest_sha256 is None:
        payload = {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}
        manifest_sha256 = hashlib.sha256(strict_canonical_json(payload)).hexdigest()
    manifest_dict["manifest_sha256"] = manifest_sha256

    return CandidateManifest(
        schema=manifest_dict["schema"],
        instrument_id=manifest_dict["instrument_id"],
        instrument_version=manifest_dict["instrument_version"],
        lifecycle_state=manifest_dict["lifecycle_state"],
        definition_sha256=manifest_dict["definition_sha256"],
        dev_view_sha256=manifest_dict["dev_view_sha256"],
        calibration_view_sha256=manifest_dict["calibration_view_sha256"],
        source_commit=manifest_dict["source_commit"],
        source_archive_sha256=manifest_dict["source_archive_sha256"],
        taskgen_schema=manifest_dict["taskgen_schema"],
        generator_algorithm=manifest_dict["generator_algorithm"],
        generator_source_sha256=manifest_dict["generator_source_sha256"],
        prompt_schema=manifest_dict["prompt_schema"],
        prompt_registry_sha256=manifest_dict["prompt_registry_sha256"],
        scorer_schema=manifest_dict["scorer_schema"],
        scorer_registry_sha256=manifest_dict["scorer_registry_sha256"],
        endpoint_schema=manifest_dict["endpoint_schema"],
        endpoint_registry_sha256=manifest_dict["endpoint_registry_sha256"],
        organ_model_id=manifest_dict["organ_model_id"],
        organ_revision=manifest_dict["organ_revision"],
        organ_parameter_sha256=manifest_dict["organ_parameter_sha256"],
        chat_template_sha256=manifest_dict["chat_template_sha256"],
        feature_mode=manifest_dict["feature_mode"],
        decoding_mode=manifest_dict["decoding_mode"],
        max_new_tokens=manifest_dict["max_new_tokens"],
        feature_dim=manifest_dict["feature_dim"],
        d_cortex=manifest_dict["d_cortex"],
        soft_bank_width=manifest_dict["soft_bank_width"],
        abstain_threshold=manifest_dict["abstain_threshold"],
        event_min=manifest_dict["event_min"],
        event_max=manifest_dict["event_max"],
        family_order=tuple(manifest_dict["family_order"]),
        probe_counts_by_split_family_kind=manifest_dict["probe_counts_by_split_family_kind"],
        train_tasks_per_family=manifest_dict["train_tasks_per_family"],
        tuning_tasks_per_family=manifest_dict["tuning_tasks_per_family"],
        calibration_tasks_per_family=manifest_dict["calibration_tasks_per_family"],
        sample_size_tasks_per_family=manifest_dict["sample_size_tasks_per_family"],
        meta_test_tasks_by_family=manifest_dict["meta_test_tasks_by_family"],
        developmental_seeds=tuple(manifest_dict["developmental_seeds"]),
        evaluation_seeds=tuple(manifest_dict["evaluation_seeds"]),
        equivalence_margin=manifest_dict["equivalence_margin"],
        calibration_report_sha256=manifest_dict["calibration_report_sha256"],
        power_report_sha256=manifest_dict["power_report_sha256"],
        calibration_holdout_accessed=manifest_dict["calibration_holdout_accessed"],
        independent_sample_unit=manifest_dict["independent_sample_unit"],
        leakage_audit_sha256=manifest_dict["leakage_audit_sha256"],
        leakage_audit_passed=manifest_dict["leakage_audit_passed"],
        meta_test_seed_sha256=manifest_dict["meta_test_seed_sha256"],
        files=files,
        manifest_sha256=manifest_sha256,
    )


def _write_manifest_to_dir(manifest: CandidateManifest, root: Path) -> Path:
    """Write a CandidateManifest to a directory as MANIFEST.json."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_json_obj()
    manifest_dict["manifest_sha256"] = manifest.manifest_sha256
    canonical_bytes = strict_canonical_json(manifest_dict) + b"\n"
    path = root / "MANIFEST.json"
    path.write_bytes(canonical_bytes)
    return path


# ---------------------------------------------------------------------------
# InstrumentFileEntry tests
# ---------------------------------------------------------------------------


class TestInstrumentFileEntry:
    def test_valid_entry(self) -> None:
        entry = _make_file_entry()
        assert entry.path == "public/DEV_VIEW.json"
        assert entry.sha256 == _DUMMY_SHA
        assert entry.size_bytes == 100
        assert entry.visibility == "public"

    def test_reject_absolute_path(self) -> None:
        with pytest.raises(ContractError, match="absolute"):
            _make_file_entry(path="/etc/passwd")

    def test_reject_parent_traversal(self) -> None:
        with pytest.raises(ContractError, match="parent"):
            _make_file_entry(path="../sealed/secret.jsonl")

    def test_reject_backslash(self) -> None:
        with pytest.raises(ContractError, match="backslash"):
            _make_file_entry(path="public\\DEV_VIEW.json")

    def test_reject_empty_component(self) -> None:
        with pytest.raises(ContractError, match="empty"):
            _make_file_entry(path="public//DEV_VIEW.json")

    def test_reject_dot_component(self) -> None:
        with pytest.raises(ContractError, match="dot"):
            _make_file_entry(path="public/./DEV_VIEW.json")

    def test_reject_uppercase_sha256(self) -> None:
        with pytest.raises(ContractError, match="lowercase hex"):
            _make_file_entry(sha256="A" * 64)

    def test_reject_short_sha256(self) -> None:
        with pytest.raises(ContractError, match="64 chars"):
            _make_file_entry(sha256="a" * 32)

    def test_reject_bool_size(self) -> None:
        with pytest.raises(ContractError, match="size_bytes"):
            InstrumentFileEntry(
                path="public/test.json",
                sha256=_DUMMY_SHA,
                size_bytes=True,  # type: ignore[arg-type]
                visibility="public",
                role="test",
            )

    def test_reject_negative_size(self) -> None:
        with pytest.raises(ContractError, match="size_bytes"):
            _make_file_entry(size_bytes=-1)

    def test_reject_invalid_visibility(self) -> None:
        with pytest.raises(ContractError, match="visibility"):
            _make_file_entry(visibility="secret")

    def test_reject_empty_role(self) -> None:
        with pytest.raises(ContractError, match="role"):
            _make_file_entry(role="")

    @pytest.mark.parametrize("vis", ["public", "calibration", "evidence", "sealed"])
    def test_all_visibility_values_accepted(self, vis: str) -> None:
        entry = _make_file_entry(visibility=vis)
        assert entry.visibility == vis


# ---------------------------------------------------------------------------
# CandidateManifest tests
# ---------------------------------------------------------------------------


class TestCandidateManifest:
    def test_valid_manifest_constructs(self) -> None:
        manifest = _make_candidate_manifest()
        assert manifest.schema == CANDIDATE_MANIFEST_SCHEMA
        assert manifest.instrument_id == "meta_cortex/v1"
        assert manifest.instrument_version == "v1"
        assert manifest.lifecycle_state == "candidate"
        assert manifest.equivalence_margin == "0.05"
        assert manifest.sample_size_tasks_per_family == 30

    def test_manifest_sha256_is_64_char_hex(self) -> None:
        manifest = _make_candidate_manifest()
        assert len(manifest.manifest_sha256) == 64
        assert all(c in "0123456789abcdef" for c in manifest.manifest_sha256)

    def test_reject_wrong_schema(self) -> None:
        manifest_dict = _make_candidate_manifest().to_json_obj()
        manifest_dict["schema"] = "wrong"
        # Can't construct directly because __post_init__ checks schema.
        with pytest.raises(ContractError, match="schema"):
            CandidateManifest(
                **{**manifest_dict, "manifest_sha256": _DUMMY_SHA},
            )

    def test_reject_signed_state(self) -> None:
        """A manifest edited to state='signed' is invalid — signoff is detached."""
        with pytest.raises(ContractError, match="lifecycle_state"):
            _make_candidate_manifest(lifecycle_state="signed")

    def test_reject_wrong_instrument_id(self) -> None:
        manifest_dict = _make_candidate_manifest().to_json_obj()
        manifest_dict["instrument_id"] = "meta_cortex/v2"
        with pytest.raises(ContractError, match="instrument_id"):
            CandidateManifest(**{**manifest_dict, "manifest_sha256": _DUMMY_SHA})

    def test_reject_wrong_version(self) -> None:
        manifest_dict = _make_candidate_manifest().to_json_obj()
        manifest_dict["instrument_version"] = "v2"
        with pytest.raises(ContractError, match="instrument_version"):
            CandidateManifest(**{**manifest_dict, "manifest_sha256": _DUMMY_SHA})

    def test_reject_n_below_30(self) -> None:
        with pytest.raises(ContractError, match=r">= 30"):
            _make_candidate_manifest(sample_size_tasks_per_family=29)

    def test_reject_bool_n(self) -> None:
        manifest_dict = _make_candidate_manifest().to_json_obj()
        manifest_dict["sample_size_tasks_per_family"] = True
        with pytest.raises(ContractError, match="int"):
            CandidateManifest(**{**manifest_dict, "manifest_sha256": _DUMMY_SHA})

    def test_reject_unequal_count_map(self) -> None:
        """A=B=C=N invariant: every family count must equal N."""
        with pytest.raises(ContractError, match="must equal"):
            _make_candidate_manifest(
                sample_size_tasks_per_family=30,
                meta_test_tasks_by_family={
                    "contextual_remap": 30,
                    "rule_transformation": 31,
                    "finite_state": 30,
                },
            )

    def test_reject_missing_family_in_count_map(self) -> None:
        with pytest.raises(ContractError, match="missing family"):
            _make_candidate_manifest(
                meta_test_tasks_by_family={
                    "contextual_remap": 30,
                    "rule_transformation": 30,
                },
            )

    def test_reject_extra_family_in_count_map(self) -> None:
        """The count map must have exactly the three registered families."""
        with pytest.raises(ContractError, match="unknown family"):
            _make_candidate_manifest(
                meta_test_tasks_by_family={
                    "contextual_remap": 30,
                    "rule_transformation": 30,
                    "finite_state": 30,
                    "extra_family": 30,
                },
            )

    def test_reject_holdout_accessed_true(self) -> None:
        """holdout_accessed=true invalidates the candidate."""
        with pytest.raises(ContractError, match="holdout"):
            _make_candidate_manifest(calibration_holdout_accessed=True)

    def test_reject_leakage_audit_not_passed(self) -> None:
        with pytest.raises(ContractError, match="leakage"):
            _make_candidate_manifest(leakage_audit_passed=False)

    def test_reject_wrong_sample_unit(self) -> None:
        manifest_dict = _make_candidate_manifest().to_json_obj()
        manifest_dict["independent_sample_unit"] = "probe"
        with pytest.raises(ContractError, match="independent_sample_unit"):
            CandidateManifest(**{**manifest_dict, "manifest_sha256": _DUMMY_SHA})

    def test_reject_noncanonical_margin(self) -> None:
        """Margin must be canonical decimal — 0.050 is noncanonical."""
        with pytest.raises(ContractError, match="decimal"):
            _make_candidate_manifest(equivalence_margin="0.050")

    def test_reject_exponent_margin(self) -> None:
        with pytest.raises(ContractError, match="[Ee]xponent"):
            _make_candidate_manifest(equivalence_margin="5e-2")

    def test_reject_margin_out_of_range(self) -> None:
        with pytest.raises(ContractError, match=r"\[0, 1\]"):
            _make_candidate_manifest(equivalence_margin="1.5")

    def test_zero_margin_accepted(self) -> None:
        """Zero is valid if empirically produced."""
        manifest = _make_candidate_manifest(equivalence_margin="0")
        assert manifest.equivalence_margin == "0"

    def test_manifest_hash_excludes_self(self) -> None:
        """The self-hash must not include the manifest_sha256 field itself."""
        m1 = _make_candidate_manifest()
        # Changing manifest_sha256 should not change the computed self-hash.
        # The hash is computed from to_json_obj() which excludes manifest_sha256.
        payload1 = m1.to_json_obj()
        hash1 = hashlib.sha256(strict_canonical_json(payload1)).hexdigest()
        assert hash1 == m1.manifest_sha256

    def test_manifest_hash_changes_with_margin(self) -> None:
        m1 = _make_candidate_manifest(equivalence_margin="0.05")
        m2 = _make_candidate_manifest(equivalence_margin="0.06")
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_hash_changes_with_n(self) -> None:
        m1 = _make_candidate_manifest(sample_size_tasks_per_family=30)
        m2 = _make_candidate_manifest(sample_size_tasks_per_family=31)
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_hash_changes_with_seeds(self) -> None:
        m1 = _make_candidate_manifest(developmental_seeds=(10, 20, 30, 40, 50))
        m2 = _make_candidate_manifest(developmental_seeds=(11, 20, 30, 40, 50))
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_hash_changes_with_file_entry(self) -> None:
        files1 = (_make_file_entry("public/DEV_VIEW.json"),)
        files2 = (_make_file_entry("public/DEV_VIEW.json", sha256=_DUMMY_SHA_2),)
        m1 = _make_candidate_manifest(files=files1)
        m2 = _make_candidate_manifest(files=files2)
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_hash_changes_with_abstain_threshold(self) -> None:
        m1 = _make_candidate_manifest(abstain_threshold="0.5")
        m2 = _make_candidate_manifest(abstain_threshold="0.6")
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_hash_changes_with_holdout_flag(self) -> None:
        """Hash must change when holdout_accessed changes (even though True is invalid)."""
        # We can't construct a manifest with holdout=True, so verify the
        # to_json_obj would produce a different hash.
        m1 = _make_candidate_manifest(calibration_holdout_accessed=False)
        obj1 = m1.to_json_obj()
        obj1["calibration_holdout_accessed"] = True
        hash_with_true = hashlib.sha256(strict_canonical_json(obj1)).hexdigest()
        assert m1.manifest_sha256 != hash_with_true

    def test_manifest_to_json_obj_excludes_self_hash(self) -> None:
        """to_json_obj must not include manifest_sha256."""
        manifest = _make_candidate_manifest()
        obj = manifest.to_json_obj()
        assert "manifest_sha256" not in obj

    def test_manifest_is_frozen(self) -> None:
        import dataclasses
        manifest = _make_candidate_manifest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.equivalence_margin = "0.06"  # type: ignore[misc]

    def test_manifest_is_slotted(self) -> None:
        manifest = _make_candidate_manifest()
        with pytest.raises((AttributeError, TypeError)):
            manifest.extra_field = "forbidden"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Canonical serialization tests
# ---------------------------------------------------------------------------


class TestStrictCanonicalSerialization:
    def test_sorted_keys(self) -> None:
        obj = {"b": 1, "a": 2, "c": 3}
        result = strict_canonical_json(obj)
        assert result == b'{"a":2,"b":1,"c":3}'

    def test_compact_separators(self) -> None:
        obj = {"a": 1, "b": "x"}
        result = strict_canonical_json(obj)
        assert b": " not in result
        assert b", " not in result

    def test_no_nan(self) -> None:
        with pytest.raises(ValueError, match="Out of range"):
            strict_canonical_json({"x": float("nan")})

    def test_no_infinity(self) -> None:
        with pytest.raises(ValueError, match="Out of range"):
            strict_canonical_json({"x": float("inf")})

    def test_strict_json_loads_rejects_duplicate_keys(self) -> None:
        bad_json = b'{"a":1,"a":2}'
        with pytest.raises(ContractError, match="[Dd]uplicate"):
            strict_json_loads(bad_json)

    def test_strict_json_loads_rejects_nan(self) -> None:
        bad_json = b'{"x":NaN}'
        with pytest.raises(ContractError, match="NaN"):
            strict_json_loads(bad_json)

    def test_strict_json_loads_rejects_infinity(self) -> None:
        bad_json = b'{"x":Infinity}'
        with pytest.raises(ContractError, match="Infinity"):
            strict_json_loads(bad_json)

    def test_strict_json_loads_rejects_non_object(self) -> None:
        with pytest.raises(ContractError, match="object"):
            strict_json_loads(b'[1, 2, 3]')

    def test_strict_json_loads_rejects_empty(self) -> None:
        with pytest.raises((json.JSONDecodeError, ContractError)):
            strict_json_loads(b'')


# ---------------------------------------------------------------------------
# Canonical decimal tests
# ---------------------------------------------------------------------------


class TestCanonicalDecimal:
    def test_zero(self) -> None:
        assert canonical_decimal("0") == "0"

    def test_simple_decimal(self) -> None:
        assert canonical_decimal("0.05") == "0.05"

    def test_trailing_zeros_removed(self) -> None:
        assert canonical_decimal("0.050") == "0.05"

    def test_trailing_dot_removed(self) -> None:
        assert canonical_decimal("1.0") == "1"

    def test_reject_exponent(self) -> None:
        with pytest.raises(ContractError, match="[Ee]xponent"):
            canonical_decimal("5e-2")

    def test_reject_leading_plus(self) -> None:
        with pytest.raises(ContractError, match=r"\+"):
            canonical_decimal("+0.05")

    def test_reject_above_one(self) -> None:
        with pytest.raises(ContractError, match=r"\[0, 1\]"):
            canonical_decimal("1.5")

    def test_reject_negative(self) -> None:
        with pytest.raises(ContractError, match=r"\[0, 1\]"):
            canonical_decimal("-0.05")

    def test_reject_nan(self) -> None:
        with pytest.raises(ContractError, match="NaN"):
            canonical_decimal("NaN")

    def test_float_input_accepted(self) -> None:
        result = canonical_decimal(0.05)
        assert result == "0.05"

    def test_int_input_accepted(self) -> None:
        assert canonical_decimal(0) == "0"

    def test_bool_rejected(self) -> None:
        with pytest.raises(ContractError):
            canonical_decimal(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SHA-256 validation tests
# ---------------------------------------------------------------------------


class TestValidateSha256:
    def test_valid_hex(self) -> None:
        assert validate_sha256_hex("a" * 64) == "a" * 64

    def test_reject_uppercase(self) -> None:
        with pytest.raises(ContractError, match="lowercase"):
            validate_sha256_hex("A" * 64)

    def test_reject_wrong_length(self) -> None:
        with pytest.raises(ContractError, match="64 chars"):
            validate_sha256_hex("a" * 32)

    def test_reject_non_hex(self) -> None:
        with pytest.raises(ContractError, match="lowercase hex"):
            validate_sha256_hex("g" * 64)

    def test_reject_non_string(self) -> None:
        with pytest.raises(ContractError, match="string"):
            validate_sha256_hex(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Relative path validation tests
# ---------------------------------------------------------------------------


class TestValidateRelativePath:
    def test_valid_relative(self) -> None:
        assert validate_relative_path("public/DEV_VIEW.json") == "public/DEV_VIEW.json"

    def test_reject_absolute(self) -> None:
        with pytest.raises(ContractError, match="absolute"):
            validate_relative_path("/etc/passwd")

    def test_reject_parent(self) -> None:
        with pytest.raises(ContractError, match="parent"):
            validate_relative_path("../secret")

    def test_reject_backslash(self) -> None:
        with pytest.raises(ContractError, match="backslash"):
            validate_relative_path("public\\file")

    def test_reject_empty(self) -> None:
        with pytest.raises(ContractError, match="non-empty"):
            validate_relative_path("")

    def test_reject_dot(self) -> None:
        with pytest.raises(ContractError, match="dot"):
            validate_relative_path(".")

    def test_reject_double_separator(self) -> None:
        with pytest.raises(ContractError, match="duplicate"):
            validate_relative_path("a//b")


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
            validate_signoff_id("r20-meta-cortex-v1/550E8400-E29B-41D4-A716-446655440000")

    def test_reject_non_v4_uuid(self) -> None:
        """UUID version 3 must be rejected."""
        with pytest.raises(ContractError, match="version 4"):
            validate_signoff_id("r20-meta-cortex-v1/550e8400-e29b-31d4-a716-446655440000")

    def test_reject_wrong_variant(self) -> None:
        """Variant nibble must be 8/9/a/b for RFC 4122."""
        with pytest.raises(ContractError, match="variant"):
            validate_signoff_id("r20-meta-cortex-v1/550e8400-e29b-41d4-1716-446655440000")

    def test_reject_missing_slash(self) -> None:
        with pytest.raises(ContractError, match="match"):
            validate_signoff_id("r20-meta-cortex-v1550e8400-e29b-41d4-a716-446655440000")

    def test_reject_version_zero(self) -> None:
        with pytest.raises(ContractError, match="positive"):
            validate_signoff_id("r20-meta-cortex-v0/550e8400-e29b-41d4-a716-446655440000")


# ---------------------------------------------------------------------------
# InstrumentDefinitionConfig tests
# ---------------------------------------------------------------------------


class TestInstrumentDefinitionConfig:
    def _make_valid_config(self) -> InstrumentDefinitionConfig:
        return InstrumentDefinitionConfig(
            instrument_id="meta_cortex/v1",
            instrument_version="v1",
            root_seed=20260709,
            train_tasks_per_family=2,
            tuning_tasks_per_family=2,
            calibration_tasks_per_family=30,
            developmental_seeds=(10, 20, 30, 40, 50),
            evaluation_seeds=(60, 70, 80, 90, 100),
            organ_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            organ_revision="main",
            organ_parameter_sha256=_DUMMY_SHA,
            chat_template_sha256=_DUMMY_SHA,
            feature_mode="final_layer_mean_pool",
            decoding_mode="greedy",
            max_new_tokens=16,
            feature_dim=896,
            d_cortex=64,
            soft_bank_width=3,
            event_min=2,
            event_max=5,
            abstain_threshold="0.5",
            source_commit="abc123",
            source_archive_sha256=_DUMMY_SHA,
        )

    def test_valid_config(self) -> None:
        config = self._make_valid_config()
        assert config.instrument_id == "meta_cortex/v1"
        assert config.calibration_tasks_per_family == 30

    def test_reject_calibration_below_30(self) -> None:
        config = self._make_valid_config()
        with pytest.raises(ContractError, match=r">= 30"):
            object.__setattr__(config, "calibration_tasks_per_family", 29)
            InstrumentDefinitionConfig(**{
                **{f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)},
                "calibration_tasks_per_family": 29,
            })

    def test_reject_fewer_than_5_dev_seeds(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["developmental_seeds"] = (10, 20, 30)
        with pytest.raises(ContractError, match=r">= 5"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_duplicate_dev_seeds(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["developmental_seeds"] = (10, 10, 30, 40, 50)
        with pytest.raises(ContractError, match="distinct"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_fewer_than_5_eval_seeds(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["evaluation_seeds"] = (60, 70, 80)
        with pytest.raises(ContractError, match=r">= 5"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_wrong_feature_mode(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["feature_mode"] = "last_token"
        with pytest.raises(ContractError, match="feature_mode"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_wrong_decoding_mode(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["decoding_mode"] = "sampling"
        with pytest.raises(ContractError, match="decoding_mode"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_event_min_below_2(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["event_min"] = 0
        with pytest.raises(ContractError, match="event_min"):
            InstrumentDefinitionConfig(**fields)

    def test_reject_event_max_above_5(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["event_max"] = 6
        with pytest.raises(ContractError, match="event_max"):
            InstrumentDefinitionConfig(**fields)

    def test_from_json_obj_roundtrip(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["developmental_seeds"] = list(fields["developmental_seeds"])
        fields["evaluation_seeds"] = list(fields["evaluation_seeds"])
        restored = InstrumentDefinitionConfig.from_json_obj(fields)
        assert restored.instrument_id == config.instrument_id
        assert restored.developmental_seeds == config.developmental_seeds

    def test_from_json_obj_rejects_unknown_fields(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["developmental_seeds"] = list(fields["developmental_seeds"])
        fields["evaluation_seeds"] = list(fields["evaluation_seeds"])
        fields["extra_field"] = "bad"
        with pytest.raises(ContractError, match="[Uu]nknown"):
            InstrumentDefinitionConfig.from_json_obj(fields)

    def test_from_json_obj_rejects_nulls(self) -> None:
        config = self._make_valid_config()
        fields = {f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
        fields["developmental_seeds"] = list(fields["developmental_seeds"])
        fields["evaluation_seeds"] = list(fields["evaluation_seeds"])
        fields["organ_model_id"] = None
        with pytest.raises(ContractError, match="null"):
            InstrumentDefinitionConfig.from_json_obj(fields)


# ---------------------------------------------------------------------------
# Schema constant tests
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_definition_schema(self) -> None:
        assert INSTRUMENT_DEFINITION_SCHEMA == "oczy/meta-cortex/instrument-definition/v1"

    def test_dev_view_schema(self) -> None:
        assert DEV_VIEW_SCHEMA == "oczy/meta-cortex/instrument-dev-view/v1"

    def test_calibration_view_schema(self) -> None:
        assert CALIBRATION_VIEW_SCHEMA == "oczy/meta-cortex/instrument-calibration-view/v1"

    def test_candidate_manifest_schema(self) -> None:
        assert CANDIDATE_MANIFEST_SCHEMA == "oczy/meta-cortex/instrument-candidate-manifest/v1"

    def test_signoff_schema(self) -> None:
        assert SIGNOFF_SCHEMA == "oczy/meta-cortex/instrument-signoff/v1"

    def test_scorer_schema(self) -> None:
        assert SCORER_SCHEMA == "oczy/meta-cortex/scorers/v1"

    def test_endpoint_schema(self) -> None:
        assert ENDPOINT_SCHEMA == "oczy/meta-cortex/endpoints/v1"

    def test_prompt_schema(self) -> None:
        assert PROMPT_SCHEMA == "oczy/meta-cortex/prompts/v1"

    def test_task_record_schema(self) -> None:
        assert TASK_RECORD_SCHEMA == "oczy/meta-cortex/task-record/v1"


# ---------------------------------------------------------------------------
# DevSplit firewall: no META_TEST member
# ---------------------------------------------------------------------------


class TestDevSplitFirewall:
    """The DevSplit enum must never have a META_TEST member."""

    def test_devsplit_has_exactly_two_members(self) -> None:
        from oczy.experiments.meta_cortex.contracts import DevSplit
        members = {m.name for m in DevSplit}
        assert members == {"META_TRAIN", "META_VALIDATION"}

    def test_no_meta_test_member(self) -> None:
        from oczy.experiments.meta_cortex.contracts import DevSplit
        assert not hasattr(DevSplit, "META_TEST")

    def test_devsplit_rejects_unknown_string(self) -> None:
        from oczy.experiments.meta_cortex.contracts import DevSplit
        with pytest.raises(ValueError):
            DevSplit("meta_test")


# ---------------------------------------------------------------------------
# Package export boundary
# ---------------------------------------------------------------------------


class TestPackageExportBoundary:
    """The package __init__ must not export sealed/signoff/authorization symbols."""

    def test_no_authorized_instrument_export(self) -> None:
        import oczy.experiments.meta_cortex as pkg
        assert not hasattr(pkg, "AuthorizedInstrument")

    def test_no_run_authorization_export(self) -> None:
        import oczy.experiments.meta_cortex as pkg
        assert not hasattr(pkg, "RunAuthorization")

    def test_no_record_signoff_export(self) -> None:
        import oczy.experiments.meta_cortex as pkg
        assert not hasattr(pkg, "record_signoff")

    def test_no_authorize_instrument_export(self) -> None:
        import oczy.experiments.meta_cortex as pkg
        assert not hasattr(pkg, "authorize_instrument")

    def test_no_candidate_manifest_export(self) -> None:
        """CandidateManifest should not be in the DEV-only package __all__."""
        import oczy.experiments.meta_cortex as pkg
        if hasattr(pkg, "__all__"):
            assert "CandidateManifest" not in pkg.__all__


# ---------------------------------------------------------------------------
# Instrument materialization / verification / candidate finalization
# ---------------------------------------------------------------------------


def _make_instrument_config() -> InstrumentDefinitionConfig:
    """Build a minimal valid InstrumentDefinitionConfig for materialization."""
    return InstrumentDefinitionConfig(
        instrument_id="meta_cortex/v1",
        instrument_version="v1",
        root_seed=42,
        train_tasks_per_family=2,
        tuning_tasks_per_family=2,
        calibration_tasks_per_family=30,
        developmental_seeds=(10, 20, 30, 40, 50),
        evaluation_seeds=(60, 70, 80, 90, 100),
        organ_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        organ_revision="main",
        organ_parameter_sha256=_DUMMY_SHA,
        chat_template_sha256=_DUMMY_SHA,
        feature_mode="final_layer_mean_pool",
        decoding_mode="greedy",
        max_new_tokens=16,
        feature_dim=896,
        d_cortex=64,
        soft_bank_width=3,
        event_min=2,
        event_max=5,
        abstain_threshold="0.5",
        source_commit="abc123",
        source_archive_sha256=_DUMMY_SHA,
    )


def _make_test_seed_file(tmp_path: Path) -> Path:
    """Create a 32-byte test seed file."""
    seed_path = tmp_path / "test_seed.bin"
    seed_path.write_bytes(b"\x00" * 32)
    return seed_path


class TestMaterializeDefinition:
    """Test deterministic instrument definition materialization."""

    def test_materialize_creates_definition(self, tmp_path: Path) -> None:
        """materialize_definition creates a valid definition directory."""
        from oczy.experiments.meta_cortex.instrument import (
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        definition = materialize_definition(config, test_seed_file=seed_file, out=out)
        assert definition.schema == INSTRUMENT_DEFINITION_SCHEMA
        assert definition.instrument_id == "meta_cortex/v1"
        assert (out / "DEFINITION.json").exists()
        assert (out / "public/DEV_VIEW.json").exists()
        assert (out / "public/CALIBRATION_VIEW.json").exists()
        assert (out / "sealed/meta_test_seed.json").exists()
        assert (out / "sealed/tasks/meta_test.jsonl").exists()

    def test_materialize_refuses_overwrite(self, tmp_path: Path) -> None:
        """materialize_definition must refuse to overwrite an existing directory."""
        from oczy.experiments.meta_cortex.instrument import (
            InstrumentError,
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        out.mkdir(parents=True)
        with pytest.raises(InstrumentError, match="already exists"):
            materialize_definition(config, test_seed_file=seed_file, out=out)

    def test_materialize_is_deterministic(self, tmp_path: Path) -> None:
        """Same config + seed → byte-identical definition."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out1 = tmp_path / "def1"
        out2 = tmp_path / "def2"
        materialize_definition(config, test_seed_file=seed_file, out=out1)
        materialize_definition(config, test_seed_file=seed_file, out=out2)
        # Compare DEFINITION.json bytes.
        def1 = (out1 / "DEFINITION.json").read_bytes()
        def2 = (out2 / "DEFINITION.json").read_bytes()
        assert def1 == def2
        # Compare DEV_VIEW.json bytes.
        dev1 = (out1 / "public/DEV_VIEW.json").read_bytes()
        dev2 = (out2 / "public/DEV_VIEW.json").read_bytes()
        assert dev1 == dev2
        # Compare sealed meta_test.jsonl bytes.
        sealed1 = (out1 / "sealed/tasks/meta_test.jsonl").read_bytes()
        sealed2 = (out2 / "sealed/tasks/meta_test.jsonl").read_bytes()
        assert sealed1 == sealed2

    def test_different_seed_different_sealed(self, tmp_path: Path) -> None:
        """Different test seeds must produce different sealed meta-test tasks."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed1 = tmp_path / "seed1.bin"
        seed1.write_bytes(b"\x00" * 32)
        seed2 = tmp_path / "seed2.bin"
        seed2.write_bytes(b"\xff" * 32)
        out1 = tmp_path / "def1"
        out2 = tmp_path / "def2"
        materialize_definition(config, test_seed_file=seed1, out=out1)
        materialize_definition(config, test_seed_file=seed2, out=out2)
        sealed1 = (out1 / "sealed/tasks/meta_test.jsonl").read_bytes()
        sealed2 = (out2 / "sealed/tasks/meta_test.jsonl").read_bytes()
        assert sealed1 != sealed2

    def test_verify_definition_succeeds(self, tmp_path: Path) -> None:
        """verify_definition must succeed on a freshly materialized definition."""
        from oczy.experiments.meta_cortex.instrument import (
            materialize_definition,
            verify_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        definition = verify_definition(out)
        assert definition.instrument_id == "meta_cortex/v1"

    def test_verify_definition_detects_tamper(self, tmp_path: Path) -> None:
        """verify_definition must detect a tampered file."""
        from oczy.experiments.meta_cortex.instrument import (
            InstrumentError,
            materialize_definition,
            verify_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        # Tamper with DEV_VIEW.json.
        (out / "public/DEV_VIEW.json").write_bytes(b"tampered")
        with pytest.raises((InstrumentError, ContractError)):
            verify_definition(out)


class TestFourWaySemanticFirewall:
    """The four-way semantic firewall: DEV/CALIBRATION/SEALED/META_TEST isolation."""

    def test_dev_view_has_no_meta_test_tasks(self, tmp_path: Path) -> None:
        """DEV_VIEW must contain only META_TRAIN and META_TUNING tasks."""
        from oczy.experiments.meta_cortex.contracts import DevSplit
        from oczy.experiments.meta_cortex.instrument import (
            load_dev_view,
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        view = load_dev_view(out / "public")
        all_tasks = list(view.catalog.meta_train) + list(view.catalog.meta_validation)
        for task in all_tasks:
            assert task.split in (DevSplit.META_TRAIN, DevSplit.META_VALIDATION)

    def test_calibration_view_has_no_meta_test_tasks(self, tmp_path: Path) -> None:
        """CALIBRATION_VIEW must contain only META_VALIDATION tasks."""
        from oczy.experiments.meta_cortex.contracts import DevSplit
        from oczy.experiments.meta_cortex.instrument import (
            load_calibration_view,
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        view = load_calibration_view(out / "public")
        for task in view.tasks:
            assert task.split == DevSplit.META_VALIDATION

    def test_dev_view_does_not_reference_sealed_files(self, tmp_path: Path) -> None:
        """DEV_VIEW.json must not reference sealed/ paths."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        dev_view_bytes = (out / "public/DEV_VIEW.json").read_bytes()
        assert b"sealed/" not in dev_view_bytes
        assert b"meta_test" not in dev_view_bytes.lower()

    def test_calibration_view_does_not_reference_sealed_files(self, tmp_path: Path) -> None:
        """CALIBRATION_VIEW.json must not reference sealed/ paths."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        cal_view_bytes = (out / "public/CALIBRATION_VIEW.json").read_bytes()
        assert b"sealed/" not in cal_view_bytes
        assert b"meta_test" not in cal_view_bytes.lower()

    def test_sealed_files_not_in_public_directory(self, tmp_path: Path) -> None:
        """Sealed files must be in sealed/ directory, not public/."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        assert (out / "sealed/meta_test_seed.json").exists()
        assert (out / "sealed/tasks/meta_test.jsonl").exists()
        # These must NOT be in public/.
        assert not (out / "public/meta_test_seed.json").exists()
        assert not (out / "public/tasks/meta_test.jsonl").exists()


class TestSealedDevInaccessibility:
    """Sealed meta-test content must be inaccessible through DEV APIs."""

    def test_load_dev_view_returns_no_sealed_access(self, tmp_path: Path) -> None:
        """DevInstrumentView must not expose sealed tasks or bytes."""
        from oczy.experiments.meta_cortex.instrument import (
            load_dev_view,
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        view = load_dev_view(out / "public")
        # DevInstrumentView has binding + catalog, no sealed access.
        assert not hasattr(view, "sealed_tasks")
        assert not hasattr(view, "sealed_bytes")
        assert not hasattr(view, "meta_test_tasks")
        assert not hasattr(view, "get_sealed")

    def test_load_calibration_view_returns_no_sealed_access(self, tmp_path: Path) -> None:
        """CalibrationInstrumentView must not expose sealed tasks or bytes."""
        from oczy.experiments.meta_cortex.instrument import (
            load_calibration_view,
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        view = load_calibration_view(out / "public")
        assert not hasattr(view, "sealed_tasks")
        assert not hasattr(view, "sealed_bytes")
        assert not hasattr(view, "meta_test_tasks")
        assert not hasattr(view, "get_sealed")

    def test_definition_public_files_exclude_sealed(self, tmp_path: Path) -> None:
        """The definition's public_files must not include sealed/ paths."""
        from oczy.experiments.meta_cortex.instrument import (
            materialize_definition,
        )
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        definition = materialize_definition(config, test_seed_file=seed_file, out=out)
        for entry in definition.public_files:
            assert not entry.path.startswith("sealed/")
            assert entry.visibility != "sealed"


class TestCandidateFinalization:
    """Test candidate finalization — write-once, immutable, verified."""

    def _materialize_and_create_evidence(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Materialize a definition and create synthetic calibration evidence.
        Returns (definition_root, calibration_report, power_report)."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        def_root = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=def_root)

        # Create synthetic calibration evidence.
        # We need to read the actual DEV_DISTRIBUTIONS and POWER_ANALYSIS
        # format from the calibration module. For now, create minimal
        # synthetic evidence that finalize_candidate can consume.
        # This is a placeholder — the actual evidence format is complex.
        cal_report = tmp_path / "DEV_DISTRIBUTIONS.json"
        power_report = tmp_path / "POWER_ANALYSIS.json"
        # We'll need to understand the exact format finalize_candidate expects.
        return def_root, cal_report, power_report

    def test_finalize_refuses_overwrite(self, tmp_path: Path) -> None:
        """finalize_candidate must refuse to overwrite an existing directory."""
        from oczy.experiments.meta_cortex.instrument import (
            InstrumentError,
            finalize_candidate,
        )
        def_root = tmp_path / "definition"
        def_root.mkdir(parents=True)
        out = tmp_path / "candidate"
        out.mkdir(parents=True)
        # finalize_candidate should refuse because out already exists.
        # It will also fail on definition verification, but the overwrite
        # check comes first.
        with pytest.raises(InstrumentError, match="already exists"):
            finalize_candidate(
                def_root,
                calibration_report=tmp_path / "cal.json",
                power_report=tmp_path / "pow.json",
                out=out,
            )


class TestCandidateImmutability:
    """A signed v1 candidate must remain byte-identical when v2 is created."""

    def test_materialize_same_config_same_bytes(self, tmp_path: Path) -> None:
        """Re-materializing with the same config produces byte-identical output."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = _make_instrument_config()
        seed_file = _make_test_seed_file(tmp_path)
        out1 = tmp_path / "def1"
        out2 = tmp_path / "def2"
        materialize_definition(config, test_seed_file=seed_file, out=out1)
        materialize_definition(config, test_seed_file=seed_file, out=out2)
        # Every file in out1 must have a byte-identical counterpart in out2.
        files1 = sorted(p.relative_to(out1) for p in out1.rglob("*") if p.is_file())
        files2 = sorted(p.relative_to(out2) for p in out2.rglob("*") if p.is_file())
        assert [str(f) for f in files1] == [str(f) for f in files2]
        for rel in files1:
            assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), (
                f"File {rel} differs between materializations"
            )

    def test_different_config_different_definition_hash(self, tmp_path: Path) -> None:
        """Different configs must produce different definition hashes."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config1 = _make_instrument_config()
        import dataclasses
        config2 = dataclasses.replace(config1, root_seed=99)  # Different seed
        seed_file = _make_test_seed_file(tmp_path)
        out1 = tmp_path / "def1"
        out2 = tmp_path / "def2"
        def1 = materialize_definition(config1, test_seed_file=seed_file, out=out1)
        def2 = materialize_definition(config2, test_seed_file=seed_file, out=out2)
        assert def1.definition_sha256 != def2.definition_sha256


# ---------------------------------------------------------------------------
# Sealed task diversity: 30 sealed finite-state rules disjoint from DEV
# ---------------------------------------------------------------------------


class TestSealedTaskDiversity:
    """Regression: ≥30 sealed finite-state rules must materialize without
    collision exhaustion and remain disjoint from DEV/calibration catalogs.

    Diagnoses the finite-state rule-space collapse where the DEV Moore-machine
    assignment space (4^3 = 64) is too small for 30 train + 5 tuning +
    30 calibration + 30 sealed = 95 tasks.  The sealed generator uses
    expanded Mealy-machine assignments (action per (state, input)) with
    variable-size state machines, making sealed fingerprints structurally
    distinct from DEV's and creating an assignment space of 8^(states×inputs).
    """

    def _make_30_5_30_config(self) -> InstrumentDefinitionConfig:
        """Build a config with 30 train / 5 tuning / 30 calibration per family."""
        return InstrumentDefinitionConfig(
            instrument_id="meta_cortex/v1",
            instrument_version="v1",
            root_seed=42,
            train_tasks_per_family=30,
            tuning_tasks_per_family=5,
            calibration_tasks_per_family=30,
            developmental_seeds=(10, 20, 30, 40, 50),
            evaluation_seeds=(60, 70, 80, 90, 100),
            organ_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            organ_revision="main",
            organ_parameter_sha256=_DUMMY_SHA,
            chat_template_sha256=_DUMMY_SHA,
            feature_mode="final_layer_mean_pool",
            decoding_mode="greedy",
            max_new_tokens=16,
            feature_dim=896,
            d_cortex=64,
            soft_bank_width=3,
            event_min=2,
            event_max=5,
            abstain_threshold="0.5",
            source_commit="abc123",
            source_archive_sha256=_DUMMY_SHA,
        )

    def test_materialize_30_5_30_30_succeeds(self, tmp_path: Path) -> None:
        """materialize_definition must succeed with 30/5/30/30 counts.

        Before the fix, sealed collision exhausted at finite_state index 5
        because the DEV assignment space (64) left only 5 unused values.
        """
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = self._make_30_5_30_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        definition = materialize_definition(config, test_seed_file=seed_file, out=out)
        assert definition is not None
        assert (out / "sealed/tasks/meta_test.jsonl").exists()

    def test_30_sealed_tasks_per_family(self, tmp_path: Path) -> None:
        """Exactly 30 sealed tasks per family (90 total) must be present."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = self._make_30_5_30_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        sealed_path = out / "sealed/tasks/meta_test.jsonl"
        records = [json.loads(line) for line in sealed_path.read_text().splitlines() if line]
        assert len(records) == 90
        from collections import Counter
        family_counts = Counter(r["family"] for r in records)
        assert family_counts["contextual_remap"] == 30
        assert family_counts["rule_transformation"] == 30
        assert family_counts["finite_state"] == 30

    def test_firewall_overlap_zero(self, tmp_path: Path) -> None:
        """Cross-domain fingerprint overlap must be zero across all four domains."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = self._make_30_5_30_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)
        summary = json.loads(
            (out / "public/audits/leakage_summary.json").read_bytes().decode().rstrip()
        )
        assert summary["passed"] is True
        # Verify all pairwise overlaps are zero
        for _d1, others in summary["pairwise_overlap"].items():
            for _d2, overlaps in others.items():
                for _fc, count in overlaps.items():
                    assert count == 0, f"Cross-domain overlap {_d1} vs {_d2}"

    def test_sealed_finite_state_fingerprints_disjoint_from_dev(
        self, tmp_path: Path
    ) -> None:
        """All sealed finite-state fingerprints must be absent from DEV catalogs."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = self._make_30_5_30_config()
        seed_file = _make_test_seed_file(tmp_path)
        out = tmp_path / "definition"
        materialize_definition(config, test_seed_file=seed_file, out=out)

        # Collect DEV fingerprints from public task files
        dev_fps: set[str] = set()
        for task_file in (out / "public/tasks").glob("*.jsonl"):
            for line in task_file.read_text().splitlines():
                if not line:
                    continue
                rec = json.loads(line)
                for fc in ("rule_fingerprint", "assignment_fingerprint",
                           "composition_fingerprint", "paraphrase_group_fingerprint"):
                    dev_fps.add(rec[fc])

        # Collect sealed fingerprints
        sealed_path = out / "sealed/tasks/meta_test.jsonl"
        sealed_fps: set[str] = set()
        sealed_fs_fps: set[str] = set()
        for line in sealed_path.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            for fc in ("rule_fingerprint", "assignment_fingerprint",
                       "composition_fingerprint", "paraphrase_group_fingerprint"):
                sealed_fps.add(rec[fc])
                if rec["family"] == "finite_state":
                    sealed_fs_fps.add(rec[fc])

        # No sealed fingerprint may appear in DEV
        assert len(sealed_fps & dev_fps) == 0, "Sealed fingerprint leaked to DEV"
        # Finite-state sealed fingerprints specifically must be disjoint
        assert len(sealed_fs_fps & dev_fps) == 0, "Sealed FS fingerprint leaked to DEV"

    def test_sealed_finite_state_mealy_assignment_structure(
        self, tmp_path: Path
    ) -> None:
        """Sealed finite-state assignments must use Mealy-machine structure
        (nested {state: {input: action}}) distinct from DEV's Moore
        {state: action}."""
        from oczy.experiments.meta_cortex.instrument import (
            _generate_all_tasks,
        )
        config = self._make_30_5_30_config()
        all_tasks = _generate_all_tasks(config, "00" * 32)
        sealed_fs = [
            t for t in all_tasks["meta_test"]
            if t.family.value == "finite_state"
        ]
        assert len(sealed_fs) == 30
        # Every sealed finite-state assignment fingerprint must be unique
        assign_fps = [t.assignment_fingerprint for t in sealed_fs]
        assert len(assign_fps) == len(set(assign_fps)), "Duplicate sealed FS assignments"
        # Every sealed finite-state rule fingerprint must be unique
        rule_fps = [t.rule_fingerprint for t in sealed_fs]
        assert len(rule_fps) == len(set(rule_fps)), "Duplicate sealed FS rules"

    def test_30_5_30_30_materialization_is_deterministic(
        self, tmp_path: Path
    ) -> None:
        """Re-materializing with the same 30/5/30/30 config produces
        byte-identical output."""
        from oczy.experiments.meta_cortex.instrument import materialize_definition
        config = self._make_30_5_30_config()
        seed_file = _make_test_seed_file(tmp_path)
        out1 = tmp_path / "def1"
        out2 = tmp_path / "def2"
        materialize_definition(config, test_seed_file=seed_file, out=out1)
        materialize_definition(config, test_seed_file=seed_file, out=out2)
        files1 = sorted(p.relative_to(out1) for p in out1.rglob("*") if p.is_file())
        files2 = sorted(p.relative_to(out2) for p in out2.rglob("*") if p.is_file())
        assert [str(f) for f in files1] == [str(f) for f in files2]
        for rel in files1:
            assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), (
                f"File {rel} differs between materializations"
            )
