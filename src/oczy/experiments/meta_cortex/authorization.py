"""Detached human signoff and exact authorization gate for meta_cortex/v2.

This module implements the strict candidate-manifest schema/hash verification
and detached ``SIGNOFF.json`` exact-tuple attestation per the R20 signoff plan.

Key invariants:

- The candidate manifest remains byte-identical; signing **never** mutates it.
- ``SIGNOFF.json`` is a detached, write-once record that binds the exact tuple
  ``(instrument_manifest_sha256, equivalence_margin,
  sample_size_tasks_per_family, human_signoff_id)``.
- The gate runs **before** any sealed bytes, model, or output access.
- No environment bypass, no ``--force``, no defaults, no aliases.
- Authorization is three-way exact equality: manifest, signoff, and externally
  supplied run values must all match.
- The signoff ID is public provenance (``r20-meta-cortex-v<version>/<UUIDv4>``),
  not a secret or bearer token.

This module owns ``RunAuthorization``, ``AuthorizedInstrument``, and the gate
functions.  It imports shared instrument types (``CandidateManifest``,
``SignoffAttestation``, ``InstrumentFileEntry``, strict canonical JSON
helpers) from :mod:`oczy.experiments.meta_cortex.instrument_contracts`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instrument_contracts import (
    CANDIDATE_MANIFEST_SCHEMA,
    SIGNOFF_SCHEMA,
    CandidateManifest,
    ContractError,
    InstrumentFileEntry,
    SignoffAttestation,
    canonical_decimal,
    strict_canonical_json,
    strict_json_loads,
    validate_relative_path,
    validate_sha256_hex,
    validate_signoff_id,
)

__all__ = [
    # Schema constants (re-exported for convenience)
    "SIGNOFF_SCHEMA",
    # Exceptions
    "SignoffError",
    "InstrumentIntegrityError",
    # Frozen types
    "RunAuthorization",
    "AuthorizedInstrument",
    # Gate functions
    "record_signoff",
    "load_signoff",
    "verify_run_authorization",
    "authorize_instrument",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SignoffError(ValueError):
    """Raised on signoff validation, authorization, or attestation failure."""


class InstrumentIntegrityError(ValueError):
    """Raised when instrument manifest or file integrity verification fails."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAMILY_KEYS = ("contextual_remap", "rule_transformation", "finite_state")


# ---------------------------------------------------------------------------
# Frozen types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunAuthorization:
    """The exact authorization tuple that a signed run must supply.

    This is the runtime-facing authorization value.  Every field must match
    exactly across three sources: the candidate manifest, the detached
    signoff, and the externally supplied run values.

    ``signoff_sha256`` pins the signoff record itself.
    """

    instrument_manifest_sha256: str
    equivalence_margin: str
    sample_size_tasks_per_family: int
    human_signoff_id: str
    signoff_sha256: str

    def __post_init__(self) -> None:
        validate_sha256_hex(self.instrument_manifest_sha256, field_name="instrument_manifest_sha256")
        try:
            canonical = canonical_decimal(self.equivalence_margin)
        except ContractError as exc:
            raise SignoffError(str(exc)) from exc
        if canonical != self.equivalence_margin:
            raise SignoffError(
                f"equivalence_margin must be canonical, got {self.equivalence_margin!r} "
                f"(canonical: {canonical!r})"
            )
        if not isinstance(self.sample_size_tasks_per_family, int) or isinstance(
            self.sample_size_tasks_per_family, bool
        ):
            raise SignoffError("sample_size_tasks_per_family must be an int, not bool")
        if self.sample_size_tasks_per_family < 30:
            raise SignoffError(
                f"sample_size_tasks_per_family must be >= 30, "
                f"got {self.sample_size_tasks_per_family}"
            )
        try:
            validate_signoff_id(self.human_signoff_id)
        except ContractError as exc:
            raise SignoffError(str(exc)) from exc
        validate_sha256_hex(self.signoff_sha256, field_name="signoff_sha256")


class AuthorizedInstrument:
    """Immutable capability granted after all authorization checks pass.

    This type is not exported from the package ``__init__``.  It exposes
    read-only metadata only — no sealed-task iterator, no sealed bytes, no
    model loader.  Sealed access is in the separate ``run_meta_test.py``
    runner.

    The constructor is private; instances are created only by
    :func:`authorize_instrument`.
    """

    __slots__ = (
        "_manifest",
        "_signoff",
        "_authorization",
    )

    _manifest: CandidateManifest
    _signoff: SignoffAttestation
    _authorization: RunAuthorization

    def __init__(
        self,
        manifest: CandidateManifest,
        signoff: SignoffAttestation,
        authorization: RunAuthorization,
    ) -> None:
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_signoff", signoff)
        object.__setattr__(self, "_authorization", authorization)

    @property
    def manifest(self) -> CandidateManifest:
        return self._manifest

    @property
    def signoff(self) -> SignoffAttestation:
        return self._signoff

    @property
    def authorization(self) -> RunAuthorization:
        return self._authorization

    def __repr__(self) -> str:
        return (
            f"AuthorizedInstrument("
            f"manifest_sha256={self._manifest.manifest_sha256[:12]}..., "
            f"signoff_sha256={self._signoff.signoff_sha256[:12]}...)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_count_map(count_map: dict[str, int], expected_n: int) -> None:
    """Validate that count_map has exactly the three families all equal to expected_n."""
    if not isinstance(count_map, dict):
        raise SignoffError("meta_test_tasks_by_family must be a dict")
    if set(count_map.keys()) != set(_FAMILY_KEYS):
        raise SignoffError(
            f"meta_test_tasks_by_family must have exactly keys "
            f"{list(_FAMILY_KEYS)}, got {sorted(count_map.keys())}"
        )
    for key in _FAMILY_KEYS:
        val = count_map[key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise SignoffError(
                f"meta_test_tasks_by_family['{key}'] must be an int, not bool"
            )
        if val != expected_n:
            raise SignoffError(
                f"meta_test_tasks_by_family['{key}'] must equal "
                f"sample_size_tasks_per_family ({expected_n}), got {val}"
            )


def _compute_self_hash(obj_dict: dict[str, Any], self_hash_field: str) -> str:
    """Compute SHA-256 of canonical encoding with the self-hash field omitted."""
    payload = {k: v for k, v in obj_dict.items() if k != self_hash_field}
    return hashlib.sha256(strict_canonical_json(payload)).hexdigest()


def _read_and_parse_manifest(manifest_path: Path) -> CandidateManifest:
    """Read and strictly parse a candidate manifest JSON file.

    Validates schema, lifecycle_state, self-hash, and all field types.
    Does NOT open any sealed files.
    """
    if not manifest_path.is_file():
        raise InstrumentIntegrityError(f"Manifest not found: {manifest_path}")
    raw_bytes = manifest_path.read_bytes()
    try:
        data = strict_json_loads(raw_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InstrumentIntegrityError(
            f"Manifest is not valid strict JSON: {exc}"
        ) from exc

    # Validate schema.
    schema = data.get("schema")
    if schema != CANDIDATE_MANIFEST_SCHEMA:
        raise InstrumentIntegrityError(
            f"Manifest schema must be {CANDIDATE_MANIFEST_SCHEMA!r}, got {schema!r}"
        )

    # Validate lifecycle_state.
    lifecycle_state = data.get("lifecycle_state")
    if lifecycle_state != "candidate":
        raise InstrumentIntegrityError(
            f"Manifest lifecycle_state must be 'candidate', got {lifecycle_state!r}. "
            f"A manifest edited to 'signed' is invalid — signoff is detached."
        )

    # Validate and verify self-hash.
    stored_hash = data.get("manifest_sha256")
    if not isinstance(stored_hash, str) or not stored_hash:
        raise InstrumentIntegrityError("Manifest missing manifest_sha256 field")
    validate_sha256_hex(stored_hash, field_name="manifest_sha256")
    computed_hash = _compute_self_hash(data, "manifest_sha256")
    if computed_hash != stored_hash:
        raise InstrumentIntegrityError(
            f"Manifest self-hash mismatch: stored {stored_hash!r}, "
            f"computed {computed_hash!r}"
        )

    # Verify on-disk bytes match canonical encoding (allow trailing newline).
    canonical_bytes = strict_canonical_json(data)
    if raw_bytes != canonical_bytes and raw_bytes != canonical_bytes + b"\n":
        raise InstrumentIntegrityError(
            "Manifest on-disk bytes do not match canonical encoding; "
            "whitespace/duplicate-key variants are rejected"
        )

    # Reconstruct CandidateManifest from the dict.
    return _candidate_manifest_from_dict(data)


def _candidate_manifest_from_dict(data: dict[str, Any]) -> CandidateManifest:
    """Reconstruct a CandidateManifest from a validated JSON dict."""
    files_data = data.get("files", [])
    if not isinstance(files_data, list):
        raise InstrumentIntegrityError("Manifest files must be a list")
    files = tuple(
        InstrumentFileEntry(
            path=entry["path"],
            sha256=entry["sha256"],
            size_bytes=entry["size_bytes"],
            visibility=entry["visibility"],
            role=entry["role"],
        )
        for entry in files_data
    )
    try:
        return CandidateManifest(
            schema=data["schema"],
            instrument_id=data["instrument_id"],
            instrument_version=data["instrument_version"],
            lifecycle_state=data["lifecycle_state"],
            definition_sha256=data["definition_sha256"],
            dev_view_sha256=data["dev_view_sha256"],
            calibration_view_sha256=data["calibration_view_sha256"],
            source_commit=data["source_commit"],
            source_archive_sha256=data["source_archive_sha256"],
            taskgen_schema=data["taskgen_schema"],
            generator_algorithm=data["generator_algorithm"],
            generator_source_sha256=data["generator_source_sha256"],
            prompt_schema=data["prompt_schema"],
            prompt_registry_sha256=data["prompt_registry_sha256"],
            scorer_schema=data["scorer_schema"],
            scorer_registry_sha256=data["scorer_registry_sha256"],
            endpoint_schema=data["endpoint_schema"],
            endpoint_registry_sha256=data["endpoint_registry_sha256"],
            organ_model_id=data["organ_model_id"],
            organ_revision=data["organ_revision"],
            organ_parameter_sha256=data["organ_parameter_sha256"],
            chat_template_sha256=data["chat_template_sha256"],
            feature_mode=data["feature_mode"],
            decoding_mode=data["decoding_mode"],
            max_new_tokens=data["max_new_tokens"],
            feature_dim=data["feature_dim"],
            d_cortex=data["d_cortex"],
            soft_bank_width=data["soft_bank_width"],
            abstain_threshold=data["abstain_threshold"],
            event_min=data["event_min"],
            event_max=data["event_max"],
            family_order=tuple(data["family_order"]),
            probe_counts_by_split_family_kind=data["probe_counts_by_split_family_kind"],
            train_tasks_per_family=data["train_tasks_per_family"],
            tuning_tasks_per_family=data["tuning_tasks_per_family"],
            calibration_tasks_per_family=data["calibration_tasks_per_family"],
            sample_size_tasks_per_family=data["sample_size_tasks_per_family"],
            meta_test_tasks_by_family=dict(data["meta_test_tasks_by_family"]),
            developmental_seeds=tuple(data["developmental_seeds"]),
            evaluation_seeds=tuple(data["evaluation_seeds"]),
            equivalence_margin=data["equivalence_margin"],
            calibration_report_sha256=data["calibration_report_sha256"],
            power_report_sha256=data["power_report_sha256"],
            calibration_holdout_accessed=data["calibration_holdout_accessed"],
            independent_sample_unit=data["independent_sample_unit"],
            leakage_audit_sha256=data["leakage_audit_sha256"],
            leakage_audit_passed=data["leakage_audit_passed"],
            meta_test_seed_sha256=data["meta_test_seed_sha256"],
            files=files,
            manifest_sha256=data["manifest_sha256"],
        )
    except ContractError as exc:
        raise InstrumentIntegrityError(f"Manifest field validation failed: {exc}") from exc


def _read_and_parse_signoff(signoff_path: Path) -> SignoffAttestation:
    """Read and strictly parse a detached signoff JSON file.

    Validates schema, lifecycle_state, self-hash, UUIDv4, and all field types.
    """
    if not signoff_path.is_file():
        raise SignoffError(f"Signoff file not found: {signoff_path}")
    raw_bytes = signoff_path.read_bytes()
    try:
        data = strict_json_loads(raw_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SignoffError(f"Signoff is not valid strict JSON: {exc}") from exc

    # Validate schema.
    schema = data.get("schema")
    if schema != SIGNOFF_SCHEMA:
        raise SignoffError(
            f"Signoff schema must be {SIGNOFF_SCHEMA!r}, got {schema!r}"
        )

    # Validate lifecycle_state.
    lifecycle_state = data.get("lifecycle_state")
    if lifecycle_state != "signed":
        raise SignoffError(
            f"Signoff lifecycle_state must be 'signed', got {lifecycle_state!r}"
        )

    # Validate and verify self-hash.
    stored_hash = data.get("signoff_sha256")
    if not isinstance(stored_hash, str) or not stored_hash:
        raise SignoffError("Signoff missing signoff_sha256 field")
    validate_sha256_hex(stored_hash, field_name="signoff_sha256")
    computed_hash = _compute_self_hash(data, "signoff_sha256")
    if computed_hash != stored_hash:
        raise SignoffError(
            f"Signoff self-hash mismatch: stored {stored_hash!r}, "
            f"computed {computed_hash!r}"
        )

    # Verify on-disk bytes match canonical encoding (allow trailing newline).
    canonical_bytes = strict_canonical_json(data)
    if raw_bytes != canonical_bytes and raw_bytes != canonical_bytes + b"\n":
        raise SignoffError(
            "Signoff on-disk bytes do not match canonical encoding; "
            "whitespace/duplicate-key variants are rejected"
        )

    # Reconstruct SignoffAttestation.
    try:
        return SignoffAttestation(
            schema=data["schema"],
            instrument_id=data["instrument_id"],
            instrument_version=data["instrument_version"],
            lifecycle_state=data["lifecycle_state"],
            instrument_manifest_sha256=data["instrument_manifest_sha256"],
            equivalence_margin=data["equivalence_margin"],
            sample_size_tasks_per_family=data["sample_size_tasks_per_family"],
            meta_test_tasks_by_family=dict(data["meta_test_tasks_by_family"]),
            human_signoff_id=data["human_signoff_id"],
            signed_at_utc=data["signed_at_utc"],
            signoff_sha256=data["signoff_sha256"],
        )
    except ContractError as exc:
        raise SignoffError(f"Signoff field validation failed: {exc}") from exc


def _verify_file_hashes(
    root: Path, files: tuple[InstrumentFileEntry, ...]
) -> None:
    """Verify SHA-256 and size of every listed file against actual bytes.

    Only verifies files that are listed in the manifest's file entries.
    Does NOT open sealed files unless they are listed and authorization
    has already succeeded.
    """
    seen_paths: set[str] = set()
    root_resolved = root.resolve(strict=True)
    for entry in files:
        # Reject duplicate paths.
        if entry.path in seen_paths:
            raise InstrumentIntegrityError(
                f"Duplicate file path in manifest: {entry.path!r}"
            )
        seen_paths.add(entry.path)

        # Validate relative path safety.
        validate_relative_path(entry.path)

        # Resolve beneath root.
        file_path = root / entry.path
        try:
            resolved = file_path.resolve(strict=True)
        except OSError as exc:
            raise InstrumentIntegrityError(
                f"Cannot resolve file {entry.path!r}: {exc}"
            ) from exc
        # Ensure resolved path is under root.
        if not str(resolved).startswith(str(root_resolved) + os.sep):
            raise InstrumentIntegrityError(
                f"File {entry.path!r} resolves outside instrument root"
            )
        # Reject symlinks.
        if file_path.is_symlink():
            raise InstrumentIntegrityError(
                f"File {entry.path!r} is a symlink — symlinks are rejected"
            )
        if not resolved.is_file():
            raise InstrumentIntegrityError(
                f"File {entry.path!r} is not a regular file"
            )

        # Read and hash.
        raw = resolved.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != entry.sha256:
            raise InstrumentIntegrityError(
                f"Hash mismatch for {entry.path!r}: "
                f"expected {entry.sha256}, got {actual_hash}"
            )
        if len(raw) != entry.size_bytes:
            raise InstrumentIntegrityError(
                f"Size mismatch for {entry.path!r}: "
                f"expected {entry.size_bytes}, got {len(raw)}"
            )


def _check_no_env_bypass() -> None:
    """Reject any environment variable that could bypass authorization.

    No env bypass is permitted.  If any plausible approval/signoff env var
    is set, it has no effect — the gate still requires the exact tuple.
    This function exists to document that env vars are never checked for
    bypass purposes.
    """
    # Deliberately empty: no environment variable can bypass the gate.
    # EVAL_CHANGE_APPROVED, R20_SIGNOFF_APPROVED, etc. are all ignored.
    pass


def _validate_signoff_id_with_version(
    signoff_id: str, expected_version: str
) -> None:
    """Validate signoff ID and check version segment matches expected_version.

    Args:
        signoff_id: The ID to validate.
        expected_version: Instrument version like "v1"; the version segment
            in the ID (e.g. "1") must match.
    """
    try:
        validate_signoff_id(signoff_id)
    except ContractError as exc:
        raise SignoffError(str(exc)) from exc
    # Extract version segment and compare.
    # ID format: r20-meta-cortex-v<version>/<uuid>
    prefix = "r20-meta-cortex-v"
    rest = signoff_id[len(prefix):]
    version_str = rest.split("/")[0]
    expected_num = expected_version.lstrip("v")
    if version_str != expected_num:
        raise SignoffError(
            f"human_signoff_id version segment v{version_str} does not "
            f"match expected instrument version {expected_version}"
        )


# ---------------------------------------------------------------------------
# Public gate functions
# ---------------------------------------------------------------------------


def record_signoff(
    candidate_root: Path,
    destination: Path,
    *,
    expected_manifest_sha256: str,
    expected_equivalence_margin: str,
    expected_sample_size_tasks_per_family: int,
    human_signoff_id: str,
) -> SignoffAttestation:
    """Record a detached human signoff, binding the exact authorization tuple.

    This is the sole signoff creation path.  It is write-once: if the
    destination already exists, it is rejected.  No ``--force``, no
    environment bypass, no defaults.

    The candidate manifest is read and verified but **never mutated**.
    The signoff is written as a separate ``SIGNOFF.json`` file.

    Args:
        candidate_root: Root directory containing ``MANIFEST.json`` (or
            ``CANDIDATE_MANIFEST.json``) and the instrument files.
        destination: Path where ``SIGNOFF.json`` will be written (write-once).
        expected_manifest_sha256: The exact manifest SHA-256 the reviewer
            is approving.  Must match the candidate manifest's self-hash.
        expected_equivalence_margin: The exact canonical decimal margin
            the reviewer is approving.
        expected_sample_size_tasks_per_family: The exact integer N the
            reviewer is approving.
        human_signoff_id: Structured ``r20-meta-cortex-v<version>/<UUIDv4>``.

    Returns:
        The validated :class:`SignoffAttestation`.

    Raises:
        SignoffError: On any mismatch, invalid ID, or write-once violation.
        InstrumentIntegrityError: On manifest corruption.
    """
    _check_no_env_bypass()

    # Validate expected values upfront.
    validate_sha256_hex(expected_manifest_sha256, field_name="expected_manifest_sha256")
    canonical_decimal(expected_equivalence_margin)
    if not isinstance(expected_sample_size_tasks_per_family, int) or isinstance(
        expected_sample_size_tasks_per_family, bool
    ):
        raise SignoffError("expected_sample_size_tasks_per_family must be an int, not bool")
    if expected_sample_size_tasks_per_family < 30:
        raise SignoffError(
            f"expected_sample_size_tasks_per_family must be >= 30, "
            f"got {expected_sample_size_tasks_per_family}"
        )

    # Read and verify the candidate manifest.
    manifest_path = candidate_root / "MANIFEST.json"
    if not manifest_path.is_file():
        manifest_path = candidate_root / "CANDIDATE_MANIFEST.json"
    if not manifest_path.is_file():
        raise InstrumentIntegrityError(
            f"No MANIFEST.json or CANDIDATE_MANIFEST.json in {candidate_root}"
        )
    manifest = _read_and_parse_manifest(manifest_path)

    # Compare expected manifest hash to the manifest's self-hash.
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise SignoffError(
            f"Expected manifest SHA-256 {expected_manifest_sha256!r} does not "
            f"match candidate manifest hash {manifest.manifest_sha256!r}"
        )

    # Compare expected margin and N to the manifest's values.
    if manifest.equivalence_margin != expected_equivalence_margin:
        raise SignoffError(
            f"Expected equivalence_margin {expected_equivalence_margin!r} does not "
            f"match manifest value {manifest.equivalence_margin!r}"
        )
    if manifest.sample_size_tasks_per_family != expected_sample_size_tasks_per_family:
        raise SignoffError(
            f"Expected sample_size_tasks_per_family "
            f"{expected_sample_size_tasks_per_family} does not match "
            f"manifest value {manifest.sample_size_tasks_per_family}"
        )

    # Validate signoff ID version matches manifest version.
    _validate_signoff_id_with_version(human_signoff_id, manifest.instrument_version)

    # Build the signoff record.
    signed_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017 -- Python 3.10
    signoff_dict: dict[str, Any] = {
        "schema": SIGNOFF_SCHEMA,
        "instrument_id": manifest.instrument_id,
        "instrument_version": manifest.instrument_version,
        "lifecycle_state": "signed",
        "instrument_manifest_sha256": manifest.manifest_sha256,
        "equivalence_margin": manifest.equivalence_margin,
        "sample_size_tasks_per_family": manifest.sample_size_tasks_per_family,
        "meta_test_tasks_by_family": dict(manifest.meta_test_tasks_by_family),
        "human_signoff_id": human_signoff_id,
        "signed_at_utc": signed_at_utc,
    }
    # Compute self-hash.
    signoff_hash = _compute_self_hash(signoff_dict, "signoff_sha256")
    signoff_dict["signoff_sha256"] = signoff_hash

    # Construct and validate the attestation.
    try:
        attestation = SignoffAttestation(
            schema=signoff_dict["schema"],
            instrument_id=signoff_dict["instrument_id"],
            instrument_version=signoff_dict["instrument_version"],
            lifecycle_state=signoff_dict["lifecycle_state"],
            instrument_manifest_sha256=signoff_dict["instrument_manifest_sha256"],
            equivalence_margin=signoff_dict["equivalence_margin"],
            sample_size_tasks_per_family=signoff_dict["sample_size_tasks_per_family"],
            meta_test_tasks_by_family=signoff_dict["meta_test_tasks_by_family"],
            human_signoff_id=signoff_dict["human_signoff_id"],
            signed_at_utc=signoff_dict["signed_at_utc"],
            signoff_sha256=signoff_hash,
        )
    except ContractError as exc:
        raise SignoffError(f"Signoff attestation validation failed: {exc}") from exc

    # Write-once: refuse if destination already exists.
    destination = Path(destination)
    if destination.exists():
        raise SignoffError(
            f"Signoff destination already exists: {destination}. "
            f"Signoff is write-once — use a new version for changes."
        )

    # Write canonical JSON atomically with exclusive-create.
    canonical_bytes = strict_canonical_json(signoff_dict) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Use exclusive create to prevent races.
    fd = os.open(
        str(destination),
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(fd, canonical_bytes)
        os.fsync(fd)
    finally:
        os.close(fd)

    # Fsync the directory.
    dir_fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    return attestation


def load_signoff(path: Path) -> SignoffAttestation:
    """Load and strictly validate a detached signoff JSON file.

    Validates schema, lifecycle_state, self-hash, UUIDv4 format, and all
    field types.  Does NOT compare to any manifest — that is done in
    :func:`verify_run_authorization`.

    Args:
        path: Path to ``SIGNOFF.json``.

    Returns:
        The validated :class:`SignoffAttestation`.

    Raises:
        SignoffError: On any validation failure.
    """
    _check_no_env_bypass()
    return _read_and_parse_signoff(Path(path))


def verify_run_authorization(
    manifest_path: Path,
    signoff_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_equivalence_margin: str,
    expected_sample_size_tasks_per_family: int,
    expected_human_signoff_id: str,
) -> RunAuthorization:
    """Verify three-way exact tuple match across manifest, signoff, and run values.

    This is the gate that must run **before** any sealed bytes, model, or
    output access.  It performs strict validation of both the candidate
    manifest and the detached signoff, then compares every externally
    supplied value exactly to both records.

    No environment bypass.  No tolerance.  No coercion.  Exact, case-sensitive
    equality on all fields.

    Gate order:
    1. Validate externally supplied expected values.
    2. Read/validate manifest (no sealed access).
    3. Read/validate signoff.
    4. Three-way exact comparison of manifest, signoff, and expected values.
    5. Verify schema/state/scope/version, self-hashes, count-map invariant.
    6. Return :class:`RunAuthorization` only if all checks pass.

    Args:
        manifest_path: Path to ``MANIFEST.json`` or ``CANDIDATE_MANIFEST.json``.
        signoff_path: Path to ``SIGNOFF.json``.
        expected_manifest_sha256: Externally supplied manifest SHA-256.
        expected_equivalence_margin: Externally supplied canonical decimal margin.
        expected_sample_size_tasks_per_family: Externally supplied integer N.
        expected_human_signoff_id: Externally supplied structured signoff ID.

    Returns:
        A :class:`RunAuthorization` if all checks pass.

    Raises:
        SignoffError: On any mismatch or validation failure.
        InstrumentIntegrityError: On manifest corruption.
    """
    _check_no_env_bypass()

    # Step 1: Validate externally supplied values upfront.
    validate_sha256_hex(expected_manifest_sha256, field_name="expected_manifest_sha256")
    canonical_decimal(expected_equivalence_margin)
    if not isinstance(expected_sample_size_tasks_per_family, int) or isinstance(
        expected_sample_size_tasks_per_family, bool
    ):
        raise SignoffError("expected_sample_size_tasks_per_family must be an int, not bool")
    if expected_sample_size_tasks_per_family < 30:
        raise SignoffError(
            f"expected_sample_size_tasks_per_family must be >= 30, "
            f"got {expected_sample_size_tasks_per_family}"
        )
    try:
        validate_signoff_id(expected_human_signoff_id)
    except ContractError as exc:
        raise SignoffError(str(exc)) from exc

    # Step 2: Read and validate manifest (no sealed access).
    manifest = _read_and_parse_manifest(Path(manifest_path))

    # Step 3: Read and validate signoff.
    signoff = _read_and_parse_signoff(Path(signoff_path))

    # Step 4: Three-way exact comparison.
    # --- Manifest hash ---
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise SignoffError(
            f"manifest hash mismatch: manifest={manifest.manifest_sha256!r}, "
            f"expected={expected_manifest_sha256!r}"
        )
    if signoff.instrument_manifest_sha256 != expected_manifest_sha256:
        raise SignoffError(
            f"Signoff manifest hash mismatch: signoff="
            f"{signoff.instrument_manifest_sha256!r}, "
            f"expected={expected_manifest_sha256!r}"
        )
    if manifest.manifest_sha256 != signoff.instrument_manifest_sha256:
        raise SignoffError(
            f"Manifest/signoff manifest hash mismatch: "
            f"manifest={manifest.manifest_sha256!r}, "
            f"signoff={signoff.instrument_manifest_sha256!r}"
        )

    # --- Equivalence margin — exact canonical string comparison ---
    if manifest.equivalence_margin != expected_equivalence_margin:
        raise SignoffError(
            f"Margin mismatch: manifest={manifest.equivalence_margin!r}, "
            f"expected={expected_equivalence_margin!r}"
        )
    if signoff.equivalence_margin != expected_equivalence_margin:
        raise SignoffError(
            f"Margin mismatch: signoff={signoff.equivalence_margin!r}, "
            f"expected={expected_equivalence_margin!r}"
        )
    if manifest.equivalence_margin != signoff.equivalence_margin:
        raise SignoffError(
            f"Manifest/signoff margin mismatch: "
            f"manifest={manifest.equivalence_margin!r}, "
            f"signoff={signoff.equivalence_margin!r}"
        )

    # --- Sample size N — exact integer comparison ---
    if manifest.sample_size_tasks_per_family != expected_sample_size_tasks_per_family:
        raise SignoffError(
            f"N mismatch: manifest={manifest.sample_size_tasks_per_family}, "
            f"expected={expected_sample_size_tasks_per_family}"
        )
    if signoff.sample_size_tasks_per_family != expected_sample_size_tasks_per_family:
        raise SignoffError(
            f"N mismatch: signoff={signoff.sample_size_tasks_per_family}, "
            f"expected={expected_sample_size_tasks_per_family}"
        )
    if manifest.sample_size_tasks_per_family != signoff.sample_size_tasks_per_family:
        raise SignoffError(
            f"Manifest/signoff N mismatch: "
            f"manifest={manifest.sample_size_tasks_per_family}, "
            f"signoff={signoff.sample_size_tasks_per_family}"
        )

    # --- Count map A=B=C=N ---
    _validate_count_map(
        manifest.meta_test_tasks_by_family, expected_sample_size_tasks_per_family
    )
    _validate_count_map(
        signoff.meta_test_tasks_by_family, expected_sample_size_tasks_per_family
    )
    if manifest.meta_test_tasks_by_family != signoff.meta_test_tasks_by_family:
        raise SignoffError(
            f"Manifest/signoff count map mismatch: "
            f"manifest={manifest.meta_test_tasks_by_family}, "
            f"signoff={signoff.meta_test_tasks_by_family}"
        )

    # --- Human signoff ID — exact string comparison ---
    if signoff.human_signoff_id != expected_human_signoff_id:
        raise SignoffError(
            f"Signoff ID mismatch: signoff={signoff.human_signoff_id!r}, "
            f"expected={expected_human_signoff_id!r}"
        )

    # --- Version consistency: signoff version must match manifest version ---
    if signoff.instrument_version != manifest.instrument_version:
        raise SignoffError(
            f"Version mismatch: manifest={manifest.instrument_version!r}, "
            f"signoff={signoff.instrument_version!r}"
        )
    # Signoff ID version must match manifest version.
    _validate_signoff_id_with_version(
        signoff.human_signoff_id, manifest.instrument_version
    )

    # --- Instrument ID consistency ---
    if signoff.instrument_id != manifest.instrument_id:
        raise SignoffError(
            f"Instrument ID mismatch: manifest={manifest.instrument_id!r}, "
            f"signoff={signoff.instrument_id!r}"
        )

    # All three-way checks passed.
    return RunAuthorization(
        instrument_manifest_sha256=manifest.manifest_sha256,
        equivalence_margin=manifest.equivalence_margin,
        sample_size_tasks_per_family=manifest.sample_size_tasks_per_family,
        human_signoff_id=signoff.human_signoff_id,
        signoff_sha256=signoff.signoff_sha256,
    )


def authorize_instrument(
    root: Path,
    authorization: RunAuthorization,
) -> AuthorizedInstrument:
    """Authorize an instrument after verifying all file hashes and bindings.

    This function verifies every public/calibration file hash listed in the
    manifest against actual bytes on disk.  It does **not** open sealed files
    — sealed verification happens later in the meta-test runner after
    authorization succeeds.

    The gate order is:
    1. Read and validate manifest (no sealed access).
    2. Read and validate signoff.
    3. Re-verify the authorization tuple against freshly read records.
    4. Verify all listed file hashes and sizes.
    5. Return :class:`AuthorizedInstrument`.

    Args:
        root: Root directory containing ``MANIFEST.json``, ``SIGNOFF.json``,
            and all instrument files.
        authorization: A :class:`RunAuthorization` obtained from
            :func:`verify_run_authorization`.

    Returns:
        An :class:`AuthorizedInstrument` if all checks pass.

    Raises:
        SignoffError: On any authorization mismatch.
        InstrumentIntegrityError: On file hash/size/path violation.
    """
    _check_no_env_bypass()

    root = Path(root)

    # Re-read and validate manifest and signoff from root.
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        manifest_path = root / "CANDIDATE_MANIFEST.json"
    if not manifest_path.is_file():
        raise InstrumentIntegrityError(
            f"No MANIFEST.json in {root}"
        )
    manifest = _read_and_parse_manifest(manifest_path)

    signoff_path = root / "SIGNOFF.json"
    signoff = _read_and_parse_signoff(signoff_path)

    # Re-verify the authorization tuple against the freshly read records.
    if manifest.manifest_sha256 != authorization.instrument_manifest_sha256:
        raise SignoffError(
            f"manifest hash does not match authorization: "
            f"manifest={manifest.manifest_sha256!r}, "
            f"authorization={authorization.instrument_manifest_sha256!r}"
        )
    if signoff.instrument_manifest_sha256 != authorization.instrument_manifest_sha256:
        raise SignoffError(
            f"signoff manifest hash does not match authorization: "
            f"signoff={signoff.instrument_manifest_sha256!r}, "
            f"authorization={authorization.instrument_manifest_sha256!r}"
        )
    if manifest.equivalence_margin != authorization.equivalence_margin:
        raise SignoffError(
            f"Manifest margin does not match authorization: "
            f"manifest={manifest.equivalence_margin!r}, "
            f"authorization={authorization.equivalence_margin!r}"
        )
    if signoff.equivalence_margin != authorization.equivalence_margin:
        raise SignoffError(
            f"Signoff margin does not match authorization: "
            f"signoff={signoff.equivalence_margin!r}, "
            f"authorization={authorization.equivalence_margin!r}"
        )
    if manifest.sample_size_tasks_per_family != authorization.sample_size_tasks_per_family:
        raise SignoffError(
            f"Manifest N does not match authorization: "
            f"manifest={manifest.sample_size_tasks_per_family}, "
            f"authorization={authorization.sample_size_tasks_per_family}"
        )
    if signoff.sample_size_tasks_per_family != authorization.sample_size_tasks_per_family:
        raise SignoffError(
            f"Signoff N does not match authorization: "
            f"signoff={signoff.sample_size_tasks_per_family}, "
            f"authorization={authorization.sample_size_tasks_per_family}"
        )
    if signoff.human_signoff_id != authorization.human_signoff_id:
        raise SignoffError(
            f"Signoff ID does not match authorization: "
            f"signoff={signoff.human_signoff_id!r}, "
            f"authorization={authorization.human_signoff_id!r}"
        )
    if signoff.signoff_sha256 != authorization.signoff_sha256:
        raise SignoffError(
            f"Signoff self-hash does not match authorization: "
            f"signoff={signoff.signoff_sha256!r}, "
            f"authorization={authorization.signoff_sha256!r}"
        )

    # Verify all listed file hashes and sizes.
    _verify_file_hashes(root, manifest.files)

    return AuthorizedInstrument(manifest, signoff, authorization)
