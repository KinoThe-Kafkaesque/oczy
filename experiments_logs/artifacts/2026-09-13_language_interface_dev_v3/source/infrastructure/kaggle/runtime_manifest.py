"""Provider-neutral runtime manifest: identity, observation, validation.

Schema ``oczy/runtime-manifest/v2``.  This is the single stdlib-only contract
owner.  Everything that ships a manifest or checks one MUST go through this
module — no duplicated loaders, no private validators, no optional subset
matching.

Exports
-------
- ``RUNTIME_MANIFEST_SCHEMA_VERSION``
- ``RuntimeManifestError``
- ``strict_json_loads``
- ``strict_canonical_json``
- ``compute_manifest_sha256``
- ``compute_component_sha256``
- ``validate_runtime_manifest``
- ``compare_runtime_manifests``
- ``observe_runtime_manifest``
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RUNTIME_MANIFEST_SCHEMA_VERSION: str = "oczy/runtime-manifest/v2"

# Recognised model load conventions (must match exactly).
_VALID_CONVENTIONS: frozenset[str] = frozenset(
    {"transformers-pretrained-directory", "llama-cpp-gguf-file", "none"}
)

# Recognised artifact roles.
_VALID_ROLES: frozenset[str] = frozenset(
    {"weights", "config", "tokenizer", "chat_template", "generation_config", "other"}
)

# Model-bearing manifests MUST cover at least these four roles.
_REQUIRED_MODEL_ROLES: frozenset[str] = frozenset(
    {"weights", "config", "tokenizer", "chat_template"}
)

# Packages observed with importlib (stdlib-only fallback tries distribution name).
_OBSERVED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchao",
    "transformers",
    "tokenizers",
    "safetensors",
)

# File-hash chunk size (8 MiB, matches existing Qwen probe pattern).
_CHUNK_SIZE: int = 8 * 1024 * 1024

# Maximum mismatch message depth (bounded, never dumps full artifacts).
_MAX_MISMATCH_DEPTH: int = 20

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RuntimeManifestError(ValueError):
    """Raised when a runtime manifest is missing, malformed, or mismatched."""


# ---------------------------------------------------------------------------
# Strict JSON helpers (stdlib-only; same contract as instrument_contracts.py)
# ---------------------------------------------------------------------------


def _strict_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys in JSON objects."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise RuntimeManifestError(f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _strict_parse_constant(value: str) -> None:
    """Reject NaN, Infinity, -Infinity in JSON."""
    raise RuntimeManifestError(
        f"JSON constant {value!r} is not allowed (NaN/Infinity rejected)"
    )


def strict_json_loads(data: bytes | str) -> dict[str, Any]:
    """Parse JSON with strict discipline.

    Returns a plain ``dict``.  Rejects:

    * Duplicate keys
    * NaN / Infinity
    * Non-object top-level
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    result = json.loads(
        data,
        object_pairs_hook=_strict_object_pairs_hook,
        parse_constant=_strict_parse_constant,
    )
    if not isinstance(result, dict):
        raise RuntimeManifestError(
            f"Expected JSON object, got {type(result).__name__}"
        )
    return result


def strict_canonical_json(obj: Any) -> bytes:
    """Serialize *obj* to canonical UTF-8 bytes.

    Uses ``sort_keys=True``, compact separators, ``ensure_ascii=False``,
    ``allow_nan=False``.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Stream SHA-256 of a file in *CHUNK_SIZE* blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_canonical_sha256(obj: Any) -> str:
    """SHA-256 of canonical JSON for *obj*."""
    return hashlib.sha256(strict_canonical_json(obj)).hexdigest()


def compute_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Compute the self-hash of *manifest*.

    Removes ``manifest_sha256`` (if present), serialises the remainder as
    canonical JSON, and returns the hex SHA-256 digest.
    """
    stripped = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return _compute_canonical_sha256(stripped)


def compute_component_sha256(files: list[dict[str, Any]]) -> str:
    """Compute the component hash over a list of artifact-file objects.

    Each file object is projected to ``{path, size_bytes, sha256}`` (roles
    are excluded).  The list MUST already be sorted by *path*.
    """
    payload = [
        {"path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]}
        for f in files
    ]
    return _compute_canonical_sha256(payload)


# ---------------------------------------------------------------------------
# Artifact path validation
# ---------------------------------------------------------------------------

_SAFE_PATH_FORBIDDEN: frozenset[str] = frozenset({".", "..", ""})


def _validate_artifact_path(raw: Any) -> None:
    """Validate a single artifact ``path`` field.

    Rejects: non-string, empty, leading/trailing whitespace, backslash, NUL,
    absolute, ``.`` / ``..`` components, non-POSIX separator.
    """
    if not isinstance(raw, str):
        raise RuntimeManifestError(
            f"artifact `path` must be a string, got {type(raw).__name__}"
        )
    if raw != raw.strip():
        raise RuntimeManifestError(f"artifact `path` has leading/trailing whitespace: {raw!r}")
    if raw == "":
        raise RuntimeManifestError("artifact `path` must not be empty")
    if "\\" in raw:
        raise RuntimeManifestError(f"artifact `path` contains backslash: {raw!r}")
    if "\x00" in raw:
        raise RuntimeManifestError(f"artifact `path` contains NUL byte: {raw!r}")
    if raw.startswith("/"):
        raise RuntimeManifestError(f"artifact `path` is absolute: {raw!r}")
    parts = raw.split("/")
    if any(p in _SAFE_PATH_FORBIDDEN for p in parts):
        raise RuntimeManifestError(f"artifact `path` has reserved component: {raw!r}")
    if any("/" not in raw and "\\" in raw for _ in []):  # redundant guard
        pass


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _validate_int_field(value: Any, name: str, *, positive: bool = True) -> None:
    """Reject non-int or bool values for an integer field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeManifestError(
            f"`{name}` must be an int, got {type(value).__name__}"
        )
    if positive and value <= 0:
        raise RuntimeManifestError(f"`{name}` must be positive, got {value}")


def _validate_greedy_block(generation: dict[str, Any] | None) -> None:
    """Validate the ``greedy_generation`` block.

    Rejects when constraints are not fail-closed greedy (do_sample=false,
    num_beams=1, temperature=0).
    """
    if generation is None:
        return
    if not isinstance(generation, dict):
        raise RuntimeManifestError("`greedy_generation` must be an object or null")

    expected = [
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "num_beams",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "length_penalty",
        "no_repeat_ngram_size",
        "use_cache",
        "eos_token_ids",
        "pad_token_id",
        "stop_strings",
    ]
    actual_keys = set(generation.keys())
    if actual_keys != set(expected):
        extra = actual_keys - set(expected)
        missing = set(expected) - actual_keys
        msg_parts: list[str] = []
        if extra:
            msg_parts.append(f"unexpected key(s): {sorted(extra)}")
        if missing:
            msg_parts.append(f"missing key(s): {sorted(missing)}")
        raise RuntimeManifestError("`greedy_generation` key set mismatch: " + "; ".join(msg_parts))

    if generation.get("do_sample") is not False:
        raise RuntimeManifestError("`greedy_generation.do_sample` must be false")
    if generation.get("num_beams") != 1:
        raise RuntimeManifestError("`greedy_generation.num_beams` must be 1")
    if generation.get("temperature") != 0.0 and generation.get("temperature") != 0:
        raise RuntimeManifestError("`greedy_generation.temperature` must be 0")

    _validate_int_field(generation.get("max_new_tokens"), "max_new_tokens", positive=True)

    eos_ids = generation.get("eos_token_ids")
    if not isinstance(eos_ids, list) or any(not isinstance(v, int) for v in eos_ids):
        raise RuntimeManifestError("`eos_token_ids` must be a list of integers")
    if len(eos_ids) != len(set(eos_ids)):
        raise RuntimeManifestError("`eos_token_ids` must be unique")

    stop_strings = generation.get("stop_strings")
    if not isinstance(stop_strings, list) or any(not isinstance(s, str) for s in stop_strings):
        raise RuntimeManifestError("`stop_strings` must be a list of strings")

    # All numeric fields must be finite
    for key in ("max_new_tokens", "min_new_tokens", "top_p", "top_k",
                "repetition_penalty", "length_penalty", "no_repeat_ngram_size"):
        val = generation.get(key)
        if isinstance(val, float):
            import math
            if math.isinf(val) or math.isnan(val):
                raise RuntimeManifestError(f"`greedy_generation.{key}` must be finite")
        elif isinstance(val, bool):
            raise RuntimeManifestError(f"`greedy_generation.{key}` must not be a boolean")


def _validate_artifact_files(files: Any) -> None:
    """Validate the ``artifact_files`` list."""
    if not isinstance(files, list):
        raise RuntimeManifestError("`artifact_files` must be a list")

    paths: list[str] = []
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            raise RuntimeManifestError(
                f"`artifact_files[{i}]` must be an object"
            )
        # Required keys
        for key in ("path", "size_bytes", "sha256", "roles"):
            if key not in f:
                raise RuntimeManifestError(
                    f"`artifact_files[{i}]` missing required key: {key}"
                )
        # No extra keys
        allowed = {"path", "size_bytes", "sha256", "roles"}
        extra = set(f.keys()) - allowed
        if extra:
            raise RuntimeManifestError(
                f"`artifact_files[{i}]` has unexpected key(s): {sorted(extra)}"
            )

        _validate_artifact_path(f["path"])
        paths.append(f["path"])

        _validate_int_field(f["size_bytes"], f"artifact_files[{i}].size_bytes", positive=True)

        sha = f["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            raise RuntimeManifestError(
                f"`artifact_files[{i}].sha256` must be 64 lowercase hex"
            )

        roles = f["roles"]
        if not isinstance(roles, list) or not roles:
            raise RuntimeManifestError(
                f"`artifact_files[{i}].roles` must be a nonempty list"
            )
        seen_roles: set[str] = set()
        for r in roles:
            if not isinstance(r, str):
                raise RuntimeManifestError(
                    f"`artifact_files[{i}].roles` contains non-string: {r!r}"
                )
            if r not in _VALID_ROLES:
                raise RuntimeManifestError(
                    f"`artifact_files[{i}].roles` invalid: {r!r}"
                )
            if r in seen_roles:
                raise RuntimeManifestError(
                    f"`artifact_files[{i}].roles` duplicate: {r!r}"
                )
            seen_roles.add(r)

    # Paths must be unique and sorted.
    if len(paths) != len(set(paths)):
        raise RuntimeManifestError("`artifact_files` paths must be unique")
    if paths != sorted(paths):
        raise RuntimeManifestError("`artifact_files` must be sorted by `path`")


def _validate_quantization_block(quantization: Any) -> None:
    """Validate the frozen TorchAO W8A32 recipe or explicit FP32/null."""
    if quantization is None:
        return
    if not isinstance(quantization, dict):
        raise RuntimeManifestError("`model.quantization` must be an object or null")
    expected = {
        "backend": "torchao",
        "config_version": 2,
        "scheme": "int8-weight-only",
        "weight_dtype": "int8",
        "activation_dtype": "float32",
        "granularity": "per-row",
        "set_inductor_config": False,
    }
    actual_keys = set(quantization)
    if actual_keys != set(expected):
        extra = actual_keys - set(expected)
        missing = set(expected) - actual_keys
        parts: list[str] = []
        if extra:
            parts.append(f"unexpected key(s): {sorted(extra)}")
        if missing:
            parts.append(f"missing key(s): {sorted(missing)}")
        raise RuntimeManifestError(
            "`model.quantization` key set mismatch: " + "; ".join(parts)
        )
    for key, value in expected.items():
        if type(quantization[key]) is not type(value) or quantization[key] != value:
            raise RuntimeManifestError(
                f"`model.quantization.{key}` must be {value!r}, "
                f"got {quantization[key]!r}"
            )


def _validate_model_block(model: dict[str, Any]) -> None:
    """Validate the ``model`` block."""
    required = [
        "logical_model_id",
        "resolved_model_convention",
        "quantization",
        "artifact_files",
        "model_weights_sha256",
        "model_config_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
    ]
    actual = set(model.keys())
    if actual != set(required):
        extra = actual - set(required)
        missing = set(required) - actual
        msg_parts: list[str] = []
        if extra:
            msg_parts.append(f"unexpected key(s): {sorted(extra)}")
        if missing:
            msg_parts.append(f"missing key(s): {sorted(missing)}")
        raise RuntimeManifestError("`model` key set mismatch: " + "; ".join(msg_parts))

    convention = model["resolved_model_convention"]
    if not isinstance(convention, str):
        raise RuntimeManifestError("`resolved_model_convention` must be a string")
    if convention not in _VALID_CONVENTIONS:
        raise RuntimeManifestError(
            f"`resolved_model_convention` unknown: {convention!r}; "
            f"expected one of {sorted(_VALID_CONVENTIONS)}"
        )

    logical_id = model["logical_model_id"]
    if logical_id is not None and not isinstance(logical_id, str):
        raise RuntimeManifestError("`logical_model_id` must be a string or null")


    _validate_quantization_block(model["quantization"])
    _validate_artifact_files(model["artifact_files"])

    artifacts: list[dict[str, Any]] = model["artifact_files"]
    is_no_model = convention == "none"

    if is_no_model:
        if logical_id is not None:
            raise RuntimeManifestError(
                "no-model branch: `logical_model_id` must be null"
            )
        if artifacts:
            raise RuntimeManifestError("no-model branch: `artifact_files` must be empty")
        if model["quantization"] is not None:
            raise RuntimeManifestError(
                "no-model branch: `quantization` must be null"
            )
        for role_hash_key in ("model_weights_sha256", "model_config_sha256",
                              "tokenizer_sha256", "chat_template_sha256"):
            if model[role_hash_key] is not None:
                raise RuntimeManifestError(
                    f"no-model branch: `{role_hash_key}` must be null"
                )
    else:
        if logical_id is None or not isinstance(logical_id, str):
            raise RuntimeManifestError(
                "model-bearing branch: `logical_model_id` must be a string"
            )
        if not artifacts:
            raise RuntimeManifestError(
                "model-bearing branch: `artifact_files` must not be empty"
            )
        # Every required role must be covered by at least one file.
        covered_roles: set[str] = set()
        for f in artifacts:
            covered_roles.update(f["roles"])
        missing_roles = _REQUIRED_MODEL_ROLES - covered_roles
        if missing_roles:
            raise RuntimeManifestError(
                f"model-bearing manifest missing required roles: {sorted(missing_roles)}"
            )

        # Validate component hashes.
        _ROLE_HASH_KEYS = {
            "weights": "model_weights_sha256",
            "config": "model_config_sha256",
            "tokenizer": "tokenizer_sha256",
            "chat_template": "chat_template_sha256",
        }
        for role, key in _ROLE_HASH_KEYS.items():
            expected = model[key]
            if expected is not None:
                role_files = [f for f in artifacts if role in f["roles"]]
                if not role_files:
                    raise RuntimeManifestError(
                        f"`{key}` is set but no files carry role `{role}`"
                    )
                computed = compute_component_sha256(role_files)
                if computed != expected:
                    raise RuntimeManifestError(
                        f"`{key}` mismatch: computed {computed}, "
                        f"authored {expected}"
                    )


def _validate_packages(packages: Any) -> None:
    """Validate the ``packages`` block."""
    required = list(_OBSERVED_PACKAGES)
    if not isinstance(packages, dict):
        raise RuntimeManifestError("`packages` must be an object")
    actual = set(packages.keys())
    if actual != set(required):
        extra = actual - set(required)
        missing = set(required) - actual
        msg_parts: list[str] = []
        if extra:
            msg_parts.append(f"unexpected key(s): {sorted(extra)}")
        if missing:
            msg_parts.append(f"missing key(s): {sorted(missing)}")
        raise RuntimeManifestError("`packages` key set mismatch: " + "; ".join(msg_parts))
    for name in required:
        ver = packages[name]
        if not isinstance(ver, str):
            raise RuntimeManifestError(
                f"`packages.{name}` must be a string, got {type(ver).__name__}"
            )


def validate_runtime_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fully validate *manifest* against ``oczy/runtime-manifest/v2``.

    Checks:
    * Exact key set (no extras, no missing)
    * Type correctness for every field
    * Cross-field constraints (model/no-model branches)
    * Artifact path safety
    * Role coverage and component hashes
    * Greedy generation constraints (fail-closed)
    * Self-hash integrity

    Returns *manifest* unchanged on success (for chaining).
    Raises :class:`RuntimeManifestError` on any failure.
    """
    if not isinstance(manifest, dict):
        raise RuntimeManifestError("manifest must be a JSON object")

    # Exact top-level key set.
    top_keys = {
        "schema_version",
        "python_version",
        "packages",
        "model",
        "greedy_generation",
        "manifest_sha256",
    }
    actual = set(manifest.keys())
    if actual != top_keys:
        extra = actual - top_keys
        missing = top_keys - actual
        msg_parts: list[str] = []
        if extra:
            msg_parts.append(f"unexpected key(s): {sorted(extra)}")
        if missing:
            msg_parts.append(f"missing key(s): {sorted(missing)}")
        raise RuntimeManifestError("manifest key set mismatch: " + "; ".join(msg_parts))

    # schema_version
    sv = manifest["schema_version"]
    if sv != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise RuntimeManifestError(
            f"`schema_version` must be {RUNTIME_MANIFEST_SCHEMA_VERSION!r}, got {sv!r}"
        )

    # python_version
    pv = manifest["python_version"]
    if not isinstance(pv, str) or pv == "":
        raise RuntimeManifestError("`python_version` must be a nonempty string")

    # packages
    _validate_packages(manifest["packages"])

    # model
    model = manifest["model"]
    if not isinstance(model, dict):
        raise RuntimeManifestError("`model` must be an object")
    _validate_model_block(model)

    # greedy_generation
    gg = manifest["greedy_generation"]
    if gg is not None and not isinstance(gg, dict):
        raise RuntimeManifestError("`greedy_generation` must be an object or null")
    is_no_model = model.get("resolved_model_convention") == "none"
    if is_no_model and gg is not None:
        raise RuntimeManifestError("no-model branch: `greedy_generation` must be null")
    if not is_no_model and gg is None:
        raise RuntimeManifestError(
            "model-bearing branch: `greedy_generation` must not be null"
        )
    _validate_greedy_block(gg)

    # manifest_sha256 must be a string.
    mh = manifest["manifest_sha256"]
    if not isinstance(mh, str) or len(mh) != 64 or not all(c in "0123456789abcdef" for c in mh):
        raise RuntimeManifestError("`manifest_sha256` must be 64 lowercase hex")

    # Self-hash integrity.
    computed = compute_manifest_sha256(manifest)
    if computed != mh:
        raise RuntimeManifestError(
            f"`manifest_sha256` mismatch: computed {computed}, authored {mh}"
        )

    return manifest


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _diff_dicts(
    expected: Any,
    observed: Any,
    path: str = "$",
    depth: int = 0,
) -> list[str]:
    """Recursively diff two canonical manifest dicts, bounded to *_MAX_MISMATCH_DEPTH*."""
    mismatches: list[str] = []
    if depth > _MAX_MISMATCH_DEPTH:
        mismatches.append(f"{path}: <max depth exceeded>")
        return mismatches
    if type(expected) is not type(observed):
        mismatches.append(
            f"{path}: type {type(expected).__name__} != {type(observed).__name__}"
        )
        return mismatches
    if isinstance(expected, dict):
        exp_keys = set(expected.keys())
        obs_keys = set(observed.keys())
        for k in sorted(exp_keys - obs_keys):
            mismatches.append(f"{path}.{k}: missing in observed")
        for k in sorted(obs_keys - exp_keys):
            mismatches.append(f"{path}.{k}: unexpected in observed")
        for k in sorted(exp_keys & obs_keys):
            mismatches.extend(
                _diff_dicts(expected[k], observed[k], f"{path}.{k}", depth + 1)
            )
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            mismatches.append(
                f"{path}: list length {len(expected)} != {len(observed)}"
            )
            return mismatches
        for i, (e, o) in enumerate(zip(expected, observed)):
            mismatches.extend(
                _diff_dicts(e, o, f"{path}[{i}]", depth + 1)
            )
    elif expected != observed:
        mismatches.append(f"{path}: {expected!r} != {observed!r}")
    return mismatches


def compare_runtime_manifests(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    """Validate and compare two runtime manifests.

    Each manifest is independently validated, then their canonical bytes are
    compared.  Returns a (possibly empty) list of human-readable mismatch
    paths.  An empty list means exact equality.
    """
    validate_runtime_manifest(expected)
    validate_runtime_manifest(observed)
    mismatches = _diff_dicts(expected, observed)
    return mismatches


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def _try_package_version(name: str) -> str | None:
    """Return the installed version of *name*, or None if not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        # Python < 3.8 fallback (importlib_metadata)
        try:
            from importlib_metadata import PackageNotFoundError, version  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _observe_packages() -> dict[str, str]:
    """Observe current Python and package versions.

    Returns ``{name: version_string}`` for each package in *_OBSERVED_PACKAGES*.
    Raises :class:`RuntimeManifestError` if any required package is not found.
    """
    packages: dict[str, str] = {}
    for name in _OBSERVED_PACKAGES:
        ver = _try_package_version(name)
        if ver is None:
            raise RuntimeManifestError(
                f"Required package not found: {name}. "
                f"Install it or use the no-model branch."
            )
        # Normalise: strip leading/trailing whitespace, reject empty.
        ver = ver.strip()
        if not ver:
            raise RuntimeManifestError(f"Package {name!r} has empty version string")
        packages[name] = ver
    return packages


def _observe_directory(
    root: Path,
    convention: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Observe model artifact files under *root*.

    Returns ``(artifact_files, component_hashes)``.

    *root* must exist and be a directory (for ``transformers-pretrained-directory``)
    or a regular file (for ``llama-cpp-gguf-file``).

    Every regular file inside the root is recorded; symlinks, directories,
    and special files are rejected.  Extra/unexpected files must still be
    recorded (``other`` role) — the caller or campaign preparation decides
    whether the inventory matches the expected manifest.
    """
    if convention not in ("transformers-pretrained-directory", "llama-cpp-gguf-file"):
        raise RuntimeManifestError(f"unsupported observation convention: {convention!r}")

    if not root.exists():
        raise RuntimeManifestError(f"model root does not exist: {root}")

    if convention == "transformers-pretrained-directory":
        if not root.is_dir():
            raise RuntimeManifestError(
                f"expected directory for {convention!r}, got: {root}"
            )
        paths = sorted(root.rglob("*"))
        # Reject symlinks
        for p in paths:
            if p.is_symlink():
                raise RuntimeManifestError(
                    f"symlinks not allowed in model directory: {p}"
                )
        files = [p for p in paths if p.is_file()]
    else:
        # GGUF: single file.
        if not root.is_file():
            raise RuntimeManifestError(
                f"expected regular file for {convention!r}, got: {root}"
            )
        if root.is_symlink():
            raise RuntimeManifestError(f"symlinks not allowed: {root}")
        files = [root]

    if not files:
        raise RuntimeManifestError(f"no regular files found under model root: {root}")

    artifact_files: list[dict[str, Any]] = []
    for f in sorted(files, key=lambda p: str(p.relative_to(root) if root.is_dir() else p.name)):
        if f.is_dir():
            continue
        if f.is_symlink():
            raise RuntimeManifestError(f"symlinks not allowed: {f}")

        rel = str(f.relative_to(root)) if root.is_dir() else f.name
        size = f.stat().st_size
        sha = _sha256_file(f)
        # Best-effort role assignment from filename (preparer should override).
        roles = _infer_roles_from_filename(rel, convention)
        artifact_files.append({
            "path": rel,
            "size_bytes": size,
            "sha256": sha,
            "roles": roles,
        })

    # Sort by path.
    artifact_files.sort(key=lambda f: f["path"])

    # Ingest chat_template embedded in tokenizer_config.json (strict / fail-closed).
    # Qwen and other models often embed the chat template directly in
    # tokenizer_config.json rather than shipping a standalone chat_template.jinja.
    if convention == "transformers-pretrained-directory":
        for f in artifact_files:
            if f["path"] == "tokenizer_config.json" and "tokenizer" in f["roles"]:
                tcfg_path = root / "tokenizer_config.json"
                try:
                    tcfg = strict_json_loads(tcfg_path.read_bytes())
                except (ValueError, OSError) as exc:
                    raise RuntimeManifestError(
                        f"failed to parse tokenizer_config.json: {exc}"
                    ) from exc
                template = tcfg.get("chat_template")
                if isinstance(template, str) and template:
                    if "chat_template" not in f["roles"]:
                        f["roles"] = sorted(set(f["roles"] + ["chat_template"]))
                break
    # Compute component hashes.
    component_hashes: dict[str, str | None] = {
        "model_weights_sha256": None,
        "model_config_sha256": None,
        "tokenizer_sha256": None,
        "chat_template_sha256": None,
    }
    _ROLE_HASH_KEYS = {
        "weights": "model_weights_sha256",
        "config": "model_config_sha256",
        "tokenizer": "tokenizer_sha256",
        "chat_template": "chat_template_sha256",
    }
    for role, key in _ROLE_HASH_KEYS.items():
        role_files = [f for f in artifact_files if role in f["roles"]]
        if role_files:
            component_hashes[key] = compute_component_sha256(role_files)

    return artifact_files, component_hashes


def _infer_roles_from_filename(filename: str, convention: str) -> list[str]:
    """Best-effort role inference from filename.

    Pre-sorted alphabetically.  The caller (campaign preparation) must
    override this with the reviewed roles from the expected manifest.
    """
    roles: list[str] = []
    base = filename.rsplit("/", 1)[-1]

    # Tokenizer files.
    if base in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        roles.append("tokenizer")
    # Config.
    if base == "config.json":
        roles.append("config")
    # Generation config.
    if base == "generation_config.json":
        roles.append("generation_config")
    # Chat template.
    if base.endswith(".jinja") and "chat" in base.lower():
        roles.append("chat_template")
    if base == "chat_template.jinja":
        if "chat_template" not in roles:
            roles.append("chat_template")
    # Weights.
    if base.endswith(".safetensors") or base.endswith(".bin") or base.endswith(".gguf"):
        roles.append("weights")
    # GGUF: single file carries all roles.
    if convention == "llama-cpp-gguf-file":
        roles = sorted(set(roles + ["weights", "config", "tokenizer", "chat_template"]))

    if not roles:
        roles.append("other")

    return sorted(set(roles))


def observe_runtime_manifest(
    *,
    model_root: Path | None = None,
    logical_model_id: str | None = None,
    resolved_model_convention: str = "none",
    generation_config: dict[str, Any] | None = None,
    quantization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Observe and build the local runtime manifest.

    Parameters
    ----------
    model_root:
        Path to the model directory (HF) or file (GGUF).  ``None`` for
        model-free jobs.
    logical_model_id:
        Reviewed logical model identifier (e.g. ``"LiquidAI/LFM2.5-1.2B-Instruct"``).
        ``None`` for model-free.
    resolved_model_convention:
        One of ``"transformers-pretrained-directory"``, ``"llama-cpp-gguf-file"``,
        or ``"none"``.
    generation_config:
        The reviewed generation contract, or ``None`` for model-free.
    quantization:
        Frozen model-loading quantization recipe, or ``None`` for FP32,
        GGUF-managed quantization, and model-free jobs.

    Returns
    -------
    dict
        The full observed manifest dict (with ``manifest_sha256`` computed).
    """
    if resolved_model_convention not in _VALID_CONVENTIONS:
        raise RuntimeManifestError(
            f"unknown convention: {resolved_model_convention!r}"
        )

    is_no_model = resolved_model_convention == "none"

    if is_no_model:
        if model_root is not None:
            raise RuntimeManifestError("no-model branch: `model_root` must be None")
        if logical_model_id is not None:
            raise RuntimeManifestError("no-model branch: `logical_model_id` must be None")
        if generation_config is not None:
            raise RuntimeManifestError(
                "no-model branch: `generation_config` must be None"
            )
        if quantization is not None:
            raise RuntimeManifestError(
                "no-model branch: `quantization` must be None"
            )
    else:
        if model_root is None:
            raise RuntimeManifestError("model-bearing: `model_root` is required")
        if logical_model_id is None:
            raise RuntimeManifestError("model-bearing: `logical_model_id` is required")
        if generation_config is None:
            raise RuntimeManifestError("model-bearing: `generation_config` is required")

    packages = _observe_packages()
    python_version = platform.python_version().strip()

    if is_no_model:
        artifact_files: list[dict[str, Any]] = []
        component_hashes: dict[str, str] = {}
    else:
        assert model_root is not None
        model_root = Path(model_root)
        artifact_files, component_hashes = _observe_directory(model_root, resolved_model_convention)

    model_block: dict[str, Any] = {
        "logical_model_id": logical_model_id,
        "resolved_model_convention": resolved_model_convention,
        "quantization": quantization,
        "artifact_files": artifact_files,
        "model_weights_sha256": component_hashes.get("model_weights_sha256"),
        "model_config_sha256": component_hashes.get("model_config_sha256"),
        "tokenizer_sha256": component_hashes.get("tokenizer_sha256"),
        "chat_template_sha256": component_hashes.get("chat_template_sha256"),
    }

    manifest: dict[str, Any] = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "python_version": python_version,
        "packages": packages,
        "model": model_block,
        "greedy_generation": generation_config,
    }

    # Compute self-hash and attach.
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)

    # Validate the built manifest.
    validate_runtime_manifest(manifest)

    return manifest
