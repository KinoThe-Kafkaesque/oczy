"""Prepare a mixed Kaggle/Colab experiment campaign from a clean source commit.

Reads a campaign manifest (``oczy/remote-experiment-campaign/v1``), validates
schema/job uniqueness/phases/providers/claim classes and source provenance,
then for each job calls the existing provider-specific preparer:

* **Kaggle** → :func:`prepare_research_kernel.prepare_kernel` with
  ``module="infrastructure.kaggle.run_experiment_module"`` so the generated
  bootstrap invokes the shared runner, which in turn executes the campaign
  job's experiment ``module`` and writes ``execution_report.json``.
* **Colab** → :func:`prepare_colab_experiment.prepare_colab_experiment` which
  generates a self-contained ``colab_bootstrap.py`` that checks out the exact
  commit from the public repo and invokes the same runner.

The preparer emits a scheduler v2 batch manifest
(``oczy/remote-parallel-batch/v2``) consumable by :func:`load_batch` in
``parallel_scheduler.py``, plus a campaign manifest recording resolved paths.

Old v1 tooling (``prepare_research_kernel.py`` standalone CLI) remains
unchanged — this module composes it, it does not modify it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath as _PurePath
from typing import Any

# --- Co-located imports (same directory) -----------------------------------
# Ensure the script's own directory is importable when run as a script.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from parallel_scheduler import (  # noqa: E402  # type: ignore[import-not-found]
    _VALID_PROVIDERS,
    BATCH_SCHEMA_V2,
    PROVIDER_COLAB,
    PROVIDER_KAGGLE,
)
from prepare_research_kernel import (  # noqa: E402  # type: ignore[import-not-found]
    COMMIT_PATTERN,
    PHASES,
    PROFILES,
    SHA256_PATTERN,
    prepare_kernel,
)

try:
    from prepare_colab_experiment import (  # noqa: E402  # type: ignore[import-not-found]
        prepare_colab_experiment,
    )
    _COLAB_PREP_AVAILABLE = True
except ImportError:  # pragma: no cover — peer-built module
    prepare_colab_experiment = None  # type: ignore[assignment]
    _COLAB_PREP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMPAIGN_SCHEMA_VERSION = "oczy/remote-experiment-campaign/v1"
CAMPAIGN_MANIFEST_SCHEMA = "oczy/remote-experiment-campaign-manifest/v1"
DEFAULT_SOURCE_REPO = "https://github.com/KinoThe-Kafkaesque/oczy.git"
RUNNER_MODULE = "infrastructure.kaggle.run_experiment_module"
DEFAULT_REPORT_FILENAME = "execution_report.json"

CLAIM_SCIENTIFIC = "scientific"
CLAIM_INFRASTRUCTURE = "infrastructure"
_VALID_CLAIM_CLASSES = frozenset({CLAIM_SCIENTIFIC, CLAIM_INFRASTRUCTURE})
_VALID_MODEL_ARTIFACT_KINDS = frozenset({"gguf", "hf_snapshot"})
_MODEL_ARTIFACT_REQUIRED_FIELDS = ("kind", "repo_id", "revision", "filename", "sha256")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Campaign validation
# ---------------------------------------------------------------------------

class CampaignValidationError(ValueError):
    """Raised when a campaign manifest fails validation."""


def validate_campaign(campaign: dict[str, Any]) -> None:
    """Validate a campaign manifest dict in-place.

    Raises :class:`CampaignValidationError` on any failure: bad schema
    version, invalid/missing source commit, duplicate job names, invalid
    providers/phases/claim classes, or missing provider-specific required
    fields.
    """
    if not isinstance(campaign, dict):
        raise CampaignValidationError("campaign must be a JSON object")

    schema = campaign.get("schema_version")
    if schema != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignValidationError(
            f"unsupported campaign schema_version: {schema!r}, "
            f"expected {CAMPAIGN_SCHEMA_VERSION!r}"
        )

    source_commit = campaign.get("source_commit")
    if not isinstance(source_commit, str) or not COMMIT_PATTERN.fullmatch(source_commit):
        raise CampaignValidationError(
            "source_commit must be a 40-character lowercase hex Git SHA"
        )

    source_repo = campaign.get("source_repo", DEFAULT_SOURCE_REPO)
    if not isinstance(source_repo, str) or not source_repo.strip():
        raise CampaignValidationError("source_repo must be a non-empty string")
    if source_repo != DEFAULT_SOURCE_REPO:
        raise CampaignValidationError(
            f"unsupported source_repo: {source_repo!r}. "
            f"Only the public Oczy repository {DEFAULT_SOURCE_REPO!r} is accepted."
        )

    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise CampaignValidationError("campaign 'jobs' must be a non-empty list")

    seen_names: set[str] = set()
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise CampaignValidationError(f"job #{i} is not an object")

        name = job.get("name")
        if not name or not isinstance(name, str):
            raise CampaignValidationError(f"job #{i} missing string 'name'")
        if name in seen_names:
            raise CampaignValidationError(f"duplicate job name: {name!r}")
        seen_names.add(name)

        provider = job.get("provider")
        if provider not in _VALID_PROVIDERS:
            raise CampaignValidationError(
                f"job {name!r} has invalid or missing 'provider' "
                f"(must be one of {sorted(_VALID_PROVIDERS)!r})"
            )

        phase = job.get("phase")
        if phase not in PHASES:
            raise CampaignValidationError(
                f"job {name!r} has invalid phase {phase!r} "
                f"(must be one of {PHASES!r})"
            )

        module = job.get("module")
        if not module or not isinstance(module, str):
            raise CampaignValidationError(f"job {name!r} missing string 'module'")

        arguments = job.get("arguments", [])
        if arguments is None:
            arguments = []
            job["arguments"] = arguments
        if not isinstance(arguments, list) or not all(
            isinstance(a, str) for a in arguments
        ):
            raise CampaignValidationError(
                f"job {name!r}: arguments must be a list of strings"
            )

        # Generic required_arguments enforcement: each required token must
        # appear exactly in the job's arguments list (case-insensitive and
        # order-independent). Applies to all providers.
        required_arguments = job.get("required_arguments")
        if required_arguments is not None:
            if not isinstance(required_arguments, list) or not all(
                isinstance(a, str) for a in required_arguments
            ):
                raise CampaignValidationError(
                    f"job {name!r}: required_arguments must be a list of strings"
                )
            args_lower = [a.lower() for a in arguments]
            for req in required_arguments:
                req_lower = req.lower()
                if req_lower not in args_lower:
                    raise CampaignValidationError(
                        f"job {name!r}: required argument {req!r} not found "
                        f"in arguments {arguments!r}"
                    )

        output_path = job.get("output_path")
        if not output_path or not isinstance(output_path, str):
            raise CampaignValidationError(f"job {name!r} missing string 'output_path'")

        claim_class = job.get("claim_class")
        if claim_class not in _VALID_CLAIM_CLASSES:
            raise CampaignValidationError(
                f"job {name!r} has invalid claim_class {claim_class!r} "
                f"(must be one of {sorted(_VALID_CLAIM_CLASSES)!r})"
            )

        # Provider-specific required fields.
        if provider == PROVIDER_KAGGLE:
            _validate_kaggle_job_fields(name, job, phase)
        # Colab-specific validation is delegated to prepare_colab_experiment.

        # Colab-only optional model provisioning fields.
        # These are rejected on Kaggle jobs to enforce the CPU-only,
        # no-external-model contract for Kaggle kernels.
        model_artifact = job.get("model_artifact")
        install_llama_cpp = job.get("install_llama_cpp")
        if provider == PROVIDER_KAGGLE:
            if model_artifact is not None:
                raise CampaignValidationError(
                    f"kaggle job {name!r}: model_artifact is only supported "
                    f"on Colab jobs"
                )
            if install_llama_cpp is not None:
                raise CampaignValidationError(
                    f"kaggle job {name!r}: install_llama_cpp is only supported "
                    f"on Colab jobs"
                )
        elif provider == PROVIDER_COLAB:
            if model_artifact is not None:
                _validate_model_artifact(name, model_artifact)
            if install_llama_cpp is not None:
                if not isinstance(install_llama_cpp, bool):
                    raise CampaignValidationError(
                        f"colab job {name!r}: install_llama_cpp must be a boolean"
                    )
            else:
                job["install_llama_cpp"] = False


def _validate_kaggle_job_fields(name: str, job: dict[str, Any], phase: str) -> None:
    """Validate Kaggle-specific required fields on a campaign job."""
    for field in ("kernel_id", "title", "source_dataset", "source_archive_sha256"):
        val = job.get(field)
        if not val or not isinstance(val, str):
            raise CampaignValidationError(
                f"kaggle job {name!r} missing required string {field!r}"
            )

    if not SHA256_PATTERN.fullmatch(job["source_archive_sha256"]):
        raise CampaignValidationError(
            f"kaggle job {name!r}: source_archive_sha256 must be a "
            f"lowercase SHA-256"
        )

    profile = job.get("profile", "cpu")
    if profile not in PROFILES:
        raise CampaignValidationError(
            f"kaggle job {name!r}: unknown profile {profile!r}"
        )

    instrument_manifest_sha256 = job.get("instrument_manifest_sha256")
    if instrument_manifest_sha256 and not SHA256_PATTERN.fullmatch(
        instrument_manifest_sha256
    ):
        raise CampaignValidationError(
            f"kaggle job {name!r}: instrument_manifest_sha256 must be a "
            f"lowercase SHA-256"
        )

    if phase == "meta-test" and not (
        instrument_manifest_sha256 and job.get("human_signoff_id")
    ):
        raise CampaignValidationError(
            f"kaggle job {name!r}: meta-test phase requires "
            "instrument_manifest_sha256 and human_signoff_id"
        )

def _validate_model_artifact(name: str, model_artifact: Any) -> None:
    """Validate a Colab job's optional ``model_artifact`` dict.

    The artifact must specify a Hugging Face repo, an exact 40-hex revision,
    a filename, and a 64-hex SHA-256 of the downloaded file.  ``kind``
    determines whether a single GGUF file is fetched via direct streaming
    or a full snapshot via ``snapshot_download``.

    For ``hf_snapshot`` with a ``files`` manifest: validates non-empty list
    of entries with safe relative filenames, positive integer sizes, exact
    64-hex SHA-256 hashes, unique filenames, and requires the top-level
    ``filename``/``sha256`` to match one entry (the primary file).
    """
    if not isinstance(model_artifact, dict):
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact must be an object"
        )
    for field in _MODEL_ARTIFACT_REQUIRED_FIELDS:
        if field not in model_artifact:
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact missing required field "
                f"{field!r}"
            )
    kind = model_artifact["kind"]
    if kind not in _VALID_MODEL_ARTIFACT_KINDS:
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact kind {kind!r} "
            f"must be one of {sorted(_VALID_MODEL_ARTIFACT_KINDS)!r}"
        )
    repo_id = model_artifact["repo_id"]
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact repo_id must be a "
            f"non-empty string"
        )
    revision = model_artifact["revision"]
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact revision must be a "
            f"40-character lowercase hex SHA"
        )
    filename = model_artifact["filename"]
    if not isinstance(filename, str) or not filename.strip():
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact filename must be a "
            f"non-empty string"
        )
    sha256 = model_artifact["sha256"]
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact sha256 must be a "
            f"64-character lowercase hex SHA-256"
        )
    # Validate optional files manifest (hf_snapshot only).
    files = model_artifact.get("files")
    if files is not None:
        if kind != "hf_snapshot":
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact 'files' is only "
                f"supported for hf_snapshot, not kind {kind!r}"
            )
        _validate_model_artifact_files(name, filename, sha256, files)


def _validate_safe_relative_filename(name: str, filename: str) -> None:
    """Validate that *filename* is a safe relative path within a cache dir."""
    if not isinstance(filename, str) or not filename:
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact files entry filename "
            f"must be a non-empty string"
        )
    if filename != filename.strip():
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact files entry filename "
            f"must not have leading/trailing whitespace (got {filename!r})"
        )
    if "\x00" in filename or "\\" in filename:
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact files entry filename "
            f"must not contain null bytes or backslashes (got {filename!r})"
        )
    p = _PurePath(filename)
    if p.is_absolute():
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact files entry filename "
            f"must be relative, not absolute (got {filename!r})"
        )
    if any(part == ".." for part in p.parts):
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact files entry filename "
            f"must not contain '..' traversal (got {filename!r})"
        )


def _validate_model_artifact_files(
    name: str, primary_filename: str, primary_sha256: str, files: Any
) -> None:
    """Validate the ``files`` manifest of an hf_snapshot model_artifact."""
    if not isinstance(files, list) or not files:
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact 'files' must be a "
            f"non-empty list"
        )
    seen: set[str] = set()
    primary_matched = False
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact files[{i}] must be "
                f"an object"
            )
        for field in ("filename", "size_bytes", "sha256"):
            if field not in entry:
                raise CampaignValidationError(
                    f"colab job {name!r}: model_artifact files[{i}] "
                    f"missing required field {field!r}"
                )
        fname = entry["filename"]
        _validate_safe_relative_filename(name, fname)
        if fname in seen:
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact files[{i}] "
                f"duplicate filename {fname!r}"
            )
        seen.add(fname)
        size_bytes = entry["size_bytes"]
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact files[{i}] size_bytes "
                f"must be a positive integer (got {size_bytes!r})"
            )
        fsha = entry["sha256"]
        if not isinstance(fsha, str) or not SHA256_PATTERN.fullmatch(fsha):
            raise CampaignValidationError(
                f"colab job {name!r}: model_artifact files[{i}] sha256 "
                f"must be a 64-character lowercase hex digest (got {fsha!r})"
            )
        if fname == primary_filename:
            if fsha != primary_sha256:
                raise CampaignValidationError(
                    f"colab job {name!r}: model_artifact files[{i}] sha256 "
                    f"for primary file {fname!r} must match top-level sha256 "
                    f"(got {fsha!r}, expected {primary_sha256!r})"
                )
            primary_matched = True
    if not primary_matched:
        raise CampaignValidationError(
            f"colab job {name!r}: model_artifact 'files' must contain an "
            f"entry for the primary filename {primary_filename!r} with "
            f"matching sha256"
        )


# ---------------------------------------------------------------------------
# Runner argument construction
# ---------------------------------------------------------------------------

def _build_runner_arguments(
    job: dict[str, Any],
    source_commit: str,
    provider: str,
) -> list[str]:
    """Build argv for ``run_experiment_module`` from a campaign job.

    The runner CLI accepts ``--module``, repeated ``--arg=<value>``,
    ``--source-commit``, ``--provider``, ``--job-name``, and ``--report``.
    Each campaign job argument becomes a single ``--arg=<value>`` token so the
    runner forwards it verbatim to the experiment module — including values
    that begin with ``-`` (e.g. ``--seed``, negative numbers, empty strings).
    """
    runner_args: list[str] = [
        "--module", job["module"],
        "--source-commit", source_commit,
        "--provider", provider,
        "--job-name", job["name"],
        "--report", DEFAULT_REPORT_FILENAME,
    ]
    for arg in job.get("arguments", []):
        runner_args.append(f"--arg={arg}")
    return runner_args


# ---------------------------------------------------------------------------
# Batch manifest construction
# ---------------------------------------------------------------------------

def _build_batch_job_entry(
    job: dict[str, Any],
    kernel_dir_rel: str | None,
    script_rel: str | None,
    output_dir_rel: str,
) -> dict[str, Any]:
    """Build a v2 batch job entry from a campaign job + resolved relative paths."""
    entry: dict[str, Any] = {
        "name": job["name"],
        "provider": job["provider"],
        "output_dir": output_dir_rel,
    }
    if job["provider"] == PROVIDER_KAGGLE:
        entry["kernel_dir"] = kernel_dir_rel
    else:
        entry["script"] = script_rel
        entry["arguments"] = []  # bootstrap bakes them in
        timeout = job.get("timeout")
        if timeout is not None:
            entry["timeout"] = float(timeout)
    return entry


# ---------------------------------------------------------------------------
# Campaign preparation
# ---------------------------------------------------------------------------

def prepare_experiment_campaign(
    campaign_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare a mixed-provider experiment campaign.

    Reads the campaign manifest at *campaign_path*, validates it, calls the
    provider-specific preparer for each job, and writes:

    * ``batch.json`` — scheduler v2 batch manifest
    * ``campaign_manifest.json`` — campaign manifest with resolved paths

    Returns a dict with ``batch_path``, ``manifest_path``, and ``jobs``.
    """
    campaign_path = Path(campaign_path).resolve()
    if not campaign_path.is_file():
        raise FileNotFoundError(f"campaign manifest not found: {campaign_path}")

    try:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampaignValidationError(f"invalid campaign JSON: {exc}") from exc

    validate_campaign(campaign)

    source_commit = campaign["source_commit"]
    source_repo = campaign.get("source_repo", DEFAULT_SOURCE_REPO)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_jobs: list[dict[str, Any]] = []
    job_paths: dict[str, dict[str, str]] = []

    for job in campaign["jobs"]:
        name = job["name"]
        provider = job["provider"]
        output_dir_rel = job["output_path"]

        if provider == PROVIDER_KAGGLE:
            kernel_subdir = f"kernels/{name}"
            kernel_dir = output_dir / kernel_subdir
            runner_args = _build_runner_arguments(job, source_commit, PROVIDER_KAGGLE)

            prepare_kernel(
                output=kernel_dir,
                kernel_id=job["kernel_id"],
                title=job["title"],
                phase=job["phase"],
                profile=job.get("profile", "cpu"),
                source_dataset=job["source_dataset"],
                source_commit=source_commit,
                source_archive_sha256=job["source_archive_sha256"],
                module=RUNNER_MODULE,
                arguments=runner_args,
                model_source=job.get("model_source"),
                instrument_manifest_sha256=job.get("instrument_manifest_sha256"),
                human_signoff_id=job.get("human_signoff_id"),
                force=force,
            )

            batch_jobs.append(
                _build_batch_job_entry(
                    job, kernel_dir_rel=kernel_subdir,
                    script_rel=None, output_dir_rel=output_dir_rel,
                )
            )
            job_paths.append({
                "name": name,
                "provider": provider,
                "kernel_dir": kernel_subdir,
                "output_dir": output_dir_rel,
            })

        elif provider == PROVIDER_COLAB:
            if not _COLAB_PREP_AVAILABLE:
                raise RuntimeError(
                    "prepare_colab_experiment is not available — "
                    "infrastructure/kaggle/prepare_colab_experiment.py "
                    "must be installed to prepare Colab jobs"
                )

            colab_subdir = f"colab/{name}"
            colab_dir = output_dir / colab_subdir

            prepare_colab_experiment(
                output=colab_dir,
                job_name=name,
                repo_url=source_repo,
                source_commit=source_commit,
                module=job["module"],
                arguments=job.get("arguments", []),
                phase=job["phase"],
                claim_class=job["claim_class"],
                output_path=output_dir_rel,
                timeout=job.get("timeout"),
                model_artifact=job.get("model_artifact"),
                install_llama_cpp=job.get("install_llama_cpp", False),
                force=force,
            )

            script_rel = f"{colab_subdir}/colab_bootstrap.py"
            batch_jobs.append(
                _build_batch_job_entry(
                    job, kernel_dir_rel=None,
                    script_rel=script_rel, output_dir_rel=output_dir_rel,
                )
            )
            job_paths.append({
                "name": name,
                "provider": provider,
                "script": script_rel,
                "output_dir": output_dir_rel,
            })
        else:
            # Unreachable — validate_campaign already checked.
            raise CampaignValidationError(
                f"job {name!r} has unknown provider {provider!r}"
            )

    # Write v2 batch manifest.
    batch_path = output_dir / "batch.json"
    batch_manifest: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_V2,
        "campaign_source_commit": source_commit,
        "jobs": batch_jobs,
    }
    _write_json(batch_path, batch_manifest)

    # Write campaign manifest.
    manifest_path = output_dir / "campaign_manifest.json"
    campaign_manifest: dict[str, Any] = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign": campaign,
        "batch_path": "batch.json",
        "prepared_at": _utc_now(),
        "job_paths": job_paths,
    }
    _write_json(manifest_path, campaign_manifest)

    return {
        "batch_path": str(batch_path),
        "manifest_path": str(manifest_path),
        "jobs": job_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prepare_experiment_campaign",
        description=__doc__,
    )
    parser.add_argument("campaign", type=Path, help="Path to campaign manifest JSON.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for generated batch + per-job artifacts.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing generated files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_experiment_campaign(args.campaign, args.output, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
