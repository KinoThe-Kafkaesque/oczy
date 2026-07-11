"""Generate a self-contained Colab bootstrap for an Oczy remote experiment.

The bootstrap clones the public Oczy repository at an exact 40-character
commit, verifies HEAD, prepends repo and workspace-package source directories
to ``sys.path``, sets a strict CPU-only environment, then invokes
``infrastructure.kaggle.run_experiment_module`` with an explicit subprocess
argv.  All structured information is left in stdout/stderr for provider
collection, plus a structured ``execution_report.json`` written by the runner.

This generator writes two artifacts into the output directory:

* ``colab_bootstrap.py`` — self-contained Colab script.
* ``job_spec.json`` — human-reviewable job specification.

It does **not** embed credentials, does **not** invoke a shell, and does
**not** modify frozen research/eval instruments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema and constants
# ---------------------------------------------------------------------------

JOB_SPEC_SCHEMA_VERSION = "oczy/colab-experiment-job/v1"

#: The single public repository URL permitted for Colab jobs.
PUBLIC_REPO_URL = "https://github.com/KinoThe-Kafkaesque/oczy.git"

#: 40-character lowercase hex Git SHA.
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

#: 64-character lowercase hex SHA-256 digest.
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Valid model artifact kinds.
_VALID_MODEL_ARTIFACT_KINDS = frozenset({"gguf", "hf_snapshot"})

#: Pinned llama-cpp-python version for CPU wheel installation.
_LLAMA_CPP_VERSION = "0.3.31"

#: CPU-only wheel index for llama-cpp-python (abetlen).
_LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

#: Valid claim classes.
_VALID_CLAIM_CLASSES = frozenset({"scientific", "infrastructure"})

#: Accelerator-related argument substrings rejected anywhere in arguments.
#: The CPU-only contract applies to the target experiment, not just the CLI.
_ACCELERATOR_PATTERNS: tuple[str, ...] = (
    "--gpu",
    "--tpu",
    "--cuda",
    "--accelerator",
    "--device",
    "cuda:",
    "device=cuda",
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ColabPrepValueError(ValueError):
    """Raised when Colab experiment preparation parameters are invalid."""


def _validate_repo_url(repo_url: str) -> None:
    if repo_url != PUBLIC_REPO_URL:
        raise ColabPrepValueError(
            f"unsupported repo_url: {repo_url!r}. "
            f"Only the public Oczy repository {PUBLIC_REPO_URL!r} is accepted."
        )


def _validate_commit(source_commit: str) -> None:
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ColabPrepValueError(
            "source_commit must be a 40-character lowercase hex Git SHA; "
            f"dirty/short/tag/branch identifiers are rejected (got {source_commit!r})."
        )


def _validate_claim_class(claim_class: str) -> None:
    if claim_class not in _VALID_CLAIM_CLASSES:
        raise ColabPrepValueError(
            f"claim_class must be one of {sorted(_VALID_CLAIM_CLASSES)!r}, "
            f"got {claim_class!r}."
        )


def _validate_arguments(arguments: list[str]) -> None:
    if not isinstance(arguments, list) or not all(
        isinstance(a, str) for a in arguments
    ):
        raise ColabPrepValueError("arguments must be a list of strings.")
    lowered = [a.lower() for a in arguments]
    for arg in lowered:
        for pattern in _ACCELERATOR_PATTERNS:
            if pattern in arg:
                raise ColabPrepValueError(
                    f"accelerator argument {arg!r} is forbidden: "
                    "CPU-only contract applies to the target experiment. "
                    "Model-bearing jobs must route to Kaggle."
                )


def _validate_module(module: str) -> None:
    if not module or not isinstance(module, str):
        raise ColabPrepValueError("module must be a non-empty string.")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", module):
        raise ColabPrepValueError(
            f"module must be a valid dotted Python module path (got {module!r})."
        )


def _validate_model_artifact(artifact: dict[str, Any]) -> None:
    """Validate an optional model_artifact specification.

    Required fields: ``kind`` (``"gguf"`` or ``"hf_snapshot"``), ``repo_id``
    (non-empty str), ``revision`` (40-char lowercase hex), ``filename``
    (non-empty str), ``sha256`` (64-char lowercase hex).
    """
    if not isinstance(artifact, dict):
        raise ColabPrepValueError("model_artifact must be a dict.")
    required = ("kind", "repo_id", "revision", "filename", "sha256")
    missing = [f for f in required if f not in artifact]
    if missing:
        raise ColabPrepValueError(
            f"model_artifact missing required fields: {missing!r}."
        )
    kind = artifact["kind"]
    if kind not in _VALID_MODEL_ARTIFACT_KINDS:
        raise ColabPrepValueError(
            f"model_artifact kind must be one of {sorted(_VALID_MODEL_ARTIFACT_KINDS)!r}, "
            f"got {kind!r}."
        )
    repo_id = artifact["repo_id"]
    if not isinstance(repo_id, str) or not repo_id:
        raise ColabPrepValueError("model_artifact repo_id must be a non-empty string.")
    revision = artifact["revision"]
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise ColabPrepValueError(
            "model_artifact revision must be a 40-character lowercase hex Git SHA "
            f"(got {revision!r})."
        )
    filename = artifact["filename"]
    if not isinstance(filename, str) or not filename:
        raise ColabPrepValueError("model_artifact filename must be a non-empty string.")
    sha256 = artifact["sha256"]
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise ColabPrepValueError(
            "model_artifact sha256 must be a 64-character lowercase hex digest "
            f"(got {sha256!r})."
        )


# ---------------------------------------------------------------------------
# Bootstrap template
# ---------------------------------------------------------------------------

BOOTSTRAP_TEMPLATE = '''\
"""Generated Oczy Colab experiment bootstrap. Do not edit by hand.

Clones the public Oczy repository at an exact commit, verifies HEAD, sets a
strict CPU-only environment, prepends source paths, optionally provisions a
hash-verified model artifact from Hugging Face and/or installs a pinned CPU
llama-cpp-python wheel, then invokes
``infrastructure.kaggle.run_experiment_module`` with an explicit subprocess
argv.  The runner owns the structured execution report and METRIC/ASI parsing;
this bootstrap owns environment setup, commit verification, and model/package
provisioning only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# --- CPU-only offline contract: set before any heavy imports ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OCZY_REMOTE_CPU_ONLY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

JOB_SPEC = json.loads(__JOB_SPEC__)


def _run(argv: list[str], *, cwd: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run *argv* with explicit subprocess argv (no shell invocation)."""
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def clone_at_commit(repo_url: str, commit: str, dest: Path) -> Path:
    """Clone *repo_url* into *dest* and check out the exact *commit*.

    Uses ``git init`` + ``git fetch`` to avoid checkout assumptions.
    GitHub does not allow fetching an arbitrary commit SHA by name, so
    the strategy is: try a shallow fetch of the exact commit first
    (works on servers with ``uploadpack.allowReachableSHA1InWant``),
    then fall back to a blobless fetch of all refs (fast, partial clone)
    and check out the commit.  HEAD is verified against the requested SHA.
    """
    dest.mkdir(parents=True, exist_ok=True)
    init = _run(["git", "init", str(dest)])
    if init.returncode != 0:
        raise RuntimeError(
            f"git init failed (exit {init.returncode}): {init.stderr.strip()}"
        )
    # Strategy 1: shallow fetch of the exact commit (fastest if supported).
    shallow = _run(
        ["git", "fetch", "--depth=1", repo_url, commit],
        cwd=str(dest),
        timeout=600,
    )
    if shallow.returncode != 0:
        # Strategy 2: blobless fetch of all refs, then checkout the commit.
        # This downloads commit/tree objects but defers blob downloads
        # until checkout, fetching only the blobs needed for this tree.
        blobless = _run(
            ["git", "fetch", "--filter=blob:none", repo_url],
            cwd=str(dest),
            timeout=600,
        )
        if blobless.returncode != 0:
            raise RuntimeError(
                f"git fetch failed (shallow exit {shallow.returncode}: "
                f"{shallow.stderr.strip()}; blobless exit {blobless.returncode}: "
                f"{blobless.stderr.strip()})"
            )
    checkout = _run(["git", "checkout", commit], cwd=str(dest))
    if checkout.returncode != 0:
        raise RuntimeError(
            f"git checkout {commit[:12]} failed (exit {checkout.returncode}): "
            f"{checkout.stderr.strip()}"
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=str(dest))
    if head.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed (exit {head.returncode}): "
            f"{head.stderr.strip()}"
        )
    actual = head.stdout.strip()
    if actual != commit:
        raise RuntimeError(
            f"HEAD mismatch: expected {commit}, got {actual}. Refusing to proceed."
        )
    return dest


def add_source_paths(repo_root: Path) -> None:
    """Prepend repo root, repo/src, and workspace-package src dirs to sys.path."""
    paths = [repo_root, repo_root / "src"]
    paths.extend(sorted(repo_root.glob("*/src")))
    for path in reversed(paths):
        if path.is_dir():
            sys.path.insert(0, str(path))


def hardware() -> dict:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def write_provenance(payload: dict) -> None:
    path = Path("/content/colab_bootstrap_provenance.json")
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provision_model_artifact(artifact: dict) -> dict:
    """Download and verify a model artifact from Hugging Face.

    Temporarily permits HF network access for the exact download only,
    verifies the SHA-256 of the downloaded file(s), sets the appropriate
    model environment variable, then forces offline mode back on.

    For ``kind=="gguf"``: downloads a single file via ``hf_hub_download``
    at the exact revision, verifies its SHA-256, and sets
    ``OCZY_MODEL_PATH``.

    For ``kind=="hf_snapshot"``: downloads a snapshot at the exact revision
    via ``snapshot_download``, verifies the named file's SHA-256, and sets
    ``OCZY_HF_MODEL_DIR``.

    Fails closed on any hash mismatch or download error.
    """
    kind = artifact["kind"]
    repo_id = artifact["repo_id"]
    revision = artifact["revision"]
    filename = artifact["filename"]
    expected_sha = artifact["sha256"]

    # Temporarily permit HF network for exact download only.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"huggingface_hub is required for model_artifact provisioning "
            f"but is not installed: {exc}"
        ) from exc

    try:
        if kind == "gguf":
            local_path = hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
            )
            actual_sha = _sha256_file(Path(local_path))
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected_sha}, "
                    f"got {actual_sha}. Refusing to proceed."
                )
            os.environ["OCZY_MODEL_PATH"] = str(local_path)
            return {
                "kind": "gguf",
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "sha256": actual_sha,
                "sha256_verified": True,
                "model_path": str(local_path),
                "env_var": "OCZY_MODEL_PATH",
            }
        else:  # hf_snapshot
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
            )
            target_file = Path(local_dir) / filename
            if not target_file.is_file():
                raise RuntimeError(
                    f"Expected file {filename} not found in snapshot for "
                    f"{repo_id}@{revision}."
                )
            actual_sha = _sha256_file(target_file)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected_sha}, "
                    f"got {actual_sha}. Refusing to proceed."
                )
            os.environ["OCZY_HF_MODEL_DIR"] = str(local_dir)
            return {
                "kind": "hf_snapshot",
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "sha256": actual_sha,
                "sha256_verified": True,
                "model_dir": str(local_dir),
                "env_var": "OCZY_HF_MODEL_DIR",
            }
    finally:
        # Force offline mode back on after download.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def install_llama_cpp() -> dict:
    """Install the pinned CPU llama-cpp-python wheel via explicit argv.

    Uses ``[sys.executable, '-m', 'pip', 'install',
    'llama-cpp-python==0.3.31', '--extra-index-url',
    'https://abetlen.github.io/llama-cpp-python/whl/cpu']`` — no shell
    invocation, no arbitrary pip args.  Fails closed on install error.
    """
    argv = [
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python==0.3.31",
        "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    ]
    proc = _run(argv, timeout=600)
    result = {
        "package": "llama-cpp-python",
        "version": "0.3.31",
        "wheel_index": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        "install_command": argv,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-2000:] if proc.stdout else "",
        "stderr": proc.stderr[-2000:] if proc.stderr else "",
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cpp-python install failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    return result


def main() -> int:
    report: dict = {
        "schema_version": "oczy/colab-bootstrap-provenance/v1",
        "job_spec": JOB_SPEC,
        "status": "starting",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware(),
        "cpu_only_contract": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "oczy_remote_cpu_only": os.environ.get("OCZY_REMOTE_CPU_ONLY"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    write_provenance(report)
    try:
        repo_root = clone_at_commit(
            JOB_SPEC["repo_url"],
            JOB_SPEC["source_commit"],
            Path("/content/oczy"),
        )
        add_source_paths(repo_root)
        os.chdir(repo_root)

        report.update(
            {
                "status": "running",
                "repo_root": str(repo_root),
                "head_commit": JOB_SPEC["source_commit"],
            }
        )
        write_provenance(report)

        # --- Optional model artifact provisioning ---
        model_artifact = JOB_SPEC.get("model_artifact")
        if model_artifact is not None:
            artifact_info = provision_model_artifact(model_artifact)
            report["model_artifact"] = artifact_info
            write_provenance(report)

        # --- Optional pinned CPU llama-cpp-python install ---
        if JOB_SPEC.get("install_llama_cpp"):
            install_info = install_llama_cpp()
            report["llama_cpp_install"] = install_info
            write_provenance(report)

        runner_argv = [
            sys.executable, "-m", "infrastructure.kaggle.run_experiment_module",
            "--module", JOB_SPEC["module"],
            "--source-commit", JOB_SPEC["source_commit"],
            "--provider", "colab",
            "--job-name", JOB_SPEC["job_name"],
            "--report", "execution_report.json",
        ]
        for arg in JOB_SPEC["arguments"]:
            runner_argv.extend(["--arg", arg])
        if JOB_SPEC.get("timeout") is not None:
            runner_argv.extend(["--timeout", str(JOB_SPEC["timeout"])])

        proc = subprocess.run(
            runner_argv,
            cwd=str(repo_root),
            check=False,
        )
        report.update(
            {
                "status": "complete" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "runner_command": runner_argv,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        write_provenance(report)
        return proc.returncode
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        report.update(
            {
                "status": "complete" if code == 0 else "error",
                "exit_code": code,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        write_provenance(report)
        return code
    except Exception as error:
        report.update(
            {
                "status": "error",
                "exit_code": 1,
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        write_provenance(report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_colab_experiment(
    *,
    output: Path,
    job_name: str,
    repo_url: str,
    source_commit: str,
    module: str,
    arguments: list[str],
    phase: str,
    claim_class: str,
    output_path: str,
    timeout: float | None = None,
    force: bool = False,
    model_artifact: dict[str, Any] | None = None,
    install_llama_cpp: bool = False,
) -> dict[str, Any]:
    """Prepare a self-contained Colab bootstrap for one remote experiment job.

    Writes ``colab_bootstrap.py`` and ``job_spec.json`` into *output*.

    Parameters
    ----------
    output:
        Directory for generated artifacts (created if absent).
    job_name:
        Unique job identifier for batch/scheduler and report labelling.
    repo_url:
        Must be exactly ``https://github.com/KinoThe-Kafkaesque/oczy.git``.
    source_commit:
        Exact 40-character lowercase hex Git SHA. Dirty/short/tag/branch
        identifiers are rejected.
    module:
        Dotted Python module path executed by the runner
        (e.g. ``"oczy.experiments.layer_l_probe"``).
    arguments:
        List of string arguments passed verbatim to the target module.
        Accelerator arguments are rejected anywhere in the list.
    phase:
        Research phase label (e.g. ``"instrument"``, ``"analysis"``).
    claim_class:
        Either ``"scientific"`` or ``"infrastructure"``.
    output_path:
        Expected output path on the runner, recorded in the job spec for the
        provider collector.
    timeout:
        Optional job timeout in seconds.
    force:
        Overwrite existing generated files if True.
    model_artifact:
        Optional dict specifying a Hugging Face model artifact to download
        and hash-verify at bootstrap time.  Required fields when present:
        ``kind`` (``"gguf"`` or ``"hf_snapshot"``), ``repo_id`` (non-empty
        str), ``revision`` (40-char lowercase hex), ``filename`` (non-empty
        str), ``sha256`` (64-char lowercase hex).  When ``None`` (default),
        no model provisioning occurs and the job spec omits the key — pure
        NumPy jobs are unchanged.
    install_llama_cpp:
        If True, the generated bootstrap installs
        ``llama-cpp-python==0.3.31`` from the abetlen CPU wheel index via
        explicit argv (no shell invocation, no arbitrary pip args).  Default
        False.

    Returns
    -------
    dict
        The job specification dict (same as written to ``job_spec.json``).

    Raises
    ------
    ColabPrepValueError
        If any parameter fails validation.
    FileExistsError
        If generated files already exist and *force* is False.
    """
    # --- Validate all inputs ---
    if not job_name or not isinstance(job_name, str):
        raise ColabPrepValueError("job_name must be a non-empty string.")
    _validate_repo_url(repo_url)
    _validate_commit(source_commit)
    _validate_module(module)
    _validate_arguments(arguments)
    _validate_claim_class(claim_class)
    if not phase or not isinstance(phase, str):
        raise ColabPrepValueError("phase must be a non-empty string.")
    if not output_path or not isinstance(output_path, str):
        raise ColabPrepValueError("output_path must be a non-empty string.")
    if timeout is not None and (
        not isinstance(timeout, (int, float)) or timeout <= 0
    ):
        raise ColabPrepValueError("timeout must be a positive number or None.")
    if model_artifact is not None:
        _validate_model_artifact(model_artifact)
    if not isinstance(install_llama_cpp, bool):
        raise ColabPrepValueError("install_llama_cpp must be a boolean.")

    # --- Prepare output directory ---
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = [output / "colab_bootstrap.py", output / "job_spec.json"]
    if any(path.exists() for path in generated) and not force:
        raise FileExistsError(
            f"refusing to overwrite generated files in {output}"
        )

    # --- Build job spec ---
    job_spec: dict[str, Any] = {
        "schema_version": JOB_SPEC_SCHEMA_VERSION,
        "provider": "colab",
        "job_name": job_name,
        "repo_url": repo_url,
        "source_commit": source_commit,
        "module": module,
        "arguments": list(arguments),
        "phase": phase,
        "claim_class": claim_class,
        "output_path": output_path,
        "timeout": float(timeout) if timeout is not None else None,
        "install_llama_cpp": install_llama_cpp,
    }
    if model_artifact is not None:
        job_spec["model_artifact"] = model_artifact

    # --- Render bootstrap ---
    rendered_spec = json.dumps(job_spec, sort_keys=True)
    bootstrap_code = BOOTSTRAP_TEMPLATE.replace(
        "__JOB_SPEC__", repr(rendered_spec)
    )
    (output / "colab_bootstrap.py").write_text(bootstrap_code, encoding="utf-8")
    _write_json(output / "job_spec.json", job_spec)

    return job_spec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument(
        "--repo-url",
        default=PUBLIC_REPO_URL,
        help=f"Repository URL (default: {PUBLIC_REPO_URL})",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument(
        "--arg",
        dest="arguments",
        action="append",
        default=[],
        help="Repeatable argument passed to the target module.",
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--claim-class",
        choices=sorted(_VALID_CLAIM_CLASSES),
        required=True,
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--model-artifact",
        default=None,
        help=(
            "JSON string specifying a model artifact: "
            '{"kind":"gguf"|"hf_snapshot","repo_id":"...","revision":"<40hex>",'
            '"filename":"...","sha256":"<64hex>"}'
        ),
    )
    parser.add_argument(
        "--install-llama-cpp",
        action="store_true",
        default=False,
        help="Install pinned llama-cpp-python==0.3.31 from abetlen CPU wheel index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_artifact = None
    if args.model_artifact is not None:
        try:
            model_artifact = json.loads(args.model_artifact)
        except json.JSONDecodeError as exc:
            print(f"error: --model-artifact must be valid JSON: {exc}", file=sys.stderr)
            return 2
    spec = prepare_colab_experiment(
        output=args.output,
        job_name=args.job_name,
        repo_url=args.repo_url,
        source_commit=args.source_commit,
        module=args.module,
        arguments=args.arguments,
        phase=args.phase,
        claim_class=args.claim_class,
        output_path=args.output_path,
        timeout=args.timeout,
        force=args.force,
        model_artifact=model_artifact,
        install_llama_cpp=args.install_llama_cpp,
    )
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
