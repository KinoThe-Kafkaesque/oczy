"""Generate a private, provenance-checked Kaggle research kernel.

The generated bootstrap verifies a commit-addressed Oczy source archive,
resolves the attached model via manifest-directed identity (no hard-coded
filename/config heuristics), disables network-backed model resolution,
observes the local runtime manifest, compares it against the expected
identity, records runtime provenance, and then executes the common runner
with both expected and observed manifests.
It does not submit the kernel; generation, review, and submission are separate
steps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "oczy/kaggle-research-job/v2"
PROFILES = {
    "cpu": {"enable_gpu": False, "machine_shape": ""},
}
PHASES = ("instrument", "oracle", "development", "meta-test", "analysis")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _title_slug(title: str) -> str:
    """Kaggle clean-URL title slug.

    Lowercase, collapse each run of non-ASCII-alphanumeric characters to a
    single hyphen, and strip leading/trailing hyphens.  This mirrors how
    Kaggle derives the final path component of a kernel URL from its title.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


BOOTSTRAP_TEMPLATE = '''\
"""Generated Oczy Kaggle research bootstrap. Do not edit by hand."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import runpy
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

# --- CPU-only offline contract: must be set before importing torch ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OCZY_REMOTE_CPU_ONLY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch


JOB_SPEC = json.loads(__JOB_SPEC__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(payload: dict) -> None:
    path = Path("/kaggle/working/remote_run_provenance.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def find_source() -> tuple[Path, dict]:
    matches = []
    for manifest_path in Path("/kaggle/input").rglob("source_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_info = manifest.get("archive", {})
        if (
            manifest.get("commit") == JOB_SPEC["source_commit"]
            and archive_info.get("sha256") == JOB_SPEC["source_archive_sha256"]
        ):
            matches.append((manifest_path, manifest))
    if len(matches) != 1:
        raise RuntimeError(f"expected one pinned source manifest, found {len(matches)}")
    manifest_path, manifest = matches[0]
    archive = manifest_path.parent / manifest["archive"]["filename"]
    actual = sha256_file(archive)
    if actual != JOB_SPEC["source_archive_sha256"]:
        raise RuntimeError(f"source archive hash mismatch: {actual}")
    return archive, manifest


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_resolved):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")
    root = destination / "oczy"
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("source archive did not contain oczy/pyproject.toml")
    return root


def add_source_paths(root: Path) -> None:
    # The archive root (root.parent) must be on sys.path so that
    # ``infrastructure.kaggle.runtime_manifest`` and other repo-level
    # packages are importable from extracted source, not from a
    # pre-installed infrastructure package.
    paths = [root.parent, root, root / "src"]
    paths.extend(sorted(root.glob("*/src")))
    valid = [str(p) for p in paths if p.is_dir()]
    for path in reversed(valid):
        sys.path.insert(0, path)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(valid + ([existing] if existing else []))


def hardware() -> dict:
    data = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": False,
        "cuda_device_count": 0,
        "torch_cuda_version": torch.version.cuda,
    }
    return data


def _resolve_model_from_manifest(expected_manifest: dict) -> Path | None:
    """Resolve the model root from expected artifact manifest entries.

    For each artifact file in the expected manifest, scans
    ``/kaggle/input`` for files/directories whose relative name,
    size, and SHA-256 match the expected entry.  Requires exactly one
    consistent candidate root and returns it.  Returns ``None`` for
    model-free jobs.
    """
    model_block = expected_manifest["model"]
    convention = model_block["resolved_model_convention"]
    if convention == "none":
        return None

    artifact_files = model_block["artifact_files"]
    first_file = artifact_files[0]
    first_name = first_file["path"]
    first_sha = first_file["sha256"]
    first_size = first_file["size_bytes"]

    candidates: list[Path] = []
    # Search for the first artifact file by path and size.
    for root_path in Path("/kaggle/input").glob("*"):
        if not root_path.is_dir():
            continue
        for candidate_path in root_path.rglob(first_name):
            if not candidate_path.is_file():
                continue
            actual_size = candidate_path.stat().st_size
            if actual_size != first_size:
                continue
            actual_sha = sha256_file(candidate_path)
            if actual_sha != first_sha:
                continue
            candidates.append(candidate_path.resolve())

    unique = sorted(set(candidates), key=str)
    if len(unique) == 0:
        raise RuntimeError(
            f"no candidate found for manifest file {first_name!r} "
            f"(sha256={first_sha[:12]}..., size={first_size})"
        )
    if len(unique) != 1:
        rendered = ", ".join(str(p) for p in unique)
        raise RuntimeError(
            f"ambiguous candidates for {first_name!r}: found "
            f"{len(unique)}: {rendered}"
        )

    # Determine the model root from the first matched file.
    matched = unique[0]
    if convention == "llama-cpp-gguf-file":
        # Single GGUF file — the file itself is the root.
        return matched
    else:
        # transformers-pretrained-directory — the parent directory.
        return matched.parent


def _set_model_env_vars(model_root: Path | None, convention: str) -> None:
    """Set convention-appropriate model environment variables."""
    if model_root is None:
        return
    if convention == "llama-cpp-gguf-file":
        os.environ["OCZY_MODEL_PATH"] = str(model_root)
    elif convention == "transformers-pretrained-directory":
        os.environ["OCZY_HF_MODEL_DIR"] = str(model_root)


def main() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    profile = JOB_SPEC.get("profile")
    if cuda_visible != "" or profile != "cpu":
        raise RuntimeError(
            "CPU-only contract violated: CUDA_VISIBLE_DEVICES must be empty "
            "and profile must be 'cpu'. Refusing to run with GPU access."
        )
    cuda_available = False
    report = {
        "schema_version": JOB_SPEC["schema_version"],
        "job_spec": JOB_SPEC,
        "status": "starting",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware(),
        "cpu_only_contract": {
            "profile": JOB_SPEC.get("profile"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_available": cuda_available,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "oczy_remote_cpu_only": os.environ.get("OCZY_REMOTE_CPU_ONLY"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    write_report(report)
    try:
        archive, source_manifest = find_source()
        source_root = safe_extract(archive, Path(tempfile.mkdtemp()))
        add_source_paths(source_root)

        # --- Manifest-directed model resolution and observation ---
        expected_manifest = JOB_SPEC["runtime_manifest"]
        import importlib.util as _ilu
        _rm_path = source_root / "infrastructure" / "kaggle" / "runtime_manifest.py"
        _spec = _ilu.spec_from_file_location("_runtime_manifest", _rm_path)
        assert _spec is not None and _spec.loader is not None, f"runtime_manifest.py not found at {_rm_path}"
        _rm_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_rm_mod)
        compare_runtime_manifests = _rm_mod.compare_runtime_manifests
        observe_runtime_manifest = _rm_mod.observe_runtime_manifest
        RuntimeManifestError = _rm_mod.RuntimeManifestError
        strict_canonical_json = _rm_mod.strict_canonical_json

        convention = expected_manifest["model"]["resolved_model_convention"]
        model_root = _resolve_model_from_manifest(expected_manifest)
        _set_model_env_vars(model_root, convention)

        try:
            observed_manifest = observe_runtime_manifest(
                model_root=model_root,
                logical_model_id=expected_manifest["model"]["logical_model_id"],
                resolved_model_convention=convention,
                generation_config=expected_manifest["greedy_generation"],
                quantization=expected_manifest["model"]["quantization"],
            )
        except RuntimeManifestError as exc:
            report.update(
                {
                    "status": "observation_failure",
                    "error": str(exc),
                    "finished_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
            )
            write_report(report)
            return 1

        mismatches = compare_runtime_manifests(expected_manifest, observed_manifest)

        report.update(
            {
                "status": "running",
                "source_manifest": source_manifest,
                "source_root": str(source_root),
                "model_root": str(model_root) if model_root is not None else None,
                "expected_runtime_manifest_sha256": expected_manifest["manifest_sha256"],
                "observed_runtime_manifest_sha256": observed_manifest["manifest_sha256"],
                "runtime_manifest_mismatches": mismatches[:20] if mismatches else [],
            }
        )
        write_report(report)
        os.chdir(source_root)

        # Append observed runtime manifest to runner argv.
        observed_json = strict_canonical_json(observed_manifest).decode("utf-8")
        sys.argv = [JOB_SPEC["module"], *JOB_SPEC["arguments"]]
        sys.argv.extend(["--observed-manifest-json", observed_json])
        runpy.run_module(JOB_SPEC["module"], run_name="__main__", alter_sys=True)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        report["status"] = "complete" if code == 0 else "error"
        report["exit_code"] = code
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_report(report)
        return code
    except Exception as error:
        report["status"] = "error"
        report["exit_code"] = 1
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["traceback"] = traceback.format_exc()
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_report(report)
        raise
    report["status"] = "complete"
    report["exit_code"] = 0
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_kernel(
    *,
    output: Path,
    kernel_id: str,
    title: str,
    phase: str,
    profile: str,
    source_dataset: str,
    source_commit: str,
    source_archive_sha256: str,
    module: str,
    arguments: list[str],
    model_source: str | None,
    instrument_manifest_sha256: str | None,
    human_signoff_id: str | None,
    runtime_manifest: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase 40-character Git SHA")
    if not SHA256_PATTERN.fullmatch(source_archive_sha256):
        raise ValueError("source archive hash must be a lowercase SHA-256")
    if source_commit[:12] not in source_dataset:
        raise ValueError("source dataset slug must include the commit's first 12 characters")
    if phase == "meta-test" and not (instrument_manifest_sha256 and human_signoff_id):
        raise ValueError("meta-test generation requires manifest hash and human sign-off ID")
    if instrument_manifest_sha256 and not SHA256_PATTERN.fullmatch(instrument_manifest_sha256):
        raise ValueError("instrument manifest hash must be a lowercase SHA-256")
    title_slug = _title_slug(title)
    id_slug = kernel_id.rsplit("/", 1)[-1]
    if title_slug != id_slug:
        raise ValueError(
            f"title {title!r} resolves to slug {title_slug!r} but kernel_id "
            f"{kernel_id!r} ends with {id_slug!r}; Kaggle would create the "
            f"kernel under a different slug and polling the requested id would fail"
        )

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = [output / "run.py", output / "kernel-metadata.json", output / "job_spec.json"]
    if any(path.exists() for path in generated) and not force:
        raise FileExistsError(f"refusing to overwrite generated files in {output}")

    job_spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "profile": profile,
        "source_dataset": source_dataset,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "model_source": model_source,
        "module": module,
        "arguments": arguments,
        "instrument_manifest_sha256": instrument_manifest_sha256,
        "human_signoff_id": human_signoff_id,
        "runtime_manifest": runtime_manifest,
    }
    rendered_spec = json.dumps(job_spec, sort_keys=True)
    (output / "run.py").write_text(
        BOOTSTRAP_TEMPLATE.replace("__JOB_SPEC__", repr(rendered_spec)),
        encoding="utf-8",
    )
    _write_json(output / "job_spec.json", job_spec)
    _write_json(
        output / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": title,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": "",
            "dataset_sources": [source_dataset],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [model_source] if model_source else [],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="cpu")
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--arg", dest="arguments", action="append", default=[])
    parser.add_argument("--model-source")
    parser.add_argument("--instrument-manifest-sha256")
    parser.add_argument("--human-signoff-id")
    parser.add_argument(
        "--runtime-manifest",
        required=True,
        help="Path to runtime manifest JSON file.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_manifest_path = Path(args.runtime_manifest)
    if not runtime_manifest_path.is_file():
        print(f"error: runtime manifest not found: {runtime_manifest_path}", file=sys.stderr)
        return 2
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: invalid runtime manifest JSON: {exc}", file=sys.stderr)
        return 2
    spec = prepare_kernel(
        output=args.output,
        kernel_id=args.kernel_id,
        title=args.title,
        phase=args.phase,
        profile=args.profile,
        source_dataset=args.source_dataset,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
        module=args.module,
        arguments=args.arguments,
        model_source=args.model_source,
        instrument_manifest_sha256=args.instrument_manifest_sha256,
        human_signoff_id=args.human_signoff_id,
        runtime_manifest=runtime_manifest,
        force=args.force,
    )
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
