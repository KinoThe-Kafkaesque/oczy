"""Generate a private, provenance-checked Kaggle research kernel.

The generated bootstrap verifies a commit-addressed Oczy source archive,
discovers the attached version-pinned model, disables network-backed model
resolution, records runtime provenance, and then executes one Python module.
It does not submit the kernel; generation, review, and submission are separate
steps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "oczy/kaggle-research-job/v1"
DEFAULT_MODEL_SOURCE = "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
PROFILES = {
    "cpu": {"enable_gpu": False, "machine_shape": ""},
    "t4": {"enable_gpu": True, "machine_shape": "NvidiaTeslaT4"},
}
PHASES = ("instrument", "oracle", "development", "meta-test", "analysis")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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
import time
import traceback
from pathlib import Path

import torch


JOB_SPEC = __JOB_SPEC__


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


def find_model() -> Path | None:
    if not JOB_SPEC.get("model_source"):
        return None
    matches = []
    for config_path in Path("/kaggle/input").rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("model_type") == "qwen2" and config.get("hidden_size") == 896:
            candidate = config_path.parent.resolve()
            if (candidate / "model.safetensors").is_file():
                matches.append(candidate)
    unique = sorted(set(matches), key=str)
    if len(unique) != 1:
        raise RuntimeError(f"expected one attached Qwen model, found {len(unique)}")
    return unique[0]


def add_source_paths(root: Path) -> None:
    paths = [root, root / "src"]
    paths.extend(sorted(root.glob("*/src")))
    for path in reversed(paths):
        if path.is_dir():
            sys.path.insert(0, str(path))


def hardware() -> dict:
    data = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        data["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "compute_capability": (
                    f"{torch.cuda.get_device_properties(index).major}."
                    f"{torch.cuda.get_device_properties(index).minor}"
                ),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return data


def main() -> int:
    report = {
        "schema_version": JOB_SPEC["schema_version"],
        "job_spec": JOB_SPEC,
        "status": "starting",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware(),
    }
    write_report(report)
    try:
        archive, source_manifest = find_source()
        source_root = safe_extract(archive, Path("/kaggle/working/source"))
        add_source_paths(source_root)
        model_dir = find_model()
        if model_dir is not None:
            os.environ["OCZY_MODEL_DIR"] = str(model_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        report.update(
            {
                "status": "running",
                "source_manifest": source_manifest,
                "source_root": str(source_root),
                "model_dir": str(model_dir) if model_dir is not None else None,
            }
        )
        write_report(report)
        sys.argv = [JOB_SPEC["module"], *JOB_SPEC["arguments"]]
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

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = [output / "run.py", output / "kernel-metadata.json", output / "job_spec.json"]
    if any(path.exists() for path in generated) and not force:
        raise FileExistsError(f"refusing to overwrite generated files in {output}")

    profile_data = PROFILES[profile]
    effective_model_source = model_source
    if effective_model_source is None and phase in {"oracle", "development", "meta-test"}:
        effective_model_source = DEFAULT_MODEL_SOURCE
    job_spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "profile": profile,
        "source_dataset": source_dataset,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "model_source": effective_model_source,
        "module": module,
        "arguments": arguments,
        "instrument_manifest_sha256": instrument_manifest_sha256,
        "human_signoff_id": human_signoff_id,
    }
    rendered_spec = json.dumps(job_spec, sort_keys=True)
    (output / "run.py").write_text(
        BOOTSTRAP_TEMPLATE.replace("__JOB_SPEC__", rendered_spec),
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
            "enable_gpu": profile_data["enable_gpu"],
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": profile_data["machine_shape"],
            "dataset_sources": [source_dataset],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [effective_model_source] if effective_model_source else [],
        },
    )
    return job_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--arg", dest="arguments", action="append", default=[])
    parser.add_argument("--model-source")
    parser.add_argument("--instrument-manifest-sha256")
    parser.add_argument("--human-signoff-id")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        force=args.force,
    )
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
