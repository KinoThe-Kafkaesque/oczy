"""Reproduce one recorded R20 DEV cell without weakening its legacy hash gate.

The historical organ digest includes Hugging Face's ``_name_or_path``.
Bind verified local model files to the provenance-recorded Kaggle path in a
private mount namespace. Source, model, runtime, checkpoint and scorer gates
remain in force. No network, remote submission or sealed evaluation is used.
Run with the Python interpreter specified by the recorded runtime manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.kaggle.runtime_manifest import (  # noqa: E402
    observe_runtime_manifest,
    validate_runtime_manifest,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_options(provenance: dict) -> dict[str, str]:
    job = provenance["job_spec"]
    if (
        provenance.get("exit_code") != 0
        or job["phase"] != "development"
        or job["module"] != "oczy.experiments.meta_cortex"
        or job["profile"] != "cpu"
        or job["arguments"][0] != "collect-calibration-shard"
    ):
        raise ValueError("Requires a successful CPU DEV calibration provenance record")
    args = job["arguments"][1:]
    if len(args) % 2 or len(set(args[::2])) != len(args[::2]):
        raise ValueError("Malformed or duplicate recorded arguments")
    return dict(zip(args[::2], args[1::2], strict=True))


def validate_cell(options: dict[str, str], task: int, evaluation_seed: int) -> None:
    if not int(options["--task-start"]) <= task < int(options["--task-end"]):
        raise ValueError("Requested task is outside the recorded DEV shard")
    if evaluation_seed not in {int(x) for x in options["--eval-seed-indices"].split(",")}:
        raise ValueError("Requested evaluation seed is outside the recorded DEV shard")


def materialize_model(snapshot: Path, destination: Path, artifacts: list[dict]) -> None:
    """Dereference HF snapshot links, verifying every pinned artifact first."""
    destination.mkdir()
    for artifact in artifacts:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe model artifact path")
        source = snapshot / relative
        if source.stat().st_size != artifact["size_bytes"] or sha256(source) != artifact["sha256"]:
            raise ValueError(f"Pinned model artifact mismatch: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # The child sees this tree read-only. Copies avoid aliasing writable
        # output files to the user's model cache.
        shutil.copyfile(source, target)


def namespace_prefix(model: Path, model_root: str, output: Path) -> list[str]:
    remote = Path(model_root)
    if not remote.is_absolute() or ".." in remote.parts or not remote.is_relative_to("/kaggle/input/models"):
        raise ValueError("Expected a provenance-recorded /kaggle/input/models path")
    return [
        "bwrap", "--die-with-parent", "--unshare-net", "--tmpfs", "/",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc", "--ro-bind", "/home", "/home",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--bind", str(output), str(output),
        "--ro-bind", str(model), str(model),
        "--ro-bind", str(model), model_root,
    ]


def compare_cell(local: dict, reference: dict) -> dict:
    """Compare all scientific and audit fields, preserving duplicate detection."""
    from collections import Counter

    headers = ("schema", "definition_sha256", "calibration_view_sha256", "scorer_sha256", "organ_hash")
    if len(local["seed_cell_records"]) != 1:
        raise ValueError("Expected exactly one local seed cell")
    cell = local["seed_cell_records"][0]
    dev = cell["developmental_seed_index"]
    evaluation = cell["evaluation_seed_index"]
    rule = cell["rule_fingerprint"]
    selected = {
        "seed_cell_records": [r for r in reference["seed_cell_records"]
                              if (r["developmental_seed_index"], r["evaluation_seed_index"], r["rule_fingerprint"])
                              == (dev, evaluation, rule)],
        "no_update_repeat_records": [r for r in reference["no_update_repeat_records"]
                                     if (r["developmental_seed_index"], r["rule_fingerprint"]) == (dev, rule)],
        "theta_hashes": [r for r in reference["theta_hashes"] if r["developmental_seed_index"] == dev],
    }
    matched = {key: local[key] == reference[key] for key in headers}
    def encode(rows):
        return Counter(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows)

    for key, records in selected.items():
        matched[key] = bool(records) and encode(local[key]) == encode(records)
    return {"all_fields_equal": all(matched.values()), "field_groups": matched,
            "local_repeat_count": len(local["no_update_repeat_records"]),
            "reference_repeat_count": len(selected["no_update_repeat_records"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "source", "model-dir", "calibration-view", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--evaluation-seed-index", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--reference-shard", type=Path)
    args = parser.parse_args(argv)
    provenance = json.loads(args.provenance.read_text())
    options = recorded_options(provenance)
    validate_cell(options, args.task_index, args.evaluation_seed_index)
    source = args.source.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != provenance["source_manifest"]["commit"]:
        raise ValueError("Source commit differs from remote provenance")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise ValueError("Reproduction source must be clean")
    expected = validate_runtime_manifest(provenance["job_spec"]["runtime_manifest"])
    output = args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output and clean source must be disjoint directories")
    output.mkdir(parents=True, exist_ok=False)
    model = output / "model"
    materialize_model(args.model_dir, model, expected["model"]["artifact_files"])
    observed = observe_runtime_manifest(
        model_root=model,
        logical_model_id=expected["model"]["logical_model_id"],
        resolved_model_convention=expected["model"]["resolved_model_convention"],
        generation_config=expected["greedy_generation"],
        quantization=expected["model"]["quantization"],
    )
    if observed != expected:
        differing = [key for key in expected if observed.get(key) != expected[key]]
        raise ValueError(f"Exact runtime/model manifest mismatch in {differing}")
    model_root = provenance["model_root"]
    command = namespace_prefix(model, model_root, output)
    command += ["--ro-bind", str(source), str(source), "--chdir", str(source), "--", sys.executable, "-m", "oczy.experiments.meta_cortex"]
    command += [
        "collect-calibration-shard", "--calibration-view", str(args.calibration_view.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()), "--model-id", options["--model-id"],
        "--organ-hash", options["--organ-hash"], "--dev-seed-index", options["--dev-seed-index"],
        "--eval-seed-indices", str(args.evaluation_seed_index),
        "--task-start", str(args.task_index), "--task-end", str(args.task_index + 1),
        "--output", str(output / "shard.json"),
    ]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OCZY_REMOTE_CPU_ONLY": "1", "OCZY_MODEL_DIR": model_root,
        "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
    })
    report = {
        "schema_version": "oczy/r20-local-reproduction/v1",
        "source_commit": commit, "runtime_manifest_sha256": observed["manifest_sha256"],
        "provenance_sha256": sha256(args.provenance), "model_root": model_root,
        "expected_organ_hash": options["--organ-hash"], "command": command,
        "task_index": args.task_index, "evaluation_seed_index": args.evaluation_seed_index,
        "developmental_seed_index": int(options["--dev-seed-index"]),
        "network_disabled": True, "meta_test_accessed": False,
    }
    started = time.monotonic()
    print("Exact runtime and model artifacts verified; running recorded DEV cell", flush=True)
    try:
        with (output / "run.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        report["exit_code"] = result.returncode
    except subprocess.TimeoutExpired:
        report["exit_code"] = 124
        report["error"] = "Timed out; no scientific verdict"
    report["elapsed_seconds"] = time.monotonic() - started
    report["shard_exists"] = (output / "shard.json").is_file()
    if report["exit_code"] == 0 and args.reference_shard is not None:
        report["reference_shard_sha256"] = sha256(args.reference_shard)
        report["comparison"] = compare_cell(
            json.loads((output / "shard.json").read_text()),
            json.loads(args.reference_shard.read_text()),
        )
        if not report["comparison"]["all_fields_equal"]:
            report["exit_code"] = 1
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "command"}, indent=2))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
