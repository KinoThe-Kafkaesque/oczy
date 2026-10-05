"""Run each frozen capability phase in a separate offline, read-only-source process."""

from __future__ import annotations

import argparse
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
from scripts.capability_validation_contract import manifest_at, write_json  # noqa: E402
from scripts.reproduce_r20_dev import namespace_prefix  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "output", "model", "provenance"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    manifest = manifest_at(root)
    provenance = json.loads(args.provenance.read_text())
    expected = validate_runtime_manifest(provenance["job_spec"]["runtime_manifest"])
    observed = observe_runtime_manifest(model_root=args.model, logical_model_id=expected["model"]["logical_model_id"],
                                        resolved_model_convention=expected["model"]["resolved_model_convention"],
                                        generation_config=expected["greedy_generation"], quantization=expected["model"]["quantization"])
    if observed != expected or observed["manifest_sha256"] != manifest["runtime_manifest_sha256"]:
        raise ValueError("Pinned runtime/model artifacts mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_manifest.json", observed)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
                "OCZY_MODEL_DIR": provenance["model_root"], "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo / "src")})
    records = []
    for phase in ("reference", "train", "restore", "actions"):
        prefix = namespace_prefix(args.model.resolve(), provenance["model_root"], output)
        hidden = [root / "audit"]
        if phase == "train":
            hidden += [root / role for role in ("probes", "oracle", "actions")]
            hidden += [output / "reference", output / "released_text"]
        if phase in ("restore", "actions"):
            hidden += [root / "training", root / "oracle", output / "reference", output / "train"]
        if phase == "restore":
            hidden += [root / "actions", output / "released_text"]
        if phase == "actions":
            hidden += [root / "probes", output / "restore"]
        forbidden = []
        for directory in hidden:
            if directory.exists():
                forbidden += [str(p) for p in directory.rglob("*") if p.is_file()]
            prefix += ["--tmpfs", str(directory)]
        # Data builder and its bytecode could reconstruct gold inputs; mask both.
        for name in ("scripts/prepare_capability_validation.py",):
            prefix += ["--ro-bind", "/dev/null", str(repo / name)]
        prefix += ["--tmpfs", str(repo / "scripts/__pycache__"), "--chdir", str(repo), "--"]
        check = ("from pathlib import Path; import json; paths=" + repr(forbidden) +
                 "; visible=[p for p in paths if Path(p).is_file()]; print(json.dumps({'forbidden_files_checked':len(paths),'visible_forbidden_files':visible})); assert not visible")
        checked = subprocess.run(prefix + [sys.executable, "-c", check], env=env, text=True, capture_output=True, check=True)
        command = prefix + [sys.executable, str(repo / "scripts/capability_validation_worker.py"), phase,
                            "--root", str(root), "--output", str(output / phase)]
        if phase in ("restore", "actions"):
            command += ["--artifacts", str(output / "released_states")]
        if phase == "actions":
            command += ["--text-artifact", str(output / "released_text/stage3-active.txt")]
        print(f"Starting {phase}: {checked.stdout.strip()}", flush=True)
        started = time.monotonic()
        with (output / f"{phase}.log").open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        record = {"phase": phase, "command": command, "exit_code": process.returncode, "seconds": time.monotonic() - started,
                  "filesystem_firewall": json.loads(checked.stdout), "network_disabled": True,
                  "runtime_manifest_sha256": observed["manifest_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                  "source_state": "local uncommitted, hash-frozen DEV code"}
        records.append(record)
        (output / "executions.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "command"}), flush=True)
        if process.returncode:
            print((output / f"{phase}.log").read_text()[-4000:], flush=True)
            raise SystemExit(process.returncode)
        if phase == "reference":
            release = output / "released_text"
            release.mkdir()
            shutil.copyfile(output / "reference/stage3-active.txt", release / "stage3-active.txt")
        if phase == "train":
            shutil.copytree(output / "train/states", output / "released_states")


if __name__ == "__main__":
    main()
