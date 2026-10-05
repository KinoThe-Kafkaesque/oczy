"""Offline local execution with frozen inputs and a fail-closed correction gate."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from infrastructure.kaggle.runtime_manifest import (
    observe_runtime_manifest,
    validate_runtime_manifest,
)
from scripts.diversity_dev.contract import admission, manifest_at, write_json
from scripts.reproduce_r20_dev import namespace_prefix


def run(root, output, model, provenance):
    repo = Path(__file__).resolve().parents[2]
    manifest = manifest_at(root)
    source = json.loads(provenance.read_text())
    expected = validate_runtime_manifest(source["job_spec"]["runtime_manifest"])
    observed = observe_runtime_manifest(model_root=model, logical_model_id=expected["model"]["logical_model_id"],
                                        resolved_model_convention=expected["model"]["resolved_model_convention"],
                                        generation_config=expected["greedy_generation"], quantization=expected["model"]["quantization"])
    assert observed == expected and observed["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_manifest.json", observed)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
           "OCZY_MODEL_DIR": source["model_root"], "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo / "src"), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    records = []
    for phase in ("train", "evaluate", "train_correction", "restore_correction"):
        if phase == "train_correction" and not json.loads((output / "admission.json").read_text())["admitted"]:
            write_json(output / "blocked.json", {"phase": phase, "reason": "Acquisition prerequisite failed", "correction_optimizer_steps": 0,
                                                 "manifest_sha256": manifest["manifest_sha256"]})
            print("Correction blocked at frozen acquisition prerequisite", flush=True)
            break
        prefix = namespace_prefix(model, source["model_root"], output)
        hidden = [repo / "scripts/tests"]
        if phase == "train":
            hidden += [root / name for name in ("probes", "oracle", "controls", "correction_training")]
        if phase in ("train_correction", "restore_correction"):
            hidden += [root / "capacity_training", root / "oracle", root / "controls"]
            hidden.append(root / ("probes" if phase == "train_correction" else "correction_training"))
        hidden += [output / r["phase"] for r in records]
        forbidden = [str(p) for directory in hidden for p in directory.rglob("*") if p.is_file()]
        for path in hidden:
            prefix += ["--tmpfs", str(path)]
        for path in (repo / "scripts/diversity_dev/prepare.py", root / "README.md"):
            if path.exists():
                prefix += ["--ro-bind", "/dev/null", str(path)]
        prefix += ["--tmpfs", str(repo / "scripts/diversity_dev/__pycache__"), "--chdir", str(repo), "--"]
        check = "from pathlib import Path; import json; paths=" + repr(forbidden) + "; visible=[p for p in paths if Path(p).is_file()]; print(json.dumps({'checked':len(paths),'visible':visible})); assert not visible"
        audit = subprocess.run(prefix + [sys.executable, "-c", check], env=env, capture_output=True, text=True, check=True)
        if phase in ("train", "evaluate"):
            command = prefix + [sys.executable, "-m", f"scripts.diversity_dev.{phase}", "--root", str(root), "--output", str(output / phase)]
            if phase == "evaluate":
                command += ["--artifacts", str(output / "released_diversity")]
        else:
            command = prefix + [sys.executable, str(repo / "scripts/context_preservation_worker.py"), phase, "--root", str(root), "--output", str(output / phase),
                                "--artifacts", str(output / ("released_diversity" if phase == "train_correction" else "released_correction"))]
            if phase == "train_correction":
                command += ["--admission", str(output / "admission.json")]
        print(f"Starting {phase}", flush=True)
        started = time.monotonic()
        with (output / f"{phase}.log").open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        records.append({"phase": phase, "command": command, "exit_code": process.returncode, "seconds": time.monotonic() - started,
                        "filesystem_firewall": json.loads(audit.stdout), "network_disabled": True, "manifest_sha256": manifest["manifest_sha256"]})
        write_json(output / "executions.json", records)
        print(json.dumps({k: v for k, v in records[-1].items() if k != "command"}), flush=True)
        if process.returncode:
            print((output / f"{phase}.log").read_text()[-5000:], flush=True)
            raise SystemExit(process.returncode)
        if phase == "train":
            shutil.copytree(output / "train/states", output / "released_diversity")
        if phase == "train_correction":
            shutil.copytree(output / phase / "states", output / "released_correction")
        if phase == "evaluate":
            evaluated = json.loads((output / phase / "results.json").read_text())
            trained = json.loads((output / "train/results.json").read_text())
            probes = json.loads((root / "probes/data.json").read_text())["confirmation"][0]["rows"]
            passed = admission(evaluated["rows"], trained["fits"], manifest, probes)
            write_json(output / "admission.json", {"manifest_sha256": manifest["manifest_sha256"], "admitted": passed})
            print(f"Acquisition gate admitted={passed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "model", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.model.resolve(), args.provenance.resolve())
