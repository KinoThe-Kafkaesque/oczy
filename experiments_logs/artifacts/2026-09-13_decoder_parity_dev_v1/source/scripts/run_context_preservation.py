"""Run frozen offline DEV-v2 phases and enforce the predeclared capacity gate."""

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
from scripts.context_preservation_contract import (  # noqa: E402
    capacity_gate,
    manifest_at,
    write_json,
)
from scripts.reproduce_r20_dev import namespace_prefix  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "model", "provenance"):
        parser.add_argument("--" + name, type=Path, required=True)
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
        raise ValueError("Pinned runtime/model mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_manifest.json", observed)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
                "OCZY_MODEL_DIR": provenance["model_root"], "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo / "src")})
    records = []
    for phase in ("reference", "train_capacity", "restore_capacity", "train_correction", "restore_correction"):
        if phase == "train_correction":
            decision = json.loads((output / "admission.json").read_text())
            if not decision["admitted"]:
                write_json(output / "blocked.json", {"phase": phase, "reason": "Joint scoped representation failed the frozen fresh-probe/teaching-fit prerequisite",
                                                     "manifest_sha256": manifest["manifest_sha256"], "correction_optimizer_steps": 0})
                print("Correction comparison blocked by the predeclared capacity gate.", flush=True)
                break
        prefix = namespace_prefix(args.model.resolve(), provenance["model_root"], output)
        hidden = [root / "audit", repo / "scripts/tests"]
        allowed_roles = {"reference": {"probes", "oracle", "capacity_training"}, "train_capacity": {"capacity_training"},
                         "restore_capacity": {"probes"}, "train_correction": {"correction_training"}, "restore_correction": {"probes"}}[phase]
        hidden += [root / role for role in ("probes", "oracle", "capacity_training", "correction_training") if role not in allowed_roles]
        for prior in records:
            hidden.append(output / prior["phase"])
        if phase == "restore_correction":
            hidden.append(output / "released_capacity")
        forbidden = []
        for directory in hidden:
            if directory.exists():
                forbidden.extend(str(p) for p in directory.rglob("*") if p.is_file())
            prefix.extend(["--tmpfs", str(directory)])
        for file in (repo / "scripts/prepare_context_preservation.py", root / "README.md"):
            if file.exists():
                prefix.extend(["--ro-bind", "/dev/null", str(file)])
        prefix.extend(["--tmpfs", str(repo / "scripts/__pycache__"), "--chdir", str(repo), "--"])
        check = ("from pathlib import Path; import json; paths=" + repr(forbidden) +
                 "; visible=[p for p in paths if Path(p).is_file()]; print(json.dumps({'forbidden_files_checked':len(paths),'visible_forbidden_files':visible})); assert not visible")
        checked = subprocess.run(prefix + [sys.executable, "-c", check], env=env, capture_output=True, text=True, check=True)
        command = prefix + [sys.executable, str(repo / "scripts/context_preservation_worker.py"), phase, "--root", str(root), "--output", str(output / phase)]
        if phase in ("restore_capacity", "train_correction"):
            command += ["--artifacts", str(output / "released_capacity")]
        if phase == "train_correction":
            command += ["--admission", str(output / "admission.json")]
        if phase == "restore_correction":
            command += ["--artifacts", str(output / "released_correction")]
        print(f"Starting {phase}: {checked.stdout.strip()}", flush=True)
        started = time.monotonic()
        with (output / f"{phase}.log").open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        record = {"phase": phase, "command": command, "exit_code": process.returncode, "seconds": time.monotonic() - started,
                  "filesystem_firewall": json.loads(checked.stdout), "network_disabled": True,
                  "runtime_manifest_sha256": observed["manifest_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                  "source_state": "local uncommitted hash-frozen DEV code"}
        records.append(record)
        (output / "executions.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "command"}), flush=True)
        if process.returncode:
            print((output / f"{phase}.log").read_text()[-4000:], flush=True)
            raise SystemExit(process.returncode)
        if phase in ("train_capacity", "train_correction"):
            shutil.copytree(output / phase / "states", output / ("released_capacity" if phase == "train_capacity" else "released_correction"))
        if phase == "restore_capacity":
            result = json.loads((output / phase / "results.json").read_text())
            fits = json.loads((output / "train_capacity/results.json").read_text())["fits"]
            fit_pass = len(fits) == len(manifest["training"]["seeds"]) and all(len(f["training_fit"]) == 9 and all(r["correct"] for r in f["training_fit"]) for f in fits)
            passed = fit_pass and capacity_gate(result["rows"], manifest["training"]["seeds"])
            write_json(output / "admission.json", {"manifest_sha256": manifest["manifest_sha256"], "admitted": passed})
            print(f"Capacity gate admitted={passed}", flush=True)


if __name__ == "__main__":
    main()
