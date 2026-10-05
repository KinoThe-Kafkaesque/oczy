"""Launch the registered read-only regression replay offline."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from infrastructure.kaggle.runtime_manifest import (
    observe_runtime_manifest,
    validate_runtime_manifest,
)
from scripts.regression_dev.contract import CAP, REPO, verify, write_json
from scripts.reproduce_r20_dev import namespace_prefix


def run(root, output, candidate, model):
    manifest = verify(root)
    source = json.loads((REPO / CAP / "selected_runtime_provenance.json").read_text())
    expected = validate_runtime_manifest(source["job_spec"]["runtime_manifest"])
    observed = observe_runtime_manifest(model_root=model, logical_model_id=expected["model"]["logical_model_id"],
               resolved_model_convention=expected["model"]["resolved_model_convention"], generation_config=expected["greedy_generation"],
               quantization=expected["model"]["quantization"])
    if observed != expected:
        raise ValueError("Runtime/model changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_manifest.json", observed)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
           "OCZY_MODEL_DIR": source["model_root"], "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO / "src"), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    command = namespace_prefix(model, source["model_root"], output) + ["--chdir", str(REPO), "--", sys.executable, "-m", "scripts.regression_dev.worker",
              "--root", str(root), "--output", str(output / "replay"), "--candidate", str(candidate)]
    started = time.monotonic()
    with (output / "worker.log").open("x") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    write_json(output / "execution.json", {"command": command, "exit_code": process.returncode, "seconds": time.monotonic() - started,
               "network_disabled": True, "optimizer_steps": 0, "manifest_sha256": manifest["manifest_sha256"]})
    if process.returncode:
        print((output / "worker.log").read_text()[-5000:])
        raise SystemExit(process.returncode)
    print((output / "replay/results.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "candidate", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(*(getattr(args, n).resolve() for n in ("root", "output", "candidate", "model")))
