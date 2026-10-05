"""Lazy offline inference subprocess; the local web server never loads the model."""

import json
import os
import re
import selectors
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from workbench.catalog import DATA, ROOT, verify_data

DEFAULT = Path.home() / ".local/state/oczy/diagnostics/2026-09-11-organ-identity"


def validate_trial(value):
    if not isinstance(value, dict) or set(value) != {"client", "input", "seed"}:
        raise ValueError("Choose a client, a word and a seed")
    if value["client"] not in ("amber", "cobalt", "silver", "quartz"):
        raise ValueError("Unknown client")
    if not isinstance(value["input"], str) or re.fullmatch(r"[a-z]{1,24}", value["input"]) is None:
        raise ValueError("Use one lowercase word, 1–24 letters")
    if type(value["seed"]) is not int or value["seed"] not in (0, 1, 2):
        raise ValueError("Choose seed 0, 1 or 2")
    return value


class LiveEngine:
    def __init__(self, runtime=None, model=None):
        self.runtime = Path(runtime or os.environ.get("OCZY_WORKBENCH_PYTHON", DEFAULT / "runtime/bin/python"))
        self.model = Path(model or os.environ.get("OCZY_WORKBENCH_MODEL", DEFAULT / "model"))
        self.process = None
        self.lock = threading.Lock()
        self.directory = None
        self.log = None

    def available(self):
        return bool(shutil.which("bwrap")) and self.runtime.is_file() and self.model.is_dir() and (DATA / "states/state_manifest.json").is_file()

    def _start(self):
        if not self.available():
            raise ValueError("Live model unavailable. Configure OCZY_WORKBENCH_PYTHON and OCZY_WORKBENCH_MODEL; saved evidence remains available.")
        verify_data()
        from scripts.reproduce_r20_dev import namespace_prefix
        source = json.loads((ROOT / "experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json").read_text())
        cache = Path.home() / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="oczy-workbench-", dir=cache)
        output = Path(self.directory.name)
        self.log = (output / "worker.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
               "OCZY_MODEL_DIR": source["model_root"], "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src"), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        command = namespace_prefix(self.model.resolve(), source["model_root"], output) + ["--ro-bind", str(ROOT), str(ROOT), "--chdir", str(ROOT), "--",
                  str(self.runtime), "-m", "workbench.model_worker", "--model", str(self.model.resolve())]
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)

    def query(self, value):
        value = validate_trial(value)
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another live trial is running. Wait for it to finish.")
        try:
            if self.process is None or self.process.poll() is not None:
                self.close()
                self._start()
            self.process.stdin.write(json.dumps(value) + "\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=180):
                    self.close()
                    raise ValueError("Live trial timed out. Retry to restart the worker.")
                line = self.process.stdout.readline()
            if not line:
                self.close()
                raise ValueError("Live worker stopped. Check runtime/model configuration and retry.")
            result = json.loads(line)
            if "error" in result:
                raise ValueError(result["error"])
            return result
        finally:
            self.lock.release()

    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self.log:
            self.log.close()
        if self.directory:
            self.directory.cleanup()
