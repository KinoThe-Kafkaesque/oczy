"""Verify the pinned Qwen language-organ artifact on a Kaggle CPU kernel.

This is a model/infrastructure probe, not a research experiment. It discovers
the attached Kaggle model without network access, hashes the artifact files,
loads the frozen model, and proves that a gradient can flow to input embeddings
without creating parameter gradients or changing a sampled parameter
fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "oczy/kaggle-qwen-model-probe/v1"
MODEL_SOURCE = "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
EXPECTED_MODEL_TYPE = "qwen2"
EXPECTED_HIDDEN_SIZE = 896


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash() -> str:
    return _sha256_file(Path(__file__))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_expected_model(path: Path) -> bool:
    try:
        config = _read_config(path / "config.json")
    except (OSError, json.JSONDecodeError):
        return False
    return (
        config.get("model_type") == EXPECTED_MODEL_TYPE
        and config.get("hidden_size") == EXPECTED_HIDDEN_SIZE
        and (path / "model.safetensors").is_file()
    )


def locate_model(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not _is_expected_model(candidate):
            raise RuntimeError(f"not the expected Qwen model directory: {candidate}")
        return candidate

    environment = os.environ.get("OCZY_MODEL_DIR")
    if environment:
        candidate = Path(environment).expanduser().resolve()
        if _is_expected_model(candidate):
            return candidate

    search_roots = [Path("/kaggle/input"), Path("/kaggle/models")]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for config_path in root.rglob("config.json"):
            candidate = config_path.parent.resolve()
            if _is_expected_model(candidate):
                candidates.append(candidate)

    unique = sorted(set(candidates), key=str)
    if len(unique) != 1:
        rendered = ", ".join(str(path) for path in unique) or "none"
        raise RuntimeError(f"expected exactly one attached Qwen model; found: {rendered}")
    return unique[0]


def artifact_manifest(model_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        manifest.append(
            {
                "path": str(path.relative_to(model_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return manifest


def _parameter_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        detached = parameter.detach()
        digest.update(name.encode())
        digest.update(str(detached.dtype).encode())
        digest.update(json.dumps(list(detached.shape)).encode())
        flat = detached.reshape(-1)
        if flat.numel() == 0:
            continue
        indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
        sample = flat[indices].float().cpu().contiguous().numpy().tobytes()
        digest.update(sample)
    return digest.hexdigest()


def _device_report(device: torch.device) -> dict[str, Any]:
    return {
        "selected": str(device),
        "cuda_available": False,
        "cuda_device_count": 0,
        "torch_cuda_version": torch.version.cuda,
        "name": platform.processor() or platform.machine(),
    }


def run_probe(model_dir: Path, *, metadata_only: bool) -> dict[str, Any]:
    config = _read_config(model_dir / "config.json")
    files = artifact_manifest(model_dir)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Model and gradient-path verification only; not meta_cortex/v1 and not a "
            "behavioral result."
        ),
        "model_source": MODEL_SOURCE,
        "runner_sha256": _source_hash(),
        "model_dir": str(model_dir),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_files": files,
        "artifact_total_bytes": sum(item["size_bytes"] for item in files),
        "config": {
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "vocab_size": config.get("vocab_size"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "metadata_only": metadata_only,
    }
    config_valid = (
        config.get("model_type") == EXPECTED_MODEL_TYPE
        and config.get("hidden_size") == EXPECTED_HIDDEN_SIZE
    )
    report["checks"] = {"config_valid": config_valid, "artifact_files_present": bool(files)}
    if metadata_only:
        report["passed"] = config_valid and bool(files)
        return report

    device = torch.device("cpu")
    report["device"] = _device_report(device)

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers import (
        __version__ as transformers_version,
    )

    dtype = torch.float32
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    load_seconds = time.perf_counter() - load_started

    fingerprint_before = _parameter_fingerprint(model)
    prompt = "Infrastructure probe: return the word ready."
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    forward_started = time.perf_counter()
    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    scalar = output.logits[:, -1, :128].float().square().mean()
    scalar.backward()
    forward_backward_seconds = time.perf_counter() - forward_started

    input_gradient = inputs_embeds.grad
    gradient_finite = input_gradient is not None and bool(torch.isfinite(input_gradient).all())
    gradient_norm = (
        float(input_gradient.float().norm().item()) if input_gradient is not None else 0.0
    )
    parameter_gradients_absent = all(parameter.grad is None for parameter in model.parameters())
    fingerprint_after = _parameter_fingerprint(model)
    fingerprint_unchanged = fingerprint_before == fingerprint_after

    report["runtime"]["transformers"] = transformers_version
    report["model"] = {
        "dtype": str(dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "load_seconds": load_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "input_tokens": int(input_ids.shape[1]),
        "logits_shape": list(output.logits.shape),
    }
    report["checks"].update(
        {
            "input_gradient_present": input_gradient is not None,
            "input_gradient_finite": gradient_finite,
            "input_gradient_norm": gradient_norm,
            "parameter_gradients_absent": parameter_gradients_absent,
            "parameter_fingerprint_before": fingerprint_before,
            "parameter_fingerprint_after": fingerprint_after,
            "parameter_fingerprint_unchanged": fingerprint_unchanged,
        }
    )
    report["passed"] = (
        config_valid
        and bool(files)
        and gradient_finite
        and gradient_norm > 0.0
        and parameter_gradients_absent
        and fingerprint_unchanged
    )
    return report


def _default_output() -> Path:
    working = Path("/kaggle/working")
    return working / "qwen_model_probe.json" if working.is_dir() else Path("qwen_model_probe.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any]
    try:
        model_dir = locate_model(args.model_dir)
        report = run_probe(
            model_dir,
            metadata_only=args.metadata_only,
        )
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "model_source": MODEL_SOURCE,
            "runner_sha256": _source_hash(),
            "passed": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": {"type": type(error).__name__, "message": str(error)},
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
