"""Qualify a versioned decoder candidate without editing historical organ code."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.context_preservation_contract import (  # noqa: E402
    canonical,
    manifest_at,
    messages,
    sha,
    write_json,
)


def prepare(root, capacity_run):
    repo = Path(__file__).resolve().parents[1]
    parent = manifest_at(repo / "experiments/language-interface-dev-v3")
    cases = json.loads((repo / "experiments/language-interface-dev-v3/cases.json").read_text())
    cases = [r for r in cases if r["split"] == "calibration" and r["condition"] == "resolved_v2"]
    assert len(cases) == 16
    cases = [{**r, "bank": "none"} for r in cases] + [{**r, "bank": "joint", "messages": messages(r)} for r in cases]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(capacity_run / "train_capacity/states/seed0-joint.npy", root / "joint.npy")
    write_json(root / "cases.json", cases)
    sources = {**parent["execution_sources"]}
    for path in ("scripts/probe_decoder_parity.py", "src/oczy/experiments/meta_cortex/generation_v2.py"):
        sources[path] = sha((repo / path).read_bytes())
    manifest = {"instrument_id": "oczy/decoder-parity/dev-v1", "parent_manifest_sha256": parent["manifest_sha256"],
                "authorization": "2026-09-13 user request to proceed on language-interface work; separate decoder diagnostic version",
                "execution_sources": sources, "cases_sha256": sha((root / "cases.json").read_bytes()),
                "bank_sha256": sha((root / "joint.npy").read_bytes()), "organ_hash": parent["organ_hash"],
                "runtime_manifest_sha256": parent["runtime_manifest_sha256"], "scorer_sha256": parent["scorer_sha256"],
                "conditions": ["scalar", "old_default", "old_neutral", "dynamic_default", "dynamic_neutral"],
                "native_control": "Native input IDs with neutral generation config on all 16 zero-width bank cases",
                "batch_control": "Dynamic-neutral batches of four variable-length prompts, all 32 cases",
                "criterion": "All candidate scalar-sized and four-item-batch strings must match raw-argmax scalar strings exactly",
                "interpretation": "Decode parity only, not a change to old scores or evidence of learned capability",
                "optimizer_steps": 0, "max_new_tokens": 32}
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(root / "MANIFEST.json", manifest)
    print(manifest["manifest_sha256"])


def verify(root):
    result = manifest_at(root)
    assert sha((root / "cases.json").read_bytes()) == result["cases_sha256"]
    assert sha((root / "joint.npy").read_bytes()) == result["bank_sha256"]
    return result


def worker(root, output):
    import numpy as np
    import torch
    from transformers import GenerationConfig

    from oczy.experiments.meta_cortex.calibration import FrozenScorer
    from oczy.experiments.meta_cortex.contracts import DialogueMessage
    from oczy.experiments.meta_cortex.generation_v2 import generate_batch
    from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan, render_chat
    manifest = verify(root)
    torch.set_num_threads(2)
    organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
    before = organ.parameter_hash()
    assert before == manifest["organ_hash"] and scorer.sha256 == manifest["scorer_sha256"]
    array = np.load(root / "joint.npy", allow_pickle=False)
    assert array.shape == (1, 8, 896) and array.dtype == np.float32 and np.isfinite(array).all()
    banks = {"none": torch.zeros((1, 0, 896)), "joint": torch.from_numpy(array.copy())}
    cases = json.loads((root / "cases.json").read_text())
    default = organ._model.generation_config
    neutral = copy.deepcopy(default)
    neutral.repetition_penalty = 1.0
    rows = []
    try:
        for case in cases:
            prompt = tuple(DialogueMessage(**m) for m in case["messages"])
            bank = banks[case["bank"]]
            outputs = {"scalar": organ.generate(prompt, bank, max_new_tokens=32),
                       "old_default": organ.generate_batch([prompt], bank, max_new_tokens=32)[0]}
            try:
                organ._model.generation_config = neutral
                outputs["old_neutral"] = organ.generate_batch([prompt], bank, max_new_tokens=32)[0]
            finally:
                organ._model.generation_config = default
            outputs["dynamic_default"] = generate_batch(organ, [prompt], bank, repetition_penalty=1.1)[0]
            outputs["dynamic_neutral"] = generate_batch(organ, [prompt], bank)[0]
            if case["bank"] == "none":
                ids = torch.tensor([organ._tokenizer.encode(render_chat(prompt, organ._tokenizer), add_special_tokens=True)])
                config = GenerationConfig(max_new_tokens=32, do_sample=False, use_cache=True, repetition_penalty=1.0,
                                          pad_token_id=organ._tokenizer.pad_token_id, eos_token_id=organ._tokenizer.eos_token_id)
                with torch.inference_mode():
                    generated = organ._model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
                outputs["native_neutral"] = organ._tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            row = {**case, "outputs": outputs, "correct": {k: scorer.score_response(case["target"], v) for k, v in outputs.items()}}
            rows.append(row)
            with (output / "rows.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps({"bank": case["bank"], "input": case["input"], "client": case["category"],
                              "matches": {k: v == outputs["scalar"] for k, v in outputs.items()}}), flush=True)
        batches = []
        for start in range(0, len(cases), 4):
            group = cases[start:start + 4]
            prompts = [tuple(DialogueMessage(**m) for m in row["messages"]) for row in group]
            outputs = generate_batch(organ, prompts, banks[group[0]["bank"]].expand(len(group), -1, -1))
            batches += [{"index": start + i, "generated": value} for i, value in enumerate(outputs)]
        after = organ.parameter_hash()
        organ.assert_frozen()
        assert before == after
        write_json(output / "results.json", {"manifest_sha256": manifest["manifest_sha256"], "rows": rows, "batches": batches,
                                            "default_generation_config": default.to_dict(), "organ_hash_before": before, "organ_hash_after": after,
                                            "scorer_sha256": scorer.sha256, "optimizer_steps": 0, "meta_test_accessed": False})
    finally:
        organ._model.generation_config = default
        organ.close()


def launch(root, output, model, provenance):
    from infrastructure.kaggle.runtime_manifest import (
        observe_runtime_manifest,
        validate_runtime_manifest,
    )
    from scripts.reproduce_r20_dev import namespace_prefix
    manifest = verify(root)
    source = json.loads(provenance.read_text())
    expected = validate_runtime_manifest(source["job_spec"]["runtime_manifest"])
    observed = observe_runtime_manifest(model_root=model, logical_model_id=expected["model"]["logical_model_id"],
                                        resolved_model_convention=expected["model"]["resolved_model_convention"],
                                        generation_config=expected["greedy_generation"], quantization=expected["model"]["quantization"])
    assert observed == expected and observed["manifest_sha256"] == manifest["runtime_manifest_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_manifest.json", observed)
    repo = Path(__file__).resolve().parents[1]
    command = namespace_prefix(model, source["model_root"], output) + ["--chdir", str(repo), "--", sys.executable,
                str(Path(__file__).resolve()), "worker", "--root", str(root), "--output", str(output)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
           "OCZY_MODEL_DIR": source["model_root"], "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    started = time.monotonic()
    with (output / "worker.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    write_json(output / "execution.json", {"command": command, "exit_code": result.returncode, "seconds": time.monotonic() - started, "network_disabled": True})
    if result.returncode:
        print((output / "worker.log").read_text()[-4000:])
    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    for name in ("capacity-run", "output", "model", "provenance"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.capacity_run.resolve())
    elif args.mode == "worker":
        worker(args.root.resolve(), args.output.resolve())
    else:
        launch(args.root.resolve(), args.output.resolve(), args.model.resolve(), args.provenance.resolve())


if __name__ == "__main__":
    main()
