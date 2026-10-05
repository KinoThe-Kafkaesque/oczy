"""Frozen DEV language-layout comparison and live greedy decoder parity audit."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.context_preservation_contract import (  # noqa: E402
    SYSTEM,
    canonical,
    messages,
    sha,
    write_json,
)


def build_cases():
    """Fixed prompts; targets are scored separately and never interpolated."""
    rules = ('Rules (write the input first, then the suffix, with no spaces):\n'
             'amber: input + "vek"\ncobalt: input + "mip"\nAny other client: input unchanged.')
    examples = [{"clients": [client], "input": word, "target": word + suffix}
                for client, suffix in (("amber", "vek"), ("cobalt", "mip"), ("silver", "")) for word in ("oak", "pine", "elm")]
    text = "Corrected examples:\n" + "\n".join(f"Client {r['clients'][0]}, input {r['input']} -> correct output {r['target']}" for r in examples)
    cases = []
    for split, words in (("calibration", ("lime", "melon", "grape", "peach")),
                         ("confirmation", ("apple", "mango", "lemon", "guava"))):
        for client, suffix in (("amber", "vek"), ("cobalt", "mip"), ("silver", ""), ("quartz", "")):
            for word in words:
                row = {"clients": [client], "input": word, "category": client, "target": word + suffix}
                operation = f'Write the input followed immediately by the literal text "{suffix}".' if suffix else "Copy the input exactly."
                resolved = messages(row, context=operation)
                query = messages(row)[-1]
                merged = [resolved[0], {"role": "user", "content": operation + "\n\n" + query["content"]}]
                direct = [{"role": "system", "content": "Return only the resulting string, with no explanation or quotation marks."},
                          {"role": "user", "content": f'Append the suffix "{suffix}" to the string "{word}".' if suffix else f'Copy the string "{word}" exactly.'}]
                chat = [{"role": "system", "content": SYSTEM}]
                for example in examples:
                    chat += [messages(example)[-1], {"role": "assistant", "content": example["target"]}]
                chat.append(query)
                conditions = {"resolved_v2": resolved, "merged_resolved": merged, "direct_operation": direct,
                              "text_examples_v2": messages(row, context=text), "chat_examples": chat,
                              "table_v2": messages(row, context=rules),
                              "table_in_system": [{"role": "system", "content": SYSTEM + "\n\n" + rules}, query]}
                for name, prompt in conditions.items():
                    cases.append({**row, "split": split, "condition": name, "messages": prompt})
    return cases


def prepare(root):
    from scripts.context_preservation_contract import manifest_at
    repo = Path(__file__).resolve().parents[1]
    parent = manifest_at(repo / "experiments/context-preservation-dev-v2")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "cases.json", build_cases())
    manifest = {"instrument_id": "oczy/language-interface/dev-v3", "parent_manifest_sha256": parent["manifest_sha256"],
                "authorization": "User request of 2026-09-13 to proceed on the language interface; separate DEV version, no historical changes",
                "organ_hash": parent["organ_hash"], "scorer_sha256": parent["scorer_sha256"],
                "runtime_manifest_sha256": parent["runtime_manifest_sha256"], "max_new_tokens": 32,
                "execution_sources": {**parent["execution_sources"], "scripts/probe_language_interface.py": sha(Path(__file__).read_bytes())},
                "cases_sha256": sha((root / "cases.json").read_bytes()), "optimizer_steps": 0,
                "selection": "All seven fixed conditions on all cases; no best-prompt selection or next adaptive variant in this run",
                "causality": ["resolved_v2 vs merged_resolved changes turn grouping", "text_examples_v2 vs chat_examples changes example representation",
                              "table_v2 vs table_in_system changes rule role", "direct_operation is oracle-scoped primitive diagnostic, not selective application"],
                "decoder_parity": "All 16 resolved_v2 calibration prompts: custom scalar vs native input_ids greedy vs batch API, same frozen model and prompts"}
    manifest["manifest_sha256"] = sha(canonical(manifest))
    write_json(root / "MANIFEST.json", manifest)
    print(manifest["manifest_sha256"])


def verify(root):
    from scripts.context_preservation_contract import manifest_at
    manifest = manifest_at(root)
    assert sha((root / "cases.json").read_bytes()) == manifest["cases_sha256"]
    return manifest


def worker(root, output):
    import torch

    from oczy.experiments.meta_cortex.calibration import FrozenScorer
    from oczy.experiments.meta_cortex.contracts import DialogueMessage
    from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan, render_chat
    manifest = verify(root)
    torch.set_num_threads(2)
    organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
    before = organ.parameter_hash()
    assert before == manifest["organ_hash"] and scorer.sha256 == manifest["scorer_sha256"]
    bank = torch.zeros((1, 0, organ.feature_dim))
    rows = []
    try:
        for case in json.loads((root / "cases.json").read_text()):
            prompt = tuple(DialogueMessage(**m) for m in case["messages"])
            generated = organ.generate(prompt, bank, max_new_tokens=32)
            row = {**case, "generated": generated, "correct": scorer.score_response(case["target"], generated)}
            if case["split"] == "calibration" and case["condition"] == "resolved_v2":
                token_ids = organ._tokenizer.encode(render_chat(prompt, organ._tokenizer), add_special_tokens=True)
                ids = torch.tensor([token_ids], dtype=torch.long, device=next(organ._model.parameters()).device)
                with torch.no_grad():
                    sequence = organ._model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                                                     do_sample=False, pad_token_id=organ._tokenizer.eos_token_id,
                                                     eos_token_id=organ._tokenizer.eos_token_id)
                native = organ._tokenizer.decode(sequence[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
                batch = organ.generate_batch([prompt], bank, max_new_tokens=32)[0]
                row["decoder_parity"] = {"native": native, "batch": batch, "all_text_equal": native == batch == generated}
            rows.append(row)
            with (output / "rows.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("split", "condition", "category", "input", "correct")}), flush=True)
        organ.assert_frozen()
        after = organ.parameter_hash()
        assert before == after
        write_json(output / "results.json", {"manifest_sha256": manifest["manifest_sha256"], "rows": rows,
                                            "organ_hash_before": before, "organ_hash_after": after, "scorer_sha256": scorer.sha256,
                                            "optimizer_steps": 0, "meta_test_accessed": False, "pid": os.getpid()})
    finally:
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
    prefix = namespace_prefix(model, source["model_root"], output)
    prefix += ["--tmpfs", str(repo / "scripts/tests"), "--tmpfs", str(repo / "scripts/__pycache__"), "--chdir", str(repo), "--"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OCZY_REMOTE_CPU_ONLY": "1",
           "OCZY_MODEL_DIR": source["model_root"], "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    command = prefix + [sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root), "--output", str(output)]
    started = time.monotonic()
    with (output / "worker.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    write_json(output / "execution.json", {"command": command, "exit_code": result.returncode, "seconds": time.monotonic() - started,
                                          "network_disabled": True, "source_state": "local uncommitted hash-frozen DEV code"})
    if result.returncode:
        print((output / "worker.log").read_text()[-4000:])
    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    for name in ("output", "model", "provenance"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(root)
    elif args.mode == "worker":
        worker(root, args.output.resolve())
    else:
        launch(root, args.output.resolve(), args.model.resolve(), args.provenance.resolve())


if __name__ == "__main__":
    main()
