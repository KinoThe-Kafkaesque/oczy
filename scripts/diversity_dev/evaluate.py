"""Fresh-process evaluation of both curricula and explicit retrieval controls."""

import argparse
import json
import zlib
from pathlib import Path

import torch

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan
from scripts.diversity_dev.contract import (
    data,
    example_text,
    manifest_at,
    messages,
    reference_prompt,
    sha,
    write_json,
)
from scripts.serialization_pilot_worker import generate, load_numeric


def run(root, output, artifacts):
    manifest = manifest_at(root)
    teaching, probes, oracle = [data(root, manifest, "evaluate", role) for role in ("capacity_training", "probes", "oracle")]
    meta = json.loads((artifacts / "state_manifest.json").read_text())
    assert meta["manifest_sha256"] == manifest["manifest_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
    before = organ.parameter_hash()
    assert before == manifest["organ_hash"] and scorer.sha256 == manifest["scorer_sha256"]
    groups = [("calibration", probes["calibration"]), ("confirmation", probes["confirmation"][0]["rows"])]
    table = oracle["stages"][0]["table"]
    rows, storage = [], []
    try:
        for arm, examples in (("control", teaching["groups"][0]), ("diversity", teaching["rows"])):
            text = example_text(examples)
            (output / f"{arm}.txt").write_text(text)
            compressed = zlib.compress(text.encode(), level=9)
            (output / f"{arm}.zlib").write_bytes(compressed)
            assert zlib.decompress(compressed).decode() == text
            storage.append({"arm": arm, "example_utf8_bytes": len(text.encode()), "zlib_bytes": len(compressed),
                            "text_sha256": sha(text.encode()), "zlib_sha256": sha(compressed), "roundtrip_exact": True})

        def record(row, split, condition, seed, bank, prompt):
            output_text = generate(organ, prompt, bank, 32)
            result = {**row, "split": split, "condition": condition, "seed": seed, "generated": output_text,
                      "correct": scorer.score_response(row["target"], output_text), "prompt_sha256": sha(json.dumps(prompt, sort_keys=True).encode())}
            rows.append(result)
            with (output / "rows.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")

        for state in meta["states"]:
            seed = state["seed"]
            parent = next(s for s in manifest["controls"] if s["seed"] == seed)
            controls = [("initial", load_numeric(root / "controls", parent["initial"], manifest)),
                        ("control", load_numeric(root / "controls", parent["trained"], manifest)),
                        ("diversity", load_numeric(artifacts, state["trained"], manifest))]
            if seed == 0:
                controls.append(("zeroed", torch.zeros_like(controls[0][1])))
            for condition, bank in controls:
                for split, cases in groups:
                    for case in cases:
                        record(case, split, condition, seed, bank, messages(case))
                print(f"numeric seed={seed} condition={condition}", flush=True)
        bank = torch.zeros((1, 0, organ.feature_dim))
        for condition in manifest["references"]:
            for split, cases in groups:
                for case in cases:
                    prompt = reference_prompt(case, condition, teaching, table)
                    if condition.startswith("zlib_"):
                        arm = condition.rsplit("_", 1)[-1]
                        restored = zlib.decompress((output / f"{arm}.zlib").read_bytes()).decode()
                        prompt = messages(case, context=restored)
                    record(case, split, condition, None, bank, prompt)
            print(f"reference condition={condition}", flush=True)
        after = organ.parameter_hash()
        organ.assert_frozen()
        assert before == after
        write_json(output / "results.json", {"manifest_sha256": manifest["manifest_sha256"], "rows": rows, "storage": storage,
                                            "organ_hash_before": before, "organ_hash_after": after, "scorer_sha256": scorer.sha256,
                                            "optimizer_steps": 0, "meta_test_accessed": False})
    finally:
        organ.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "artifacts"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.artifacts)
