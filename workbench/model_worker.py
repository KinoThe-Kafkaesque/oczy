"""Exploratory inference only. No query is written into research datasets."""

import argparse
import contextlib
import json
import sys
from pathlib import Path


def main(model):
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch

        from infrastructure.kaggle.runtime_manifest import (
            observe_runtime_manifest,
            validate_runtime_manifest,
        )
        from oczy.experiments.meta_cortex.calibration import FrozenScorer
        from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan
        from scripts.diversity_dev.contract import manifest_at, messages, reference_prompt
        from scripts.serialization_pilot_worker import generate, load_numeric
        from workbench.catalog import DATA, ROOT, STUDY, read, verify_data
        from workbench.live import validate_trial
        verify_data()
        manifest = manifest_at(ROOT / STUDY)
        provenance = read(ROOT / "experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json")
        expected = validate_runtime_manifest(provenance["job_spec"]["runtime_manifest"])
        observed = observe_runtime_manifest(model_root=model, logical_model_id=expected["model"]["logical_model_id"],
                   resolved_model_convention=expected["model"]["resolved_model_convention"], generation_config=expected["greedy_generation"], quantization=expected["model"]["quantization"])
        if observed != expected:
            raise ValueError("Pinned runtime/model mismatch")
        torch.set_num_threads(4)
        organ, scorer = QwenFrozenOrgan.load(), FrozenScorer()
        before = organ.parameter_hash()
        assert before == manifest["organ_hash"] and scorer.sha256 == manifest["scorer_sha256"]
        states = read(DATA / "states/state_manifest.json")["states"]
        teaching = read(ROOT / STUDY / "capacity_training/data.json")
        table = read(ROOT / STUDY / "oracle/data.json")["stages"][0]["table"]
        banks = {}
        for state in states:
            control = next(c for c in manifest["controls"] if c["seed"] == state["seed"])
            banks[state["seed"]] = {"diversity": load_numeric(DATA / "states", state["trained"], manifest),
                                   "control": load_numeric(ROOT / STUDY / "controls", control["trained"], manifest),
                                   "initial": load_numeric(DATA / "states", state["initial"], manifest)}
        empty = torch.zeros((1, 0, organ.feature_dim))
        for line in sys.stdin:
            try:
                value = validate_trial(json.loads(line))
                suffix = {"amber": "vek", "cobalt": "mip"}.get(value["client"], "")
                row = {"clients": [value["client"]], "input": value["input"], "target": value["input"] + suffix}
                rows = []
                for condition in ("diversity", "control", "initial", "no_context", "chat_diversity", "text_diversity", "direct_oracle"):
                    bank = banks[value["seed"]].get(condition, empty)
                    prompt = messages(row) if condition in banks[value["seed"]] else reference_prompt(row, condition, teaching, table)
                    output = generate(organ, prompt, bank, 32)
                    rows.append({"condition": condition, "generated": output, "correct": scorer.score_response(row["target"], output)})
                organ.assert_frozen()
                result = {"mode": "exploratory", **value, "target": row["target"], "rows": rows, "organ_hash": before,
                          "manifest_sha256": manifest["manifest_sha256"], "optimizer_steps": 0}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"error": str(exc)}
            protocol.write(json.dumps(result) + "\n")
            protocol.flush()
        assert organ.parameter_hash() == before
        organ.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    main(parser.parse_args().model)
