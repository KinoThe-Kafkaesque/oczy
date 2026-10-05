"""Compare one complete historical DEV seed cell; never write a calibration shard.

Run in the verified namespace from reproduce_r20_dev.py. The original six
condition collector is called directly. No-update repeat sampling is outside
this diagnostic's scope; its absence is explicit in the result artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["OCZY_DIAGNOSTIC_SOURCE"])

import torch  # noqa: E402

from oczy.experiments.meta_cortex.artifacts import (  # noqa: E402
    canonical_theta_hash,
    load_developmental_checkpoint,
)
from oczy.experiments.meta_cortex.calibration import (  # noqa: E402
    FrozenScorer,
    load_calibration_shard,
)
from oczy.experiments.meta_cortex.calibration_runner import (  # noqa: E402
    _build_donor_task_map,
    _collect_seed_cell_for_task,
)
from oczy.experiments.meta_cortex.contracts import ModelConfig  # noqa: E402
from oczy.experiments.meta_cortex.instrument import load_calibration_view  # noqa: E402
from oczy.experiments.meta_cortex.model import MetaCortex  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402
from oczy.experiments.meta_cortex.training import (  # noqa: E402
    OptimizationBoundary,
    unroll_online_episode,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-root", "checkpoint", "reference-shard", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--dev-seed-index", type=int, required=True)
    parser.add_argument("--eval-seed-index", type=int, required=True)
    args = parser.parse_args()
    view = load_calibration_view(args.public_root)
    reference = load_calibration_shard(args.reference_shard, view)
    task = view.tasks[args.task_index]
    expected = [r for r in reference.seed_cell_records
                if (r.developmental_seed_index, r.evaluation_seed_index, r.rule_fingerprint)
                == (args.dev_seed_index, args.eval_seed_index, task.rule_fingerprint)]
    if len(expected) != 1:
        raise ValueError("Requires exactly one matching historical reference cell")
    config = json.loads((args.checkpoint / "checkpoint.json").read_text())["model_config"]
    torch.set_num_threads(4)
    model = MetaCortex(ModelConfig(**config)).eval()
    metadata = load_developmental_checkpoint(args.checkpoint, model)
    organ = QwenFrozenOrgan.load(feature_dim=config["feature_dim"])
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        before = organ.parameter_hash()
        if before != metadata.organ_hash or before != reference.organ_hash:
            raise ValueError("Historical organ hash mismatch")
        if organ.organ_identity != metadata.organ_identity:
            raise ValueError("Historical organ identity mismatch")
        if canonical_theta_hash(model) != metadata.theta_hash:
            raise ValueError("Checkpoint theta mismatch")
        scorer = FrozenScorer()
        if scorer.sha256 != view.scorer_sha256:
            raise ValueError("Frozen scorer mismatch")
        boundary = OptimizationBoundary()
        donor = _build_donor_task_map(view.tasks)[args.task_index]
        print("All identities verified; collecting complete six-condition cell", flush=True)
        with torch.inference_mode():
            donor_state = unroll_online_episode(
                model, organ, donor, boundary, update_enabled=True, gradient_enabled=False,
            ).state
        record = _collect_seed_cell_for_task(
            model, organ, task, boundary, scorer,
            dev_seed_index=args.dev_seed_index, eval_seed_index=args.eval_seed_index,
            dev_seed=view.developmental_seeds[args.dev_seed_index],
            eval_seed=view.evaluation_seeds[args.eval_seed_index],
            theta_hash=metadata.theta_hash, organ_hash=before, donor_state=donor_state,
            device=next(model.parameters()).device, dtype=next(model.parameters()).dtype,
        ).to_json_obj()
        after = organ.parameter_hash()
        theta_after = canonical_theta_hash(model)
        original = expected[0].to_json_obj()
        report = {
            "schema_version": "oczy/r20-cell-reproduction-diagnostic/v1",
            "classification": "DEV_DIAGNOSTIC_ONLY", "meta_test_accessed": False,
            "no_update_repeats_reproduced": False, "calibration_shard_written": False,
            "organ_hash_before": before, "organ_hash_after": after,
            "theta_hash_before": metadata.theta_hash, "theta_hash_after": theta_after,
            "all_cell_fields_equal": record == original,
            "optimizer_steps": boundary.optimizer_step_count,
            "record": record,
        }
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "record"}, indent=2))
        return 0 if record == original and before == after and metadata.theta_hash == theta_after and boundary.optimizer_step_count == 0 else 1
    finally:
        organ.close()


if __name__ == "__main__":
    raise SystemExit(main())
