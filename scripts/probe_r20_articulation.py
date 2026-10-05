"""DEV-only output-path diagnostic on existing frozen R20 tuning tasks.

Uses the first tuning task in each family, every same-rule probe, the complete
teaching transcript, and existing oracle-context probes. It does not optimize,
change the scorer, construct new episodes, or access calibration/meta-test.
Raw outputs and special-token handling are retained for adapter diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get(
    "OCZY_DIAGNOSTIC_SOURCE", str(Path(__file__).resolve().parents[1] / "src")
)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from oczy.experiments.meta_cortex.contracts import DialogueMessage, TaskFamily  # noqa: E402
from oczy.experiments.meta_cortex.instrument import load_dev_view  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402


def with_teaching_context(messages, transcript):
    """Keep an approved system instruction first in the full conversation."""
    if messages and messages[0].role == "system":
        return messages[:1] + tuple(transcript) + messages[1:]
    return tuple(transcript) + messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-instrument", type=Path)
    args = parser.parse_args()
    view = load_dev_view(args.public_root)
    catalog = view.catalog
    amendment = None
    if args.repair_instrument is not None:
        from materialize_dev_prompt_repair import load_repair
        catalog, amendment = load_repair(args.public_root, args.repair_instrument)
    scorer = FrozenScorer()
    if scorer.sha256 != view.binding.scorer_registry_sha256:
        raise ValueError("Scorer differs from frozen DEV binding")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    organ = QwenFrozenOrgan.load()
    before = organ.parameter_hash()
    if before != view.binding.organ_hash:
        raise ValueError("Organ differs from frozen DEV binding")
    bank = torch.zeros((1, 0, organ.feature_dim))
    rows = []
    try:
        for family in TaskFamily:
            task = next(t for t in catalog.meta_validation if t.family == family)
            transcript = []
            for event in task.events:
                transcript.extend(event.observation_messages)
                transcript.append(DialogueMessage("assistant", event.attempted_behavior))
                transcript.append(DialogueMessage("user", event.correction))
            for condition in ("no_context", "teaching_context", "oracle_context"):
                probes = task.probes.oracle_context if condition == "oracle_context" else task.probes.same_rule
                for index, probe in enumerate(probes):
                    messages = with_teaching_context(probe.messages, transcript) if condition == "teaching_context" else probe.messages
                    generated = organ.generate(messages, bank, max_new_tokens=32)
                    # Diagnostic counterfactual only: the official score below
                    # always receives the unmodified adapter return value.
                    tokens = organ._tokenizer.encode(generated, add_special_tokens=False)
                    content_only = organ._tokenizer.decode(tokens, skip_special_tokens=True)
                    row = {
                        "family": family.value, "rule_fingerprint": task.rule_fingerprint,
                        "condition": condition, "probe_index": index,
                        "prompt_sha256": hashlib.sha256(json.dumps([
                            {"role": m.role, "content": m.content} for m in messages
                        ], sort_keys=True).encode()).hexdigest(),
                        "expected": probe.expected_response, "generated": generated,
                        "correct": scorer.score_response(probe.expected_response, generated),
                        "contains_special_tokens": content_only != generated,
                        "content_only_counterfactual": content_only,
                        "content_only_counterfactual_correct": scorer.score_response(probe.expected_response, content_only),
                    }
                    rows.append(row)
                print(f"Completed {family.value}/{condition}", flush=True)
        organ.assert_frozen()
        after = organ.parameter_hash()
        if before != after:
            raise ValueError("Frozen organ changed")
        result = {
            "schema_version": "oczy/r20-articulation-diagnostic/v1",
            "classification": "DEV_DIAGNOSTIC_ONLY", "meta_test_accessed": False,
            "source_root": str(SOURCE_ROOT),
            "organ_source_sha256": hashlib.sha256((SOURCE_ROOT / "oczy/experiments/meta_cortex/organ.py").read_bytes()).hexdigest(),
            "calibration_accessed": False, "optimizer_steps": 0,
            "dev_view_sha256": view.binding.dev_view_sha256, "scorer_sha256": scorer.sha256,
            "instrument_id": amendment["instrument_id"] if amendment else "meta_cortex/v2",
            "amendment_manifest_sha256": amendment["manifest_sha256"] if amendment else None,
            "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "organ_hash_before": before, "organ_hash_after": after, "rows": rows,
        }
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        for condition in ("no_context", "teaching_context", "oracle_context"):
            selected = [r for r in rows if r["condition"] == condition]
            print(json.dumps({"condition": condition, "correct": sum(r["correct"] for r in selected),
                              "total": len(selected), "special_token_counterfactual_correct": sum(
                                  r["content_only_counterfactual_correct"] for r in selected)}))
    finally:
        organ.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
