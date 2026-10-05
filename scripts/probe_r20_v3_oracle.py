"""R20 gate G2 — oracle capability screen on the frozen meta_cortex/v3 instrument.

Question G2 asks: **can the frozen organ express a v3 task under oracle
control?** The soft bank is empty and no cortex state exists, so any correct
answer is produced by the frozen language organ alone through the frozen
protocol. This is a mouth–cortex protocol test, not a test of the learner.

Conditions, all on the public DEV tuning split only:

- ``no_context``     — the probe alone. Expects ~0; it bounds the floor.
- ``teaching_context`` — probe plus the task's full teaching transcript. This is
  the **retrieval comparator** and the bar any changed-dynamics result must
  clear (AGENTS.md rule 5). Not a primary result.
- ``oracle_context`` — probe plus the oracle description. This is the gate.

DEV only. No optimizer step, no training, no sealed or meta-test access, no
threshold selection, no calibration. The organ hash must equal the hash bound by
the frozen instrument, before and after. Exit 0 if the screen completes; the
gate verdict is reported in the JSON, not in the exit code, because a *failed*
G2 is a result that must be recorded rather than an execution error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from oczy.experiments.meta_cortex.calibration import FrozenScorer  # noqa: E402
from oczy.experiments.meta_cortex.contracts import (  # noqa: E402
    DialogueMessage,
    ProbeKind,
    TaskFamily,
)
from oczy.experiments.meta_cortex.instrument import load_dev_view  # noqa: E402
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402


def with_teaching_context(messages, transcript):
    """Prepend the full teaching transcript, keeping any system message first."""
    if messages and messages[0].role == "system":
        return messages[:1] + tuple(transcript) + messages[1:]
    return tuple(transcript) + messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    view = load_dev_view(args.public_root)
    catalog = view.catalog
    scorer = FrozenScorer()
    if scorer.sha256 != view.binding.scorer_registry_sha256:
        raise ValueError("Scorer differs from the frozen instrument binding")

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    organ = QwenFrozenOrgan.load()
    before = organ.parameter_hash()
    if before != view.binding.organ_hash:
        raise ValueError(
            f"Organ identity {before} differs from the frozen binding "
            f"{view.binding.organ_hash}"
        )

    # Empty soft bank: there is no cortex state, so the oracle screen measures
    # the frozen organ's articulation alone.
    bank = torch.zeros((1, 0, organ.feature_dim))

    started = time.time()
    rows: list[dict] = []
    try:
        for family in TaskFamily:
            tasks = [t for t in catalog.meta_validation if t.family == family]
            for task_index, task in enumerate(tasks):
                transcript = []
                for event in task.events:
                    transcript.extend(event.observation_messages)
                    transcript.append(DialogueMessage("assistant", event.attempted_behavior))
                    transcript.append(DialogueMessage("user", event.correction))
                for condition, kind in (
                    ("no_context", ProbeKind.SAME_RULE),
                    ("teaching_context", ProbeKind.SAME_RULE),
                    ("oracle_context", ProbeKind.ORACLE_CONTEXT),
                ):
                    for probe_index, probe in enumerate(task.probes.by_kind(kind)):
                        messages = (
                            with_teaching_context(probe.messages, transcript)
                            if condition == "teaching_context"
                            else probe.messages
                        )
                        generated = organ.generate(messages, bank, max_new_tokens=32)
                        rows.append(
                            {
                                "family": family.value,
                                "task_index": task_index,
                                "rule_fingerprint": task.rule_fingerprint,
                                "condition": condition,
                                "probe_kind": kind.value,
                                "probe_index": probe_index,
                                "prompt_sha256": hashlib.sha256(
                                    json.dumps(
                                        [
                                            {"role": m.role, "content": m.content}
                                            for m in messages
                                        ],
                                        sort_keys=True,
                                    ).encode()
                                ).hexdigest(),
                                "expected": probe.expected_response,
                                "generated": generated,
                                "correct": scorer.score_response(
                                    probe.expected_response, generated
                                ),
                            }
                        )
                    print(
                        f"Completed {family.value}/{condition} task {task_index}",
                        flush=True,
                    )
        organ.assert_frozen()
        after = organ.parameter_hash()
        if before != after:
            raise ValueError("Frozen organ changed during the screen")
    finally:
        organ.close()

    scores = {}
    for condition in ("no_context", "teaching_context", "oracle_context"):
        selected = [r for r in rows if r["condition"] == condition]
        scores[condition] = {
            "correct": sum(r["correct"] for r in selected),
            "total": len(selected),
        }
    oracle_by_family = {}
    for family in TaskFamily:
        selected = [
            r for r in rows if r["condition"] == "oracle_context" and r["family"] == family.value
        ]
        oracle_by_family[family.value] = {
            "correct": sum(r["correct"] for r in selected),
            "total": len(selected),
        }

    # G2 verdict. The gate is the oracle condition only; no_context and
    # teaching_context are reported as the floor and the retrieval bar, and
    # neither decides the gate.
    oracle = scores["oracle_context"]
    gate_passed = oracle["total"] > 0 and oracle["correct"] > 0

    result = {
        "schema_version": "oczy/r20-g2-oracle-screen/v1",
        "classification": "DEV_GATE_G2_ORACLE_SCREEN",
        "gate": "G2",
        "meta_test_accessed": False,
        "calibration_accessed": False,
        "sealed_accessed": False,
        "optimizer_steps": 0,
        "training_run": False,
        "thresholds_selected": False,
        "instrument_id": "meta_cortex/v3",
        "taskgen_schema": "oczy/meta-cortex/taskgen/v2-dev",
        "dev_view_sha256": view.binding.dev_view_sha256,
        "definition_sha256": view.binding.definition_sha256,
        "scorer_sha256": scorer.sha256,
        "organ_hash_before": before,
        "organ_hash_after": after,
        "organ_hash_matches_frozen_binding": before == view.binding.organ_hash,
        "soft_bank_width": 0,
        "max_new_tokens": 32,
        "split_used": "public DEV tuning",
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scores": scores,
        "oracle_by_family": oracle_by_family,
        "gate_criterion": "oracle_context correct > 0",
        "gate_passed": gate_passed,
        "gate_meaning": (
            "A pass means the frozen organ can express at least one v3 task "
            "under oracle control, i.e. the mouth-cortex protocol is not "
            "refuted for these tasks. It is not evidence of cortex learning, "
            "and a failure is an articulation block, not a cortex refutation."
        ),
        "rows": rows,
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gate": "G2",
                "gate_passed": gate_passed,
                "scores": scores,
                "oracle_by_family": oracle_by_family,
                "organ_hash_matches_frozen_binding": before == view.binding.organ_hash,
                "seconds": result["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
