#!/usr/bin/env python3
"""R20 gate G2 (successor) — oracle capability screen on ``meta_cortex/v4-r20``.

Question G2 asks: **can the frozen organ express a task under oracle control?**
The soft bank is empty and no cortex state exists, so any correct answer is
produced by the frozen language organ alone through the frozen protocol.  This
is a mouth-cortex protocol test, not a test of the learner.

This is the successor run of ``scripts/probe_r20_v3_oracle.py``.  It is
identical in structure and criterion; the differences are exactly the three
authorized successor changes:

1. the instrument is ``meta_cortex/v4-r20`` (the frozen v3 task content with the
   2026-09-11 Amendment A + B prompts carried in);
2. ``max_new_tokens`` is the instrument's authorized 128 rather than 32;
3. the instrument is verified fail-closed against its own pinned definition
   before a single model call.

Conditions, all on the public DEV tuning split only:

- ``no_context``     — the probe alone.  Expects ~0; it bounds the floor.
- ``teaching_context`` — probe plus the task's full teaching transcript.  This is
  the **retrieval comparator** and the bar any changed-dynamics result must
  clear (AGENTS.md rule 5).  Not a primary result.
- ``oracle_context`` — probe plus the oracle description.  This is the gate.

DEV only.  No optimizer step, no training, no sealed or meta-test access, no
threshold selection, no calibration.  The organ hash must equal the hash bound
by the frozen instrument, before and after.  Exit 0 if the screen completes; the
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
from oczy.experiments.meta_cortex.instrument_v4_r20 import (  # noqa: E402
    BARE_ANSWER,
    INSTRUMENT_ID,
    MAX_NEW_TOKENS,
    verify_v4_r20_definition,
)
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan  # noqa: E402

#: The organ control tokens that terminate transport.  Used only to report a
#: truncation accounting line; the scorer never sees this list.
_EOS_SPELLINGS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def with_teaching_context(messages, transcript):
    """Prepend the full teaching transcript, keeping any system message first."""
    if messages and messages[0].role == "system":
        return messages[:1] + tuple(transcript) + messages[1:]
    return tuple(transcript) + messages


def looks_truncated(text: str) -> bool:
    """Heuristic truncation signal: the generation ends mid-sentence.

    Reported as a descriptive accounting line only.  It decides nothing: the
    gate is the exact scorer on the oracle condition.

    A short bare token with no whitespace (``q2``, ``dax``) is a complete answer
    form, not a truncated sentence; the check only fires on prose that stops
    without terminal punctuation.
    """
    stripped = text.rstrip()
    if not stripped:
        return True
    if len(stripped) < 32 and not any(ch.isspace() for ch in stripped):
        return False
    return stripped[-1] not in ".!?\"')]}`"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    instrument_root = Path(args.instrument_root)
    definition = verify_v4_r20_definition(instrument_root)
    if definition.max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError(
            f"Instrument max_new_tokens {definition.max_new_tokens} is not {MAX_NEW_TOKENS}"
        )

    public_root = instrument_root / "public"
    view = load_dev_view(public_root)
    catalog = view.catalog
    scorer = FrozenScorer()
    if scorer.sha256 != view.binding.scorer_registry_sha256:
        raise ValueError("Scorer differs from the frozen instrument binding")

    # The instrument must actually carry Amendment A on every probe.
    for task in (*catalog.meta_train, *catalog.meta_validation):
        for kind in ProbeKind:
            for probe in task.probes.by_kind(kind):
                if not probe.messages or probe.messages[0].role != "system":
                    raise ValueError("Instrument probe is missing the Amendment A system message")
                if probe.messages[0].content != BARE_ANSWER:
                    raise ValueError("Instrument probe system message is not Amendment A")

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    organ = QwenFrozenOrgan.load()
    before = organ.parameter_hash()
    if before != view.binding.organ_hash:
        raise ValueError(
            f"Organ identity {before} differs from the frozen binding {view.binding.organ_hash}"
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
                        generated = organ.generate(messages, bank, max_new_tokens=MAX_NEW_TOKENS)
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
                                "truncation_heuristic": looks_truncated(generated),
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

    # Descriptive accounting only; it decides nothing.
    truncation_accounting = {}
    for condition in ("no_context", "teaching_context", "oracle_context"):
        selected = [r for r in rows if r["condition"] == condition]
        truncation_accounting[condition] = {
            "truncation_heuristic_true": sum(r["truncation_heuristic"] for r in selected),
            "total": len(selected),
        }

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
        "instrument_id": INSTRUMENT_ID,
        "taskgen_schema": "oczy/meta-cortex/taskgen/v2-dev",
        "base_instrument_id": "meta_cortex/v3",
        "base_definition_sha256": "ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee",
        "dev_view_sha256": view.binding.dev_view_sha256,
        "definition_sha256": view.binding.definition_sha256,
        "scorer_sha256": scorer.sha256,
        "organ_hash_before": before,
        "organ_hash_after": after,
        "organ_hash_matches_frozen_binding": before == view.binding.organ_hash,
        "soft_bank_width": 0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "base_max_new_tokens": 32,
        "split_used": "public DEV tuning",
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scores": scores,
        "oracle_by_family": oracle_by_family,
        "truncation_accounting": truncation_accounting,
        "gate_criterion": "oracle_context correct > 0",
        "gate_passed": gate_passed,
        "gate_meaning": (
            "A pass means the frozen organ can express at least one v4-r20 task "
            "under oracle control, i.e. the mouth-cortex protocol is not refuted "
            "for these tasks. It is not evidence of cortex learning, and a failure "
            "is an articulation block, not a cortex refutation. v3 and v4-r20 "
            "scores are not comparable as a causal improvement."
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
                "truncation_accounting": truncation_accounting,
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
