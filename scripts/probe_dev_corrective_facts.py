"""Prepare/run a separately approved DEV teaching-presentation comparison.

Preparing emits all proposed prompts without running a model. Running requires
an explicit approval record bound to that exact proposal hash. V2/v3/v4 inputs
and prior diagnostic scripts are never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oczy.experiments.meta_cortex.contracts import DialogueMessage, TaskFamily  # noqa: E402
from scripts.audit_dev_teaching_coverage import probe_key, taught_facts  # noqa: E402
from scripts.materialize_dev_prompt_repair import digest, load_repair  # noqa: E402
from scripts.probe_r20_articulation import with_teaching_context  # noqa: E402

CONDITIONS = ("no_context", "teaching_context", "corrective_facts_context", "oracle_context")


def corrective_facts_messages(events, query_messages):
    """Only public corrections and query text enter this model-facing renderer."""
    facts = "Corrected examples:\n" + "\n".join("- " + event.correction for event in events)
    return with_teaching_context(query_messages, (DialogueMessage("user", facts),))


def serialize(messages):
    return [{"role": message.role, "content": message.content} for message in messages]


def prompt_hash(messages):
    # Match the existing articulation diagnostic's JSON encoding exactly.
    return hashlib.sha256(json.dumps(serialize(messages), sort_keys=True).encode()).hexdigest()


def build_cases(catalog):
    cases = []
    for family in TaskFamily:
        task = next(t for t in catalog.meta_validation if t.family == family)
        transcript = []
        for event in task.events:
            transcript.extend(event.observation_messages)
            transcript.extend((DialogueMessage("assistant", event.attempted_behavior),
                               DialogueMessage("user", event.correction)))
        for condition in CONDITIONS:
            probes = task.probes.oracle_context if condition == "oracle_context" else task.probes.same_rule
            for index, probe in enumerate(probes):
                messages = probe.messages
                if condition == "teaching_context":
                    messages = with_teaching_context(messages, transcript)
                elif condition == "corrective_facts_context":
                    messages = corrective_facts_messages(task.events, messages)
                taught = None if condition == "oracle_context" else probe_key(
                    family.value, "same_rule", probe.messages[-1].content) in taught_facts(task)
                cases.append({"family": family.value, "condition": condition, "probe_index": index,
                              "rule_fingerprint": task.rule_fingerprint,
                              "messages": serialize(messages), "prompt_sha256": prompt_hash(messages),
                              "expected": probe.expected_response, "directly_taught": taught})
    return cases


def proposal(public_root, parent):
    catalog, manifest = load_repair(public_root, parent)
    if manifest["instrument_version"] != "v4":
        raise ValueError("This proposal requires the unchanged v4 parent")
    cases = build_cases(catalog)
    body = {
        "schema": "oczy/dev-corrective-facts-proposal/v1", "instrument_id": "meta_cortex/v5",
        "status": "PROPOSED_NOT_AUTHORIZED", "scope": "DEV_DIAGNOSTIC_ONLY",
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "base_dev_view_sha256": manifest["base_dev_view_sha256"],
        "organ_hash": manifest["organ_hash"], "scorer_sha256": manifest["scorer_sha256"],
        "generation": {"mode": "greedy", "max_new_tokens": 32, "soft_bank_width": 0},
        "selection": "First tuning task of each family, all existing same-rule and oracle probes",
        "conditions": CONDITIONS, "changed_variable": "Public teaching transcript presentation",
        "new_prefix": "Corrected examples:\n- {verbatim event.correction in original order}",
        "optimizer_steps": 0, "meta_test_authorized": False, "calibration_authorized": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "organ_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/oczy/experiments/meta_cortex/organ.py").read_bytes()).hexdigest(),
        "cases": cases,
        "interpretation": "Keep untaught probes in the table. A presentation effect is neither cortex learning nor a repair of task coverage. Recovery remains undefined if the context-minus-no-context denominator is zero.",
    }
    # Round trip tuple metadata into JSON-native data before comparisons.
    body = json.loads(json.dumps(body))
    return dict(body, manifest_sha256=digest(body))


def validate_approval(approval, manifest):
    if (approval.get("decision") != "approved"
            or approval.get("manifest_sha256") != manifest["manifest_sha256"]
            or approval.get("scope") != "DEV_DIAGNOSTIC_ONLY"
            or not isinstance(approval.get("human_reply"), str)
            or not approval["human_reply"].strip()):
        raise ValueError("A human approval record for this exact DEV proposal is required")


def run(manifest, output, approval):
    validate_approval(approval, manifest)
    import torch

    from oczy.experiments.meta_cortex.calibration import FrozenScorer
    from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan

    if output.exists():
        raise ValueError("Output must be new; preserve every attempt")
    scorer = FrozenScorer()
    if scorer.sha256 != manifest["scorer_sha256"]:
        raise ValueError("Scorer changed")
    output.mkdir(parents=True)
    (output / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
    torch.set_num_threads(4)
    organ = QwenFrozenOrgan.load()
    try:
        before = organ.parameter_hash()
        if before != manifest["organ_hash"]:
            raise ValueError("Historical organ identity changed")
        rows = []
        bank = torch.zeros((1, 0, organ.feature_dim))
        for case in manifest["cases"]:
            messages = tuple(DialogueMessage(**m) for m in case["messages"])
            generated = organ.generate(messages, bank, max_new_tokens=32)
            row = dict(case, generated=generated, correct=scorer.score_response(case["expected"], generated))
            rows.append(row)
            with (output / "rows.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
        organ.assert_frozen()
        after = organ.parameter_hash()
        if after != before:
            raise ValueError("Organ parameters changed")
        result = {"schema": "oczy/dev-corrective-facts-results/v1", "instrument_id": manifest["instrument_id"],
                  "classification": "DEV_DIAGNOSTIC_ONLY", "manifest_sha256": manifest["manifest_sha256"],
                  "organ_hash_before": before, "organ_hash_after": after,
                  "scorer_sha256": scorer.sha256, "optimizer_steps": 0,
                  "meta_test_accessed": False, "calibration_accessed": False, "rows": rows}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            print(json.dumps({"condition": condition, "correct": sum(r["correct"] for r in selected), "total": len(selected)}), flush=True)
    finally:
        organ.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = proposal(args.public_root, args.parent)
    if args.action == "prepare":
        with args.proposal.open("x") as file:
            json.dump(expected, file, indent=2)
            file.write("\n")
        print(expected["manifest_sha256"])
    else:
        if args.approval is None or args.output is None:
            parser.error("run requires --approval and --output")
        saved = json.loads(args.proposal.read_text())
        if saved != expected:
            raise ValueError("Proposal, source, or original instrument changed")
        run(saved, args.output, json.loads(args.approval.read_text()))


if __name__ == "__main__":
    main()
