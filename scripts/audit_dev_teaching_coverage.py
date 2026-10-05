"""Read-only coverage audit of public DEV prompts; no model or scorer changes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.materialize_dev_prompt_repair import load_repair  # noqa: E402


def match(pattern, text):
    found = re.fullmatch(pattern, text)
    if found is None:
        raise ValueError(f"Unrecognized frozen text: {text!r}")
    return found.groups()


def taught_facts(task):
    """Parse public corrections only; no oracle/target/fingerprint input."""
    facts = {}
    for event in task.events:
        text = event.correction
        if task.family.value == "contextual_remap":
            context, symbol, answer = match(r"In the (\w+) room, (\w+) requires the token (\w+)\.", text)
            key = (context, symbol)
        elif task.family.value == "finite_state":
            state, signal, answer = match(r"From (\w+), input (\w+) transitions to (\w+)\.", text)
            key = (state, signal)
        else:
            operand, answer = match(r"The correct result for (\w+) is (\w+)\.", text)
            key = (operand,)
        if key in facts and facts[key] != answer:
            raise ValueError("Contradictory public teaching facts")
        facts[key] = answer
    return facts


def probe_key(family, kind, text):
    if family == "contextual_remap":
        if kind == "same_rule":
            return match(r"You are in the (\w+) chamber\. (\w+) demands what token\?", text)
        if text.startswith("Setting:"):
            return match(r"Setting: (\w+) environment\. Command word: (\w+)\. What is the correct response\?", text)
        return match(r"Within the (\w+) domain, the signal (\w+) elicits what\?", text)
    if family == "finite_state":
        if kind == "same_rule":
            return match(r"Given current state (\w+) and signal (\w+), which state follows\?", text)
        return match(r"State: (\w+)\. Signal: (\w+)\. Next state\?", text)
    if kind == "same_rule":
        return match(r"What is the transformed output for input (\w+)\?", text)
    return match(r"Transform: (\w+)", text)


def audit_catalog(catalog):
    rows = []
    for split, tasks in (("train", catalog.meta_train), ("tuning", catalog.meta_validation)):
        for index, task in enumerate(tasks):
            facts = taught_facts(task)
            family = task.family.value
            for kind in ("same_rule", "transfer"):
                for probe_index, probe in enumerate(getattr(task.probes, kind)):
                    key = probe_key(family, kind, probe.messages[-1].content)
                    supported = key in facts
                    if supported and facts[key] != probe.expected_response:
                        raise ValueError("Frozen probe target contradicts its teaching fact")
                    rows.append({"split": split, "task_index": index, "family": family,
                                 "rule_fingerprint": task.rule_fingerprint, "kind": kind,
                                 "probe_index": probe_index, "queried_key": key,
                                 "directly_taught": supported,
                                 "unconstrained_lookup_not_taught": not supported and family != "rule_transformation"})
            if family == "contextual_remap":
                # Inspect oracle data for an audit only. It is never supplied
                # to teaching-context generation by this script.
                oracle = task.probes.oracle_context[0].messages[0 if task.probes.oracle_context[0].messages[0].role != "system" else 1].content
                defined = {(ctx, sym) for ctx, sym, _ in re.findall(r"  (\w+) / (\w+) -> (\w+)", oracle)}
                for probe_index, probe in enumerate(task.probes.composition):
                    ctx1, sym1, ctx2, sym2 = match(r"First, in the (\w+) room, respond to (\w+)\. Then, in the (\w+) room, respond to (\w+)\. Give both tokens in order\.", probe.messages[-1].content)
                    rows.append({"split": split, "task_index": index, "family": family,
                                 "rule_fingerprint": task.rule_fingerprint, "kind": "composition",
                                 "probe_index": probe_index, "second_operand_defined": (ctx2, sym2) in defined,
                                 "first_key_taught": (ctx1, sym1) in facts})
            elif family == "finite_state":
                for probe_index, probe in enumerate(task.probes.composition):
                    rows.append({"split": split, "task_index": index, "family": family,
                                 "rule_fingerprint": task.rule_fingerprint, "kind": "composition",
                                 "probe_index": probe_index, "requires_action": len(probe.expected_response.split()) == 2,
                                 "actions_in_teaching": False})
    summary = {}
    for row in rows:
        group = "/".join(row[k] for k in ("split", "family", "kind"))
        count = summary.setdefault(group, Counter())
        count["total"] += 1
        for name in ("directly_taught", "unconstrained_lookup_not_taught", "second_operand_defined", "first_key_taught", "requires_action", "actions_in_teaching"):
            if name in row:
                count[name] += int(row[name])
    return {"schema": "oczy/dev-teaching-coverage-audit/v1", "classification": "READ_ONLY_INSTRUMENT_AUDIT",
            "meta_test_accessed": False, "calibration_accessed": False, "model_runs": 0,
            "scoring_or_instrument_changed": False, "summary": dict(sorted(summary.items())), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog, manifest = load_repair(args.public_root, args.instrument)
    report = audit_catalog(catalog)
    report["parent_manifest_sha256"] = manifest["manifest_sha256"]
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
