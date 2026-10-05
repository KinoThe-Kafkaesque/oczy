"""Audit public curricula without model runs, score changes or sealed data access."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_dev_teaching_coverage import audit_catalog, taught_facts  # noqa: E402
from scripts.materialize_dev_prompt_repair import load_repair  # noqa: E402


def audit_eval_v2():
    from eval.v2 import verify_manifest
    from oczy.eval_v2.scoring import matches, probe_matches
    from oczy.eval_v2.validation import validate_curriculum, validate_split
    from oczy.experiments.organism_curriculum.dataset import build_curriculum, split_probes

    verify_manifest()
    stages = build_curriculum()
    rows = []
    for stage in stages:
        pairs = [(episode, probe) for episode in stage.episodes for probe in episode.probes]
        dev, holdout = split_probes(stage)
        shared_episodes = sum(
            any(f"{episode.id}|{p.request}|{p.category}" in dev for p in episode.probes)
            and any(f"{episode.id}|{p.request}|{p.category}" in holdout for p in episode.probes)
            for episode in stage.episodes
        )
        rows.append({"stage": stage.name, "episodes": len(stage.episodes), "probes": len(pairs),
                     "match_modes": dict(Counter(p.match_mode for _, p in pairs)),
                     "blank_correct": sum(probe_matches("", p, e) for e, p in pairs),
                     "expected_answer_correct": sum(probe_matches(p.expected, p, e) for e, p in pairs),
                     "dev": len(dev), "holdout": len(holdout), "episodes_in_both_splits": shared_episodes})
    foils = [("fashion", "machine learning model", "model"),
             ("git", "tree branch", "branch"), ("spreadsheet", "biology cell", "cell")]
    return {"stages": rows, "validator": asdict(validate_curriculum(stages)),
            "split_validator": asdict(validate_split(stages)),
            "wrong_sense_foils": [{"answer": a, "expected": e, "ambiguous_token": w,
                "accepted_default": matches(a, e, w), "accepted_semantic": matches(a, e, w, semantic=True)}
                for a, e, w in foils]}


def transformation_candidates():
    """Enumerate the public generator grammar, independently of task targets."""
    from oczy.experiments.meta_cortex.taskgen import _SYMBOLS

    candidates = [("reverse", lambda x: x[::-1])]
    for token in _SYMBOLS:
        candidates.append((f"substitute:{token}", lambda x, t=token: "".join(t if c in "aeiou" else c for c in x)))
        candidates.append((f"reverse_substitute:{token}", lambda x, t=token: "".join(t if c in "aeiou" else c for c in x[::-1])))
    for suffix, prefix in itertools.product(_SYMBOLS, repeat=2):
        candidates.append((f"conditional:{suffix}:{prefix}", lambda x, s=suffix, p=prefix: x + s if x and x[0] in "aeiou" else p + x))
    return candidates


def audit_r20(catalog):
    coverage = audit_catalog(catalog)
    grammar = transformation_candidates()
    rows, conflicts, duplicates = [], [], []
    for split, tasks in (("train", catalog.meta_train), ("tuning", catalog.meta_validation)):
        for index, task in enumerate(tasks):
            facts = taught_facts(task)
            family = task.family.value
            if len(facts) < len(task.events):
                duplicates.append({"split": split, "index": index, "family": family,
                                   "events": len(task.events), "distinct_facts": len(facts)})
            if family == "rule_transformation":
                compatible = [(name, fn) for name, fn in grammar if all(fn(key[0]) == value for key, value in facts.items())]
                if not compatible:
                    raise ValueError("No public-grammar rule explains teaching data")
                for kind in ("transfer", "composition"):
                    for probe in getattr(task.probes, kind):
                        operand = probe.messages[-1].content.split(": ", 1)[1]
                        outputs = {name: fn(operand) if kind == "transfer" else fn(fn(operand)) for name, fn in compatible}
                        distinct = sorted(set(outputs.values()))
                        witnesses = {}
                        for name, value in outputs.items():
                            if value not in witnesses.values():
                                witnesses[name] = value
                            if len(witnesses) == 2:
                                break
                        if probe.expected_response not in distinct:
                            raise ValueError("Expected target outside all teaching-consistent rules")
                        rows.append({"split": split, "index": index, "kind": kind, "input": operand,
                                     "teaching": [{"input": k[0], "target": v} for k, v in facts.items()],
                                     "compatible_rules": len(compatible), "possible_targets": distinct,
                                     "determined_by_teaching": len(distinct) == 1,
                                     "witnesses": witnesses})
            for probe in task.probes.specificity:
                # Ignore the shared v3/v4 bare-answer system instruction. It
                # controls rendering, and does not introduce a new task scope.
                for event in task.events:
                    question = tuple(m.content for m in probe.messages if m.role == "user")
                    taught_question = tuple(m.content for m in event.observation_messages if m.role == "user")
                    if question == taught_question:
                        taught = next(iter(taught_facts(SimpleNamespace(events=(event,), family=task.family)).values()))
                        if taught != probe.expected_response:
                            conflicts.append({"split": split, "index": index, "family": family,
                                              "prompt": probe.messages[-1].content, "taught": taught,
                                              "specificity_target": probe.expected_response})
    grouped = defaultdict(Counter)
    for row in rows:
        count = grouped[f"{row['split']}/{row['kind']}"]
        count["total"] += 1
        count["underdetermined"] += not row["determined_by_teaching"]
    return {"coverage": coverage, "transformation_identifiability": {"grammar_size": len(grammar), "summary": dict(grouped), "rows": rows},
            "specificity_contradictions": conflicts, "duplicate_teaching": duplicates}


def audit_tools():
    from oczy.experiments.tool_calling_curriculum.dataset import build_tool_curriculum
    from oczy.experiments.tool_calling_curriculum.scoring import score_episode
    from oczy.experiments.tool_calling_curriculum.validation import validate_tool_curriculum

    stages = build_tool_curriculum()
    parameter_rows, extra_rows = [], []
    for stage in stages:
        for episode in stage.episodes:
            if not episode.expected_tools:
                continue
            calls = [json.dumps({"name": name, "arguments": dict(episode.expected_params) if i == 0 else {}})
                     for i, name in enumerate(episode.expected_tools)]
            answer = " ".join(episode.expected_answer_terms)
            extra = calls + [json.dumps({"name": "unexpected_call", "arguments": {}}), answer]
            extra_rows.append({"episode": episode.id, "extra_call_accepted": score_episode(episode, extra).passed})
            if episode.expected_params:
                wrong = [json.dumps({"name": episode.expected_tools[0], "arguments":
                                    {k: "WRONG/" + v + ".bak" for k, v in episode.expected_params}})] + calls[1:] + [answer]
                parameter_rows.append({"episode": episode.id, "wrong_parameters_accepted": score_episode(episode, wrong).passed})
    return {"episodes": sum(len(s.episodes) for s in stages), "validation_errors": validate_tool_curriculum(stages),
            "wrong_parameter_cases": parameter_rows, "extra_call_cases": extra_rows,
            "scope": "Recorded-output scoring only; no tool execution or environment-result validation"}


def audit_toy_rule_algebra():
    from oczy.experiments.r24_tiny_decoder.toy_catalog_v3 import (
        brute_force_identifiability_audit,
        make_interaction,
        matching_candidates,
        registered_candidates,
    )

    # No catalog builder or split selection: enumerate public algebra only.
    rules = registered_candidates()
    report = asdict(brute_force_identifiability_audit(rules))
    report["unique_after_A_and_C"] = sum(
        len(matching_candidates(tuple(make_interaction(rule, i) for i in (0, 2)), candidates=rules)) == 1
        for rule in rules
    )
    report["scope"] = "Public rule algebra only; no sealed catalog/assignments loaded"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog, manifest = load_repair(args.public_root, args.instrument)
    report = {"schema": "oczy/curriculum-eval-audit/v1", "model_runs": 0,
              "meta_test_accessed": False, "instrument_changed": False,
              "r20_public_manifest_sha256": manifest["manifest_sha256"],
              "eval_v2": audit_eval_v2(), "r20": audit_r20(catalog),
              "tool_curriculum": audit_tools(), "r24_public_algebra": audit_toy_rule_algebra()}
    files = ["scripts/audit_curricula_evals.py", "src/oczy/eval_v2/scoring.py",
             "src/oczy/eval_v2/validation.py", "src/oczy/experiments/meta_cortex/taskgen.py",
             "src/oczy/experiments/tool_calling_curriculum/scoring.py",
             "src/oczy/experiments/r24_tiny_decoder/toy_catalog_v3.py"]
    repo = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files}
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"eval_blank_correct": sum(s["blank_correct"] for s in report["eval_v2"]["stages"]),
                      "r20_identifiability": report["r20"]["transformation_identifiability"]["summary"],
                      "specificity_contradictions": len(report["r20"]["specificity_contradictions"]),
                      "duplicate_teaching": len(report["r20"]["duplicate_teaching"]),
                      "tool_wrong_parameters_accepted": sum(r["wrong_parameters_accepted"] for r in report["tool_curriculum"]["wrong_parameter_cases"]),
                      "tool_extra_calls_accepted": sum(r["extra_call_accepted"] for r in report["tool_curriculum"]["extra_call_cases"]),
                      "r24_unique_after_three": report["r24_public_algebra"]["unique_after_three"]}, indent=2))


if __name__ == "__main__":
    main()
