#!/usr/bin/env python3
"""Independent task-support checker for the R20 public DEV task generators.

This checker is deliberately **separate from the generators**: it re-derives
every scored target from rendered teaching text and rendered probe questions
only.  It never reads a generator's rule structure, fingerprints or support
certificate to decide whether a task is supported — the certificate is checked
against the rendered text, not the other way around.

It evaluates two catalog sources with the same defect classes:

* the historical v1 generator (``oczy/meta-cortex/taskgen/v1-dev``), either
  rebuilt in memory from its recorded config or loaded from a materialized
  public instrument root;
* the repaired v2 lineage (``oczy/meta-cortex/taskgen/v2-dev``).

No model, scorer, threshold, sealed file, calibration file or meta-test task
is read, generated, or changed.  The measurement scorer stays
``normalized-exact/v1``; this script scores nothing.

Commands
--------
``dry-run``
    Build both lineages, count every defect class before and after, verify
    the v2 support certificates (including delete-a-fact / flip-a-target
    mutation checks, counted separately because the flip-target family is
    non-discriminating by construction), and write a JSON report.  Exits
    nonzero unless the v1 counts reproduce the recorded 2026-09-12 audit
    numbers and every v2 defect count is zero.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oczy.experiments.meta_cortex.contracts import (  # noqa: E402
    MetaTask,
    ProbeKind,
    TaskGeneratorConfig,
)

SCHEMA = "oczy/r20-task-support-check/v1"

# Recorded defect counts of the 2026-09-12 public DEV audit
# (experiments_logs/2026-09-12_dev_teaching_coverage.json and
# experiments_logs/2026-09-12_curriculum_eval_audit.json).  The dry run must
# reproduce them on the unchanged v1 generator before any repair is believed.
EXPECTED_V1_DEFECTS = {
    "contextual_same_rule_lookup_untaught": {"train": 37, "tuning": 7, "total": 44,
                                             "of": {"train": 75, "tuning": 12, "total": 87}},
    "contextual_transfer_lookup_untaught": {"train": 25, "tuning": 5, "total": 30,
                                            "of": {"train": 60, "tuning": 10, "total": 70}},
    "fsm_transfer_edge_untaught": {"train": 52, "tuning": 7, "total": 59,
                                   "of": {"train": 52, "tuning": 7, "total": 59}},
    "composition_second_operand_undefined": {"train": 30, "tuning": 5, "total": 35,
                                             "of": {"train": 30, "tuning": 5, "total": 35}},
    "fsm_composition_action_untaught": {"train": 30, "tuning": 5, "total": 35,
                                        "of": {"train": 30, "tuning": 5, "total": 35}},
    "fsm_action_mapping_untaught": {"train": 30, "tuning": 5, "total": 35,
                                    "of": {"train": 30, "tuning": 5, "total": 35}},
    "fsm_goal_untaught": {"train": 30, "tuning": 5, "total": 35,
                          "of": {"train": 30, "tuning": 5, "total": 35}},
    "transform_transfer_underdetermined": {"train": 1, "tuning": 2, "total": 3,
                                           "of": {"train": 60, "tuning": 10, "total": 70}},
    "transform_composition_underdetermined": {"train": 1, "tuning": 1, "total": 2,
                                              "of": {"train": 30, "tuning": 5, "total": 35}},
    "specificity_contradiction": {"train": 20, "tuning": 3, "total": 23,
                                  "of": {"train": 90, "tuning": 15, "total": 105}},
    "duplicate_teaching_event": {"train": 1, "tuning": 0, "total": 1,
                                 "of": {"train": 90, "tuning": 15, "total": 105}},
}

# Denominator unit per defect class: "probes" (scored probe instances) or
# "tasks" (one row per task).
_CLASS_UNITS = {
    "contextual_same_rule_lookup_untaught": "probes",
    "fsm_same_rule_lookup_untaught": "probes",
    "transform_same_rule_underdetermined": "probes",
    "contextual_transfer_lookup_untaught": "probes",
    "fsm_transfer_edge_untaught": "probes",
    "transform_transfer_underdetermined": "probes",
    "composition_second_operand_undefined": "probes",
    "contextual_composition_first_lookup_untaught": "probes",
    "contextual_composition_second_lookup_untaught": "probes",
    "contextual_composition_chain_untyped": "probes",
    "fsm_composition_edge_untaught": "probes",
    "fsm_composition_action_untaught": "probes",
    "fsm_composition_goal_untaught": "probes",
    "fsm_composition_underivable": "probes",
    "transform_composition_underdetermined": "probes",
    "target_mismatch": "probes",
    "specificity_scope_unstated": "probes",
    "specificity_contradiction": "probes",
    "question_target_contradiction": "questions",
    "duplicate_teaching_event": "tasks",
    "fsm_action_mapping_untaught": "tasks",
    "fsm_goal_untaught": "tasks",
}

# Which scored probes each probe-level class counts over.
_CLASS_PROBES = {
    "contextual_same_rule_lookup_untaught": ("contextual_remap", "same_rule"),
    "fsm_same_rule_lookup_untaught": ("finite_state", "same_rule"),
    "transform_same_rule_underdetermined": ("rule_transformation", "same_rule"),
    "contextual_transfer_lookup_untaught": ("contextual_remap", "transfer"),
    "fsm_transfer_edge_untaught": ("finite_state", "transfer"),
    "transform_transfer_underdetermined": ("rule_transformation", "transfer"),
    "composition_second_operand_undefined": ("contextual_remap", "composition"),
    "contextual_composition_first_lookup_untaught": ("contextual_remap", "composition"),
    "contextual_composition_second_lookup_untaught": ("contextual_remap", "composition"),
    "contextual_composition_chain_untyped": ("contextual_remap", "composition"),
    "fsm_composition_edge_untaught": ("finite_state", "composition"),
    "fsm_composition_action_untaught": ("finite_state", "composition"),
    "fsm_composition_goal_untaught": ("finite_state", "composition"),
    "fsm_composition_underivable": ("finite_state", "composition"),
    "transform_composition_underdetermined": ("rule_transformation", "composition"),
    "target_mismatch": None,  # all scored probes
    "specificity_scope_unstated": None,  # all specificity probes
    "specificity_contradiction": None,  # all specificity probes
}

_TASK_FAMILIES = {
    "duplicate_teaching_event": None,  # all tasks
    "fsm_action_mapping_untaught": "finite_state",
    "fsm_goal_untaught": "finite_state",
}


# ---------------------------------------------------------------------------
# Independent enumeration of the public transformation grammar (289 rules)
# ---------------------------------------------------------------------------

_SYMBOLS = (
    "dax", "fep", "grim", "hool", "jir", "kal", "lom", "nurp",
    "pex", "quob", "ral", "siv", "twem", "urb", "vol", "wix",
)
_VOWELS = "aeiou"


def grammar_candidates() -> list[tuple[str, Callable[[str], str]]]:
    """All rules of the public grammar; mirrors the 2026-09-12 audit."""
    candidates: list[tuple[str, Callable[[str], str]]] = [("reverse", lambda x: x[::-1])]
    for token in _SYMBOLS:
        candidates.append(
            (f"substitute:{token}",
             lambda x, t=token: "".join(t if c in _VOWELS else c for c in x))
        )
        candidates.append(
            (f"reverse_substitute:{token}",
             lambda x, t=token: "".join(t if c in _VOWELS else c for c in x[::-1]))
        )
    for suffix, prefix in itertools.product(_SYMBOLS, repeat=2):
        candidates.append(
            (f"conditional:{suffix}:{prefix}",
             lambda x, s=suffix, p=prefix: x + s if x and x[0] in _VOWELS else p + x)
        )
    return candidates


GRAMMAR = grammar_candidates()

# ---------------------------------------------------------------------------
# Rendered-text patterns (teaching and probes)
# ---------------------------------------------------------------------------

_CORRECTIONS = (
    ("contextual_lookup", re.compile(r"In the (\w+) room, (\w+) requires the token (\w+)\.")),
    ("transformation_pair", re.compile(r"The correct result for (\w+) is (\w+)\.")),
    ("fsm_edge", re.compile(r"From (\w+), input (\w+) transitions to (\w+)\.")),
    ("fsm_action", re.compile(r"In state (\w+), the action is (\w+)\.")),
    ("fsm_goal", re.compile(r"The goal is to reach state (\w+)\.")),
)

_QUESTIONS: dict[tuple[str, str], tuple[re.Pattern[str], ...]] = {
    ("contextual_remap", "same_rule"): (
        re.compile(r"You are in the (\w+) chamber\. (\w+) demands what token\?"),
    ),
    ("contextual_remap", "transfer"): (
        re.compile(r"Setting: (\w+) environment\. Command word: (\w+)\. What is the correct response\?"),
        re.compile(r"Within the (\w+) domain, the signal (\w+) elicits what\?"),
    ),
    ("contextual_remap", "composition"): (
        re.compile(r"First, in the (\w+) room, respond to (\w+)\. Then, in the (\w+) room, respond to (\w+)\. Give both tokens in order\."),
    ),
    ("rule_transformation", "same_rule"): (
        re.compile(r"What is the transformed output for input (\w+)\?"),
    ),
    ("rule_transformation", "transfer"): (
        re.compile(r"Transform: (\w+)"),
    ),
    ("rule_transformation", "composition"): (
        re.compile(r"Apply the rule twice to: (\w+)"),
    ),
    ("finite_state", "same_rule"): (
        re.compile(r"Given current state (\w+) and signal (\w+), which state follows\?"),
    ),
    ("finite_state", "transfer"): (
        re.compile(r"State: (\w+)\. Signal: (\w+)\. Next state\?"),
    ),
    ("finite_state", "composition"): (
        # v2: goal-stated goal-dependent decision
        re.compile(r"Start at (\w+)\. First input: (\w+)\. Then input: (\w+)\. Goal: reach (\w+)\. Report the final state and its action, using (\w+) for the action when the goal is reached\."),
        # v1: goal-free two-step result
        re.compile(r"Start at (\w+)\. First input: (\w+)\. Then input: (\w+)\. What is the result\?"),
    ),
}

_ORACLE_MAPPING = re.compile(r"  (\w+) / (\w+) -> (\w+)")
_ORACLE_EDGE = re.compile(r"  (\w+) \+ (\w+) -> (\w+)")
_ORACLE_ACTION = re.compile(r"  (\w+): (\w+)")
_ORACLE_GOAL = re.compile(r"Goal: reach (\w+)\.")
_SCOPE_CLAUSE = re.compile(r"^(Unrelated|Unchanged)", re.IGNORECASE)

# Explicit-scope question forms (v2 specificity probes).  Each states its own
# rule in the question, so the target is derivable from visible text only.
_SPECIFICITY_QUESTIONS = {
    "contextual_remap": (
        re.compile(r"Unrelated rule for the (\w+) room: (\w+) answers with (\w+)\. What response follows (\w+)\?"),
    ),
    "rule_transformation": (
        re.compile(r"Unrelated rule: the input is returned unchanged\. Apply the unrelated rule to: (\w+)"),
    ),
    "finite_state": (
        re.compile(r"Unrelated machine: from state (\w+), input (\w+) transitions to (\w+)\. State: (\w+)\. Signal: (\w+)\. Next state\?"),
    ),
}

_ORACLE_QUESTIONS = {
    "contextual_remap": re.compile(
        r"Given the above mapping, in the (\w+) room, what token follows (\w+)\?"
    ),
    "rule_transformation": re.compile(r"Given this rule, what is the output for: (\w+)\?"),
    "finite_state": re.compile(
        r"Given the above graph, from (\w+) with input (\w+), what is the next state\?"
    ),
}

_TRANSFORM_HEADER = re.compile(
    r"Rule: (permutation|substitution|conditional|composition) with parameters "
    r"'([a-z]*)' and '([a-z]*)'\.\nWorked examples:\n(.*)", re.DOTALL
)
_TRANSFORM_EXAMPLE = re.compile(r"  (\w+) -> (\w+)")


@dataclass
class ParsedProbe:
    kind: str
    question: str
    expected: str
    query: dict[str, Any] = field(default_factory=dict)
    payload: str = ""


@dataclass
class ParsedTask:
    split: str
    family: str
    index: int
    events: list[dict[str, Any]]
    facts: dict[tuple[str, ...], str]
    fact_kinds: dict[tuple[str, ...], str]
    probes: dict[str, list[ParsedProbe]]
    oracle: dict[str, Any]
    rule_fingerprint: str


def _user_question(probe: Any) -> str:
    return [m.content for m in probe.messages if m.role == "user"][-1]


def _user_questions(event: Any) -> tuple[str, ...]:
    return tuple(m.content for m in event.observation_messages if m.role == "user")


def parse_task(split: str, index: int, task: MetaTask) -> tuple[ParsedTask, list[dict[str, Any]]]:
    defects: list[dict[str, Any]] = []
    family = task.family.value
    events: list[dict[str, Any]] = []
    facts: dict[tuple[str, ...], str] = {}
    fact_kinds: dict[tuple[str, ...], str] = {}
    for event in task.events:
        matched = None
        for kind, pattern in _CORRECTIONS:
            found = pattern.fullmatch(event.correction)
            if found is not None:
                matched = (kind, found.groups())
                break
        if matched is None:
            defects.append({
                "class": "unrecognized_teaching_sentence", "split": split,
                "family": family, "task_index": index, "correction": event.correction,
            })
            continue
        kind, groups = matched
        if kind == "contextual_lookup":
            key, target = (groups[0], groups[1]), groups[2]
        elif kind == "transformation_pair":
            key, target = (groups[0],), groups[1]
        elif kind == "fsm_edge":
            key, target = (groups[0], groups[1]), groups[2]
        elif kind == "fsm_action":
            key, target = (groups[0],), groups[1]
        else:  # fsm_goal
            key, target = ("goal",), groups[0]
        events.append({"kind": kind, "key": key, "target": target,
                       "correction": event.correction, "questions": _user_questions(event)})
        if key in facts:
            defects.append({
                "class": "duplicate_teaching_event", "split": split, "family": family,
                "task_index": index, "key": list(key),
            })
        facts[key] = target
        fact_kinds[key] = kind

    oracle: dict[str, Any] = {}
    for probe in task.probes.oracle_context:
        head = probe.messages[0].content
        if family == "contextual_remap":
            oracle["mapping"] = {
                (c, s): o for c, s, o in _ORACLE_MAPPING.findall(head)
            }
        elif family == "finite_state":
            oracle["edges"] = {(s, i): n for s, i, n in _ORACLE_EDGE.findall(head)}
            oracle["actions"] = dict(_ORACLE_ACTION.findall(head))
            goal = _ORACLE_GOAL.search(head)
            oracle["goal"] = goal.group(1) if goal else None

    probes: dict[str, list[ParsedProbe]] = {}
    for kind in ProbeKind:
        parsed: list[ParsedProbe] = []
        for probe in task.probes.by_kind(kind):
            user_contents = [m.content for m in probe.messages if m.role == "user"]
            question = user_contents[-1]
            payload = user_contents[0] if len(user_contents) > 1 else ""
            query: dict[str, Any] = {}
            if kind.value in ("same_rule", "transfer", "composition"):
                patterns = _QUESTIONS.get((family, kind.value), ())
                for pattern in patterns:
                    found = pattern.fullmatch(question)
                    if found is not None:
                        query["groups"] = found.groups()
                        query["pattern"] = pattern.pattern
                        break
                else:
                    defects.append({
                        "class": "unrecognized_probe_question", "split": split,
                        "family": family, "task_index": index, "kind": kind.value,
                        "question": question,
                    })
            parsed.append(
                ParsedProbe(kind=kind.value, question=question,
                            expected=probe.expected_response, query=query,
                            payload=payload)
            )
        probes[kind.value] = parsed

    return (
        ParsedTask(split=split, family=family, index=index, events=events, facts=facts,
                   fact_kinds=fact_kinds, probes=probes, oracle=oracle,
                   rule_fingerprint=task.rule_fingerprint),
        defects,
    )


# ---------------------------------------------------------------------------
# Independent derivation from rendered teaching text
# ---------------------------------------------------------------------------


def _derive_stated_scope(task: ParsedTask, probe: ParsedProbe) -> tuple[str | None, str]:
    """Derive a target from a rule that the probe itself states in visible text."""
    if probe.kind == "specificity":
        if _SCOPE_CLAUSE.match(probe.question) is None:
            return None, "specificity scope not stated in the question"
        for pattern in _SPECIFICITY_QUESTIONS.get(task.family, ()):
            found = pattern.fullmatch(probe.question)
            if found is None:
                continue
            groups = found.groups()
            if task.family == "contextual_remap":
                _ctx, subject, answer, queried = groups
                if subject != queried:
                    return None, "stated scope does not cover the queried symbol"
                return answer, "unchanged scope stated verbatim in the question"
            if task.family == "rule_transformation":
                return groups[0], "unchanged identity scope stated in the question"
            state, signal, nxt, q_state, q_signal = groups
            if (state, signal) != (q_state, q_signal):
                return None, "stated scope does not cover the queried state/signal"
            return nxt, "unrelated machine's transition stated in the question"
        return None, "specificity scope not stated in the question"

    # oracle_context: the payload states the complete rule.
    found = _ORACLE_QUESTIONS.get(task.family)
    if found is None:
        return None, "no oracle question form for this family"
    match = found.fullmatch(probe.question)
    if match is None:
        return None, "oracle question not recognized"
    if task.family == "contextual_remap":
        ctx, sym = match.groups()
        mapping = {(c, s): o for c, s, o in _ORACLE_MAPPING.findall(probe.payload)}
        if (ctx, sym) not in mapping:
            return None, "oracle query outside the stated mapping"
        return mapping[(ctx, sym)], "target read from the stated complete mapping"
    if task.family == "finite_state":
        state, signal = match.groups()
        edges = {(s, i): n for s, i, n in _ORACLE_EDGE.findall(probe.payload)}
        if (state, signal) not in edges:
            return None, "oracle query outside the stated graph"
        return edges[(state, signal)], "target read from the stated transition graph"
    (operand,) = match.groups()
    header = _TRANSFORM_HEADER.fullmatch(probe.payload)
    if header is None:
        return None, "transformation oracle header not recognized"
    template, param1, param2, _examples = header.groups()
    if template == "permutation":
        return operand[::-1], "stated rule (reverse) applied to the query operand"
    if template == "substitution":
        return "".join(param1 if c in _VOWELS else c for c in operand), \
            "stated rule (substitution) applied to the query operand"
    if template == "conditional":
        suffix, prefix = param1, param2
        if operand and operand[0] in _VOWELS:
            return operand + suffix, "stated conditional rule (vowel branch)"
        return prefix + operand, "stated conditional rule (consonant branch)"
    sub_char = param2
    reversed_op = operand[::-1]
    return "".join(sub_char if c in _VOWELS else c for c in reversed_op), \
        "stated composition rule applied to the query operand"


def derive_target(task: ParsedTask, probe: ParsedProbe) -> tuple[str | None, str]:
    """Return (target, reason) using teaching text and the question only.

    The one exception is ``kind == "pre"`` (the pre-learning baseline), where
    no teaching support is claimed and the recorded ``probe.expected`` is
    returned as the target; every other branch derives from rendered text.
    """
    groups = probe.query.get("groups")
    if probe.kind == "specificity":
        return _derive_stated_scope(task, probe)
    if probe.kind == "oracle_context":
        return _derive_stated_scope(task, probe)
    if probe.kind == "pre":
        return probe.expected, "pre-learning baseline: no teaching support required"
    if groups is None:
        return None, "question not recognized"

    if task.family == "contextual_remap":
        if probe.kind in ("same_rule", "transfer"):
            key = (groups[0], groups[1])
            if key not in task.facts:
                return None, f"lookup {key} not taught"
            return task.facts[key], f"taught lookup {key}"
        # composition
        ctx1, sym1, ctx2, sym2 = groups
        mapping = task.oracle.get("mapping", {})
        if (ctx1, sym1) not in task.facts:
            return None, f"first lookup {(ctx1, sym1)} not taught"
        first_out = task.facts[(ctx1, sym1)]
        if (ctx2, sym2) not in mapping:
            return None, f"second operand {(ctx2, sym2)} outside the rule domain"
        if sym2 != first_out:
            return None, "second operand is not the first lookup's output (untyped chain)"
        if (ctx2, sym2) not in task.facts:
            return None, f"second lookup {(ctx2, sym2)} not taught"
        second_out = task.facts[(ctx2, sym2)]
        return f"{first_out} then {second_out}", "typed chain over two taught lookups"

    if task.family == "finite_state":
        if probe.kind in ("same_rule", "transfer"):
            key = (groups[0], groups[1])
            if key not in task.facts:
                return None, f"transition {key} not taught"
            return task.facts[key], f"taught transition {key}"
        if len(groups) == 5:
            start, in1, in2, goal, terminal = groups
        else:
            start, in1, in2 = groups[:3]
            goal, terminal = None, None
        if goal is None:
            return None, "goal-dependent result but no goal is stated in the question"
        if (start, in1) not in task.facts:
            return None, f"first transition {(start, in1)} not taught"
        mid = task.facts[(start, in1)]
        if (mid, in2) not in task.facts:
            return None, f"second transition {(mid, in2)} not taught"
        final = task.facts[(mid, in2)]
        goal_fact = task.facts.get(("goal",))
        if goal_fact is None:
            return None, "goal not taught"
        if goal_fact != goal:
            return None, "probe goal contradicts the taught goal"
        if final == goal:
            return f"{final} {terminal}", "goal reached: stated terminal action"
        if (final,) not in task.facts:
            return None, f"action for {final} not taught"
        return f"{final} {task.facts[(final,)]}", "taught action for the final state"

    # rule_transformation: the target must be agreed on by every rule of the
    # public grammar that is consistent with the rendered teaching pairs.
    taught = {key[0]: target for key, target in task.facts.items()
              if task.fact_kinds.get(key) == "transformation_pair"}
    if probe.kind == "composition":
        values: set[str] = set()
        for _name, fn in GRAMMAR:
            if all(fn(k) == v for k, v in taught.items()):
                values.add(fn(fn(groups[0])))
    else:
        values = set()
        for _name, fn in GRAMMAR:
            if all(fn(k) == v for k, v in taught.items()):
                values.add(fn(groups[0]))
    if len(values) != 1:
        return None, f"teaching leaves {len(values)} possible targets"
    return values.pop(), "unique teaching-consistent rule of the public grammar"


def _counterexample(task: ParsedTask, probe: ParsedProbe, required: list[tuple[str, ...]]) -> bool:
    """True when dropping the required teaching facts removes the derivation."""
    kept = {k: v for k, v in task.facts.items() if k not in set(required)}
    mutated = ParsedTask(
        split=task.split, family=task.family, index=task.index, events=task.events,
        facts=kept, fact_kinds={k: v for k, v in task.fact_kinds.items() if k in kept},
        probes=task.probes, oracle=task.oracle, rule_fingerprint=task.rule_fingerprint,
    )
    target, _reason = derive_target(mutated, probe)
    return target is None


# ---------------------------------------------------------------------------
# Catalog checks
# ---------------------------------------------------------------------------


def check_task(task: ParsedTask) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []

    def add(cls: str, probe: ParsedProbe | None, **extra: Any) -> None:
        row: dict[str, Any] = {"class": cls, "split": task.split, "family": task.family,
                               "task_index": task.index}
        if probe is not None:
            row["kind"] = probe.kind
            row["question"] = probe.question
            row["expected"] = probe.expected
        row.update(extra)
        defects.append(row)

    lookup_cls = {
        "contextual_remap": ("contextual_same_rule_lookup_untaught",
                            "contextual_transfer_lookup_untaught"),
        "finite_state": ("fsm_same_rule_lookup_untaught", "fsm_transfer_edge_untaught"),
        "rule_transformation": ("transform_same_rule_underdetermined",
                               "transform_transfer_underdetermined"),
    }[task.family]

    for kind, cls in (("same_rule", lookup_cls[0]), ("transfer", lookup_cls[1])):
        for probe in task.probes[kind]:
            if probe.query.get("groups") is None:
                continue
            target, reason = derive_target(task, probe)
            if target is None:
                add(cls, probe, reason=reason)
            elif target != probe.expected:
                add("target_mismatch", probe, reason=reason, derived=target)

    if task.family == "contextual_remap":
        mapping = task.oracle.get("mapping", {})
        for probe in task.probes["composition"]:
            groups = probe.query.get("groups")
            if groups is None:
                continue
            ctx1, sym1, ctx2, sym2 = groups
            first_out = task.facts.get((ctx1, sym1))
            if first_out is None:
                add("contextual_composition_first_lookup_untaught", probe,
                    reason=f"first lookup {(ctx1, sym1)} not taught")
            if (ctx2, sym2) not in mapping:
                add("composition_second_operand_undefined", probe,
                    reason=f"second operand {(ctx2, sym2)} outside the rule domain")
                continue
            if first_out is not None and sym2 != first_out:
                add("contextual_composition_chain_untyped", probe,
                    reason="second operand is not the first lookup's output")
            if (ctx2, sym2) not in task.facts:
                add("contextual_composition_second_lookup_untaught", probe,
                    reason=f"second lookup {(ctx2, sym2)} not taught")
                continue
            expected = f"{first_out} then {task.facts[(ctx2, sym2)]}"
            if first_out is not None and expected != probe.expected:
                add("target_mismatch", probe, derived=expected)

    elif task.family == "finite_state":
        taught_actions = {
            target for key, target in task.facts.items()
            if task.fact_kinds.get(key) == "fsm_action"
        }
        for probe in task.probes["composition"]:
            groups = probe.query.get("groups")
            if groups is None:
                continue
            if len(groups) == 5:
                start, in1, in2, goal = groups[0], groups[1], groups[2], groups[3]
            else:
                start, in1, in2, goal = groups[0], groups[1], groups[2], None
            mid = task.facts.get((start, in1))
            edges_ok = mid is not None and (mid, in2) in task.facts
            if not edges_ok:
                add("fsm_composition_edge_untaught", probe,
                    reason="one of the two traversed transitions is not taught")
            parts = probe.expected.split()
            if len(parts) == 2 and parts[1] not in taught_actions:
                add("fsm_composition_action_untaught", probe, action=parts[1],
                    reason="the scored action token is not in teaching")
            if goal is not None:
                goal_fact = task.facts.get(("goal",))
                if goal_fact is None or goal_fact != goal:
                    add("fsm_composition_goal_untaught", probe,
                        reason="the probe's goal is not taught")
            target, reason = derive_target(task, probe)
            if target is None and edges_ok:
                add("fsm_composition_underivable", probe, reason=reason)
            elif target is not None and target != probe.expected:
                add("target_mismatch", probe, reason=reason, derived=target)

    else:  # rule_transformation
        for probe in task.probes["composition"]:
            if probe.query.get("groups") is None:
                continue
            target, reason = derive_target(task, probe)
            if target is None:
                add("transform_composition_underdetermined", probe, reason=reason)
            elif target != probe.expected:
                add("target_mismatch", probe, reason=reason, derived=target)

    # Task-level teaching-support facts.
    if task.family == "finite_state":
        if not any(task.fact_kinds.get(k) == "fsm_action" for k in task.facts):
            add("fsm_action_mapping_untaught", None,
                reason="no state action mapping is taught")
        if not any(task.fact_kinds.get(k) == "fsm_goal" for k in task.facts):
            add("fsm_goal_untaught", None, reason="no goal is taught")

    # Contradiction check: one observable question, one target — independent of
    # the hidden probe category.  Teaching observations take part too.  The
    # historical 2026-09-12 audit counted only specificity probes whose
    # question duplicates a teaching question with a different target; that
    # count is preserved as ``specificity_contradiction`` and the broader
    # category-free check is reported as ``question_target_contradiction``.
    targets_by_question: dict[tuple[str, str], set[str]] = {}
    teaching_questions: set[str] = set()
    for event in task.events:
        for question in event["questions"]:
            teaching_questions.add(question)
            targets_by_question.setdefault((task.family, question), set()).add(event["target"])
    for kind in ProbeKind:
        for probe in task.probes[kind.value]:
            targets_by_question.setdefault((task.family, probe.question), set()).add(probe.expected)
    for probe in task.probes["specificity"]:
        taught_targets = set()
        for event in task.events:
            if probe.question in event["questions"]:
                taught_targets.add(event["target"])
        if taught_targets and probe.expected not in taught_targets:
            add("specificity_contradiction", probe,
                taught_targets=sorted(taught_targets),
                reason="the same question is taught with a different target")
    for (family, question), targets in targets_by_question.items():
        if len(targets) > 1:
            add("question_target_contradiction", None, family_hint=family,
                question=question, targets=sorted(targets))

    # Specificity probes must state their unchanged scope explicitly.
    for probe in task.probes["specificity"]:
        target, reason = derive_target(task, probe)
        if target is None:
            add("specificity_scope_unstated", probe, reason=reason)
        elif target != probe.expected:
            add("target_mismatch", probe, reason=reason, derived=target)

    return defects


def check_catalog(catalog: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parsed_tasks: list[ParsedTask] = []
    parse_defects: list[dict[str, Any]] = []
    counters: dict[tuple[str, str], int] = {}
    for split_name, tasks in (("train", catalog.meta_train), ("tuning", catalog.meta_validation)):
        for task in tasks:
            family_key = (split_name, task.family.value)
            index = counters.get(family_key, 0)
            counters[family_key] = index + 1
            parsed, defects = parse_task(split_name, index, task)
            parsed_tasks.append(parsed)
            parse_defects.extend(defects)
            rows.extend(check_task(parsed))

    summary: dict[str, dict[str, Any]] = {}
    for row in rows + parse_defects:
        cls = row["class"]
        counts = summary.setdefault(cls, {"train": 0, "tuning": 0, "total": 0})
        counts[row["split"]] += 1
        counts["total"] += 1

    # Denominators: how many probes/tasks each class is counted over.
    denominators: dict[str, dict[str, int]] = {}

    def bump(cls: str, split: str, n: int = 1) -> None:
        counts = denominators.setdefault(cls, {"train": 0, "tuning": 0, "total": 0})
        counts[split] += n
        counts["total"] += n

    for parsed in parsed_tasks:
        for cls, spec in _CLASS_PROBES.items():
            if spec is None:
                kinds = ("same_rule", "transfer", "composition") if cls == "target_mismatch" else ("specificity",)
                for kind in kinds:
                    bump(cls, parsed.split, len(parsed.probes[kind]))
                continue
            family, kind = spec
            if parsed.family == family:
                bump(cls, parsed.split, len(parsed.probes[kind]))
        for cls, family in _TASK_FAMILIES.items():
            if family is None or parsed.family == family:
                bump(cls, parsed.split)

    distinct_questions: dict[str, set[tuple[str, str]]] = {"train": set(), "tuning": set()}
    for parsed in parsed_tasks:
        bucket = distinct_questions[parsed.split]
        for event in parsed.events:
            bucket.update((parsed.family, question) for question in event["questions"])
        for kind in ProbeKind:
            bucket.update((parsed.family, probe.question) for probe in parsed.probes[kind.value])
    for split_name, bucket in distinct_questions.items():
        bump("question_target_contradiction", split_name, len(bucket))

    for cls, counts in summary.items():
        counts["unit"] = _CLASS_UNITS.get(cls, "probes")
        counts["of"] = denominators.get(cls, {"train": 0, "tuning": 0, "total": 0})
    for cls, counts in denominators.items():
        summary.setdefault(cls, {"train": 0, "tuning": 0, "total": 0})
        summary[cls].setdefault("unit", _CLASS_UNITS.get(cls, "probes"))
        summary[cls].setdefault("of", counts)

    return {
        "summary": dict(sorted(summary.items())),
        "rows": rows,
        "parse_defects": parse_defects,
        "parsed_tasks": parsed_tasks,
    }


# ---------------------------------------------------------------------------
# Support-certificate verification (v2 only)
# ---------------------------------------------------------------------------


def verify_certificates(catalog: Any, bundle: Any) -> dict[str, Any]:
    by_key = {
        (entry.split, entry.family, entry.task_index): entry for entry in bundle.tasks
    }
    parsed_by_key: dict[tuple[str, str, int], ParsedTask] = {}
    counters: dict[tuple[str, str], int] = {}
    for split_name, tasks in (("meta_train", catalog.meta_train),
                              ("meta_validation", catalog.meta_validation)):
        for task in tasks:
            key = (split_name, task.family.value)
            index = counters.get(key, 0)
            counters[key] = index + 1
            parsed, _ = parse_task(split_name, index, task)
            parsed_by_key[(split_name, task.family.value, index)] = parsed

    results: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    for key, entry in sorted(by_key.items()):
        parsed = parsed_by_key.get(key)
        if parsed is None:
            results.append({"probe_id": key, "status": "missing_task"})
            continue
        corrections = {event["correction"] for event in parsed.events}
        for cert in entry.probes:
            probe = next(
                (p for kind in parsed.probes.values() for p in kind
                 if p.question == cert.question and p.kind == cert.kind),
                None,
            )
            row: dict[str, Any] = {"probe_id": cert.probe_id, "category": cert.category}
            if probe is None:
                row["status"] = "probe_not_found"
                results.append(row)
                continue
            ok = True
            if probe.expected != cert.expected_response:
                row["status"] = "certificate_target_differs_from_probe"
                results.append(row)
                continue
            missing = [f.correction for f in cert.required_facts if f.correction not in corrections]
            if missing:
                row["status"] = "required_teaching_fact_absent"
                row["missing"] = missing
                results.append(row)
                continue
            derived, reason = derive_target(parsed, probe)
            if cert.category == "pre_learning_baseline":
                # By design: a pre-learning baseline has no derivation to
                # verify, so it is reported as its own count
                # (``baseline_not_required``), never folded into ``verified``.
                row["status"] = "baseline_not_required"
                results.append(row)
                continue
            if derived is None:
                row["status"] = "independent_derivation_failed"
                row["reason"] = reason
                results.append(row)
                continue
            if derived != cert.expected_response:
                row["status"] = "derivation_disagrees"
                row["derived"] = derived
                results.append(row)
                continue
            if cert.minimal:
                for fact in cert.required_facts:
                    if not _counterexample(parsed, probe, [fact.key]):
                        row["status"] = "deleting_fact_keeps_derivation"
                        row["fact"] = fact.correction
                        ok = False
                        break
            if ok:
                for goal, expected in cert.paired_goal_counterfactual:
                    # A registered paired example changes the goal everywhere
                    # it is observable: the taught goal fact and the probe.
                    probe_q = probe.question
                    flipped_q = re.sub(r"Goal: reach (\w+)\.", f"Goal: reach {goal}.", probe_q)
                    flipped_facts = dict(parsed.facts)
                    flipped_facts[("goal",)] = goal
                    flipped_task = ParsedTask(
                        split=parsed.split, family=parsed.family, index=parsed.index,
                        events=parsed.events, facts=flipped_facts,
                        fact_kinds=dict(parsed.fact_kinds), probes=parsed.probes,
                        oracle=parsed.oracle, rule_fingerprint=parsed.rule_fingerprint,
                    )
                    counterfactual = ParsedProbe(
                        kind=probe.kind, question=flipped_q,
                        expected=expected, query=dict(probe.query),
                    )
                    flipped_match = re.fullmatch(
                        _QUESTIONS[("finite_state", "composition")][0].pattern, flipped_q
                    )
                    if flipped_match is None:
                        row["status"] = "goal_counterfactual_question_unparsed"
                        ok = False
                    else:
                        counterfactual.query["groups"] = flipped_match.groups()
                        derived_cf, reason_cf = derive_target(flipped_task, counterfactual)
                        if derived_cf != expected or expected == cert.expected_response:
                            row["status"] = "goal_counterfactual_not_goal_dependent"
                            row["reason"] = reason_cf
                            ok = False
            if ok:
                row["status"] = "verified"
            results.append(row)

            # Mutation checks: delete a required teaching fact or flip the
            # target and confirm admission fails for this certificate.
            # Over-determined teaching (``minimal`` False, e.g. a transformation
            # task with several teaching pairs) is checked at set level: its
            # whole required teaching set must not be removable.
            #
            # The two mutation families are NOT equally discriminating and are
            # counted separately (``kind``): the delete checks can genuinely
            # fail, while the flip-target check below cannot fail once the
            # certificate verifies — ``derive_target`` reads ``expected`` only
            # in its ``pre`` (pre-learning baseline) branch, and flip-target
            # rows are generated exclusively for non-``pre_learning_baseline``
            # certificates, so on every row here ``derived_flip`` is computed
            # without the flipped ``expected`` and ``derived_flip !=
            # flipped.expected`` holds for any certificate whose derivation
            # agrees with the recorded target (the real guarantee for target
            # fidelity is the independent derivation disagree-check above).
            # Were a ``pre`` probe ever certified as non-baseline, its
            # flip-target row would NOT be independent of ``expected``.
            # ``delete_fact`` covers both the per-fact deletes and the
            # whole-required-set delete used for over-determined teaching.
            if cert.category != "pre_learning_baseline":
                if cert.minimal:
                    for fact in cert.required_facts:
                        mutations.append({
                            "probe_id": cert.probe_id,
                            "kind": "delete_fact",
                            "mutation": f"delete fact {fact.correction!r}",
                            "detected": _counterexample(parsed, probe, [fact.key]),
                        })
                elif cert.required_facts:
                    mutations.append({
                        "probe_id": cert.probe_id,
                        "kind": "delete_fact",
                        "mutation": "delete the full required teaching set",
                        "detected": _counterexample(
                            parsed, probe, [f.key for f in cert.required_facts]
                        ),
                    })
                flipped = ParsedProbe(
                    kind=probe.kind, question=probe.question,
                    expected=probe.expected + "x", query=dict(probe.query),
                )
                derived_flip, _ = derive_target(parsed, flipped)
                mutations.append({
                    "probe_id": cert.probe_id,
                    "kind": "flip_target",
                    "mutation": "flip target",
                    "detected": derived_flip != flipped.expected,
                })

    # Verified vs baseline split: ``pre_learning_baseline`` certificates are
    # ``baseline_not_required`` by design (no derivation is claimed for them),
    # so folding them into ``verified`` would overstate demonstrated strength.
    verified = sum(1 for r in results if r["status"] == "verified")
    baseline = sum(1 for r in results if r["status"] == "baseline_not_required")
    failed_rows = [
        r for r in results if r["status"] not in ("verified", "baseline_not_required")
    ]
    checks_by_kind = Counter(m["kind"] for m in mutations)
    detected_by_kind = Counter(m["kind"] for m in mutations if m["detected"])
    return {
        "certificates": len(results),
        "verified": verified,
        "baseline_not_required": baseline,
        "failed": len(failed_rows),
        "failures": failed_rows,
        "mutation_checks": len(mutations),
        "mutations_detected": sum(1 for m in mutations if m["detected"]),
        "mutations_missed": [m for m in mutations if not m["detected"]],
        "mutation_checks_by_kind": dict(sorted(checks_by_kind.items())),
        "mutations_detected_by_kind": dict(sorted(detected_by_kind.items())),
    }


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def distributions(catalog: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, tasks in (("train", catalog.meta_train), ("tuning", catalog.meta_validation)):
        event_counts = Counter(len(t.events) for t in tasks)
        templates = Counter()
        branches = Counter()
        answer_lengths = Counter()
        support = Counter()
        for task in tasks:
            parsed, _ = parse_task(split_name, 0, task)
            if task.family.value == "rule_transformation":
                for event in parsed.events:
                    if event["kind"] == "transformation_pair":
                        operand = event["key"][0]
                        branches["vowel_start" if operand[0] in _VOWELS else "consonant_start"] += 1
            for kind in ProbeKind:
                for probe in task.probes.by_kind(kind):
                    answer_lengths[len(probe.expected_response.split())] += 1
                    support[kind.value] += 1
            templates[task.family.value] += 1
        out[split_name] = {
            "tasks": len(tasks),
            "event_counts": dict(sorted(event_counts.items())),
            "families": dict(sorted(templates.items())),
            "teaching_branches": dict(sorted(branches.items())),
            "answer_token_lengths": dict(sorted(answer_lengths.items())),
            "probes_by_kind": dict(sorted(support.items())),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_v1(config: TaskGeneratorConfig) -> Any:
    from oczy.experiments.meta_cortex.taskgen import build_dev_catalog

    return build_dev_catalog(config)


def _build_v2(config: TaskGeneratorConfig) -> tuple[Any, Any]:
    from oczy.experiments.meta_cortex.taskgen_v2 import build_dev_catalog_v2

    return build_dev_catalog_v2(config)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    config = TaskGeneratorConfig(
        root_seed=args.root_seed,
        train_tasks_per_family=args.train_tasks_per_family,
        validation_tasks_per_family=args.tuning_tasks_per_family,
    )
    v1 = _build_v1(config)
    v2, bundle = _build_v2(config)

    before = check_catalog(v1)
    after = check_catalog(v2)
    certificates = verify_certificates(v2, bundle)

    materialized = None
    if args.public_root is not None:
        from oczy.experiments.meta_cortex.instrument import load_dev_view

        loaded = load_dev_view(Path(args.public_root)).catalog
        materialized = {
            "public_root": str(args.public_root),
            "tasks": len(loaded.meta_train) + len(loaded.meta_validation),
            "digest_matches_in_memory_v1": loaded.catalog_sha256 == v1.catalog_sha256,
        }

    baseline_ok = True
    baseline_mismatches: list[dict[str, Any]] = []
    for cls, expected in EXPECTED_V1_DEFECTS.items():
        observed = before["summary"].get(cls, {"train": 0, "tuning": 0, "total": 0})
        for split_name in ("train", "tuning", "total"):
            if observed.get(split_name, 0) != expected[split_name]:
                baseline_ok = False
                baseline_mismatches.append({
                    "class": cls, "split": split_name,
                    "expected": expected[split_name], "observed": observed.get(split_name, 0),
                })
        of = observed.get("of", {"train": 0, "tuning": 0, "total": 0})
        for split_name in ("train", "tuning", "total"):
            if of.get(split_name, 0) != expected["of"][split_name]:
                baseline_ok = False
                baseline_mismatches.append({
                    "class": cls, "split": f"{split_name}.of",
                    "expected": expected["of"][split_name],
                    "observed": of.get(split_name, 0),
                })

    after_zero = all(
        counts["total"] == 0 for cls, counts in after["summary"].items()
    )
    report = {
        "schema": SCHEMA,
        "classification": "READ_ONLY_INSTRUMENT_AUDIT",
        "meta_test_accessed": False,
        "calibration_accessed": False,
        "sealed_accessed": False,
        "model_runs": 0,
        "scoring_or_instrument_changed": False,
        "generator_lineages": {
            "before": "oczy/meta-cortex/taskgen/v1-dev (unchanged)",
            "after": "oczy/meta-cortex/taskgen/v2-dev (new lineage)",
        },
        "config": {
            "root_seed": config.root_seed,
            "train_tasks_per_family": config.train_tasks_per_family,
            "validation_tasks_per_family": config.validation_tasks_per_family,
        },
        "catalog_sha256": {"before": v1.catalog_sha256, "after": v2.catalog_sha256},
        "materialized_instrument_cross_check": materialized,
        "before": {"summary": before["summary"], "row_count": len(before["rows"]),
                   "parse_defect_count": len(before["parse_defects"])},
        "after": {"summary": after["summary"], "row_count": len(after["rows"]),
                  "parse_defect_count": len(after["parse_defects"])},
        "certificates": certificates,
        "distributions": {"before": distributions(v1), "after": distributions(v2)},
        "baseline_reproduced": baseline_ok,
        "baseline_mismatches": baseline_mismatches,
        "after_all_defect_counts_zero": after_zero,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("dry-run", help="count defects before/after the repair")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--root-seed", type=int, default=20260709)
    run.add_argument("--train-tasks-per-family", type=int, default=30)
    run.add_argument("--tuning-tasks-per-family", type=int, default=5)
    run.add_argument("--public-root", type=Path, default=None,
                     help="optional materialized public instrument root to cross-check")
    args = parser.parse_args(argv)

    if args.command == "dry-run":
        report = dry_run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({
            "baseline_reproduced": report["baseline_reproduced"],
            "after_all_defect_counts_zero": report["after_all_defect_counts_zero"],
            "certificates": {
                k: report["certificates"][k]
                for k in ("certificates", "verified", "baseline_not_required",
                          "failed", "mutation_checks", "mutations_detected",
                          "mutation_checks_by_kind",
                          "mutations_detected_by_kind")
            },
            "before": report["before"]["summary"],
            "after": report["after"]["summary"],
        }, indent=2, sort_keys=True))
        if not report["baseline_reproduced"]:
            print("FAILED: v1 baseline counts differ from the recorded audit",
                  file=sys.stderr)
            return 1
        if not report["after_all_defect_counts_zero"]:
            print("FAILED: repaired generator still has defect counts",
                  file=sys.stderr)
            return 1
        if report["certificates"]["failed"] or report["certificates"]["mutations_missed"]:
            print("FAILED: support certificates do not hold", file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
