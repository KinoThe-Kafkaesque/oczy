"""Deterministic DEV task generation — lineage ``oczy/meta-cortex/taskgen/v2-dev``.

This module is the **task-support repair** of the public DEV generator
(``taskgen.py``, lineage ``oczy/meta-cortex/taskgen/v1-dev``).  It is a
separate lineage: the v1 generator, its rendered tasks, and all recorded v1
scores stay exactly as they are.  ``taskgen.py`` and ``contracts.py`` are
eval-guard protected; this module only reads them.

What this lineage repairs (evidence: ``experiments_logs/2026-09-12_dev_teaching_coverage.json``
and ``experiments_logs/2026-09-12_curriculum_eval_audit.md``):

1. **Untaught lookups.** v1 same-rule and transfer probes query
   context/symbol and state/signal pairs that were never taught.  v2 draws
   every such probe from the task's own teaching pairs.
2. **Undefined contextual composition.** v1 feeds a mapping *output* back as
   a second lookup key although outputs and symbols are disjoint vocabularies,
   then silently substitutes ``symbols[-1]``.  v2 draws outputs from the
   symbol vocabulary (typed chains) and teaches both lookups of the chain.
3. **Untaught FSM actions and absent goal.** v1 scores ``"<state> <action>"``
   while no action mapping and no goal is ever taught.  v2 teaches the final
   state's action and the goal, states the goal and the goal-terminal
   convention (``halt``) in the probe, and registers a paired counterfactual
   probe in which changing the goal changes the correct answer.
4. **Underdetermined transformation targets.** v1 can score a target that its
   teaching cannot determine (uncovered conditional branches).  v2 forces
   branch coverage in teaching and refuses to construct a task whose scored
   targets are not agreed on by every teaching-consistent rule of the public
   289-rule grammar.
5. **Contradictory specificity targets.** v1 scores a hidden category flip:
   the same observable question carries one target in teaching and a
   different one in the specificity probe.  v2 gives every specificity probe
   an explicit unchanged scope stated in the question, and no observable
   question may carry two different targets anywhere in the task.
6. **Silently duplicated teaching events.** v2 samples teaching facts without
   replacement; the declared event count always equals the number of distinct
   teaching records.

Every scored probe carries an offline support certificate
(``SupportBundle``) naming the exact teaching facts and derivation that
determine its answer.  The bundle is audit metadata: it is never passed to a
model-facing method, never rendered into teaching or probe text, and never
supplied to the learner.  A separately implemented checker
(``scripts/r20_task_support_check.py``) re-derives every target from rendered
teaching text only and rejects the task when a required fact is deleted or a
target is flipped.

Determinism matches the v1 contract: counter-mode SHA-256 over
``TASKGEN_SCHEMA_V2 | root_seed | split | family | task_index |
collision_nonce | counter``, rejection-sampled, no ``random``, no ``hash()``,
no timestamps, no file I/O, no model output.  Split is assigned before
surface rendering; the v1 split firewall API is reused unchanged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    DevSplit,
    DevTaskCatalog,
    DialogueMessage,
    LearningEvent,
    MetaTask,
    OutcomeCode,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from .taskgen import (
    _CONTEXTS,
    _INPUTS,
    _OPERANDS,
    _STATES,
    _SYMBOLS,
    MAX_COLLISION_NONCE,
    SplitFirewallError,
    assert_split_firewall,
    audit_split_firewall,
)

__all__ = [
    "TASKGEN_SCHEMA_V2",
    "SupportFact",
    "ProbeSupport",
    "TaskSupport",
    "SupportBundle",
    "build_dev_catalog_v2",
    "transformation_grammar",
    "SupportError",
]

TASKGEN_SCHEMA_V2 = "oczy/meta-cortex/taskgen/v2-dev"

# Goal-terminal decision token.  Deliberately disjoint from the state-action
# pool so that the correct composition answer strictly depends on the goal.
GOAL_TERMINAL_ACTION = "halt"
# Same action vocabulary as the v1 family-C rule (taskgen._ACTIONS); the goal
# terminal token is not a state action.
_STATE_ACTIONS = ("proceed", "wait", "yield", "hold")

_VOWELS = "aeiou"


class SupportError(ValueError):
    """Raised when a task cannot be constructed with full teaching support."""


# ---------------------------------------------------------------------------
# Deterministic RNG for the v2 lineage
# ---------------------------------------------------------------------------


class _HashStream:
    """Counter-mode SHA-256 stream bound to ``TASKGEN_SCHEMA_V2``."""

    __slots__ = ("_base", "_counter", "_buffer", "_pos")

    def __init__(
        self,
        root_seed: int,
        split: DevSplit,
        family: TaskFamily,
        task_index: int,
        collision_nonce: int = 0,
    ) -> None:
        self._base = "|".join(
            [
                TASKGEN_SCHEMA_V2,
                str(root_seed),
                split.value,
                family.value,
                str(task_index),
                str(collision_nonce),
            ]
        ).encode("utf-8")
        self._counter = 0
        self._buffer = b""
        self._pos = 0

    def _refill(self) -> None:
        self._buffer = hashlib.sha256(
            self._base + b"|" + str(self._counter).encode("utf-8")
        ).digest()
        self._counter += 1
        self._pos = 0

    def _read_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            if self._pos >= len(self._buffer):
                self._refill()
            need = min(n - len(out), len(self._buffer) - self._pos)
            out.extend(self._buffer[self._pos : self._pos + need])
            self._pos += need
        return bytes(out)

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound == 1:
            return 0
        bit_len = bound.bit_length()
        byte_len = (bit_len + 7) // 8
        while True:
            val = int.from_bytes(self._read_bytes(byte_len), "big") & (
                (1 << bit_len) - 1
            )
            if val < bound:
                return val

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.randbelow(hi - lo + 1)

    def choice(self, seq: Sequence[Any]) -> Any:
        return seq[self.randbelow(len(seq))]

    def sample(self, seq: Sequence[Any], k: int) -> list[Any]:
        n = len(seq)
        if k < 0 or k > n:
            raise ValueError(f"cannot sample {k} from {n} elements")
        pool = list(seq)
        for i in range(k):
            j = i + self.randbelow(n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


# ---------------------------------------------------------------------------
# Support certificates (offline audit metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportFact:
    """One teaching fact, bound to the rendered correction that carries it."""

    kind: str
    key: tuple[str, ...]
    target: str
    correction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": list(self.key),
            "target": self.target,
            "correction": self.correction,
        }


@dataclass(frozen=True, slots=True)
class ProbeSupport:
    """Offline support certificate for one probe."""

    probe_id: str
    kind: str
    category: str  # "scored" | "pre_learning_baseline" | "explicit_scope"
    question: str
    expected_response: str
    required_facts: tuple[SupportFact, ...]
    derivation: str
    minimal: bool
    paired_goal_counterfactual: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind,
            "category": self.category,
            "question": self.question,
            "expected_response": self.expected_response,
            "required_facts": [f.as_dict() for f in self.required_facts],
            "derivation": self.derivation,
            "minimal": self.minimal,
            "paired_goal_counterfactual": [
                {"goal": g, "expected_response": e}
                for g, e in self.paired_goal_counterfactual
            ],
        }


@dataclass(frozen=True, slots=True)
class TaskSupport:
    split: str
    family: str
    task_index: int
    rule_fingerprint: str
    facts: tuple[SupportFact, ...]
    probes: tuple[ProbeSupport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "family": self.family,
            "task_index": self.task_index,
            "rule_fingerprint": self.rule_fingerprint,
            "facts": [f.as_dict() for f in self.facts],
            "probes": [p.as_dict() for p in self.probes],
        }


@dataclass(frozen=True, slots=True)
class SupportBundle:
    lineage: str
    catalog_sha256: str
    tasks: tuple[TaskSupport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "oczy/meta-cortex/task-support/v2-dev",
            "lineage": self.lineage,
            "catalog_sha256": self.catalog_sha256,
            "tasks": [t.as_dict() for t in self.tasks],
        }


# ---------------------------------------------------------------------------
# Teaching sentence renderers (the only place a fact becomes text)
# ---------------------------------------------------------------------------


def _contextual_correction(ctx: str, sym: str, out: str) -> str:
    return f"In the {ctx} room, {sym} requires the token {out}."


def _transformation_correction(op: str, result: str) -> str:
    return f"The correct result for {op} is {result}."


def _fsm_edge_correction(state: str, inp: str, nxt: str) -> str:
    return f"From {state}, input {inp} transitions to {nxt}."


def _fsm_action_correction(state: str, action: str) -> str:
    return f"In state {state}, the action is {action}."


def _fsm_goal_correction(goal: str) -> str:
    return f"The goal is to reach state {goal}."


def _event(obs_content: str, attempted: str, correction: str) -> LearningEvent:
    return LearningEvent(
        observation_messages=(DialogueMessage(role="user", content=obs_content),),
        attempted_behavior=attempted,
        correction=correction,
        outcome=OutcomeCode.CORRECTED,
    )


# ---------------------------------------------------------------------------
# Family A: contextual remapping (typed chains)
# ---------------------------------------------------------------------------


def _build_contextual_remapping_v2(
    stream: _HashStream, config: TaskGeneratorConfig
) -> tuple[MetaTask, tuple[SupportFact, ...], tuple[ProbeSupport, ...]]:
    n_contexts = stream.randint(2, 3)
    n_symbols = stream.randint(2, 3)
    contexts = stream.sample(_CONTEXTS, n_contexts)
    symbols = stream.sample(_SYMBOLS, n_symbols)
    # Typed chain: outputs are drawn from the symbol vocabulary so that a
    # first lookup's output is a legal second lookup key (v1 drew outputs from
    # the disjoint ``_OUTPUTS`` vocabulary and then silently substituted
    # ``symbols[-1]`` for the undefined second input).
    outputs = stream.sample(symbols, n_symbols)
    rule: dict[str, dict[str, str]] = {
        ctx: {sym: stream.choice(outputs) for sym in symbols} for ctx in contexts
    }

    # Composition chain: both lookups must be taught and typed.
    first_ctx, second_ctx = contexts[0], contexts[-1]
    first_sym = stream.choice(symbols)
    first_out = rule[first_ctx][first_sym]
    second_out = rule[second_ctx][first_out]
    if first_out not in rule[second_ctx]:
        raise SupportError("contextual chain is not typed: second key undefined")

    required_pairs = [(first_ctx, first_sym), (second_ctx, first_out)]
    population = [(ctx, sym) for ctx in contexts for sym in symbols]
    available = [p for p in population if p not in required_pairs]
    low = max(config.min_events, len(required_pairs))
    high = min(config.max_events, len(population))
    if low > high:
        raise SupportError("not enough distinct contextual pairs for teaching support")
    n_events = stream.randint(low, high)
    taught: list[tuple[str, str]] = required_pairs + stream.sample(
        available, n_events - len(required_pairs)
    )

    rule_spec = {
        "contexts": sorted(contexts),
        "symbols": sorted(symbols),
        "outputs": sorted(set(outputs)),
        "cells": {ctx: dict(sorted(rule[ctx].items())) for ctx in sorted(rule)},
    }
    composition_spec = {
        "first_context": first_ctx,
        "first_symbol": first_sym,
        "first_output": first_out,
        "second_context": second_ctx,
        "second_input": first_out,
        "second_output": second_out,
        "typed_chain": True,
    }
    paraphrase_group = {
        "family": TaskFamily.CONTEXTUAL_REMAP.value,
        "contexts": sorted(contexts),
        "symbols": sorted(symbols),
    }

    events: list[LearningEvent] = []
    facts: list[SupportFact] = []
    for ctx, sym in taught:
        out = rule[ctx][sym]
        correction = _contextual_correction(ctx, sym, out)
        events.append(_event(f"In the {ctx} room, respond to {sym}.", sym, correction))
        facts.append(
            SupportFact(
                kind="contextual_lookup",
                key=(ctx, sym),
                target=out,
                correction=correction,
            )
        )

    def fact_for(pair: tuple[str, str]) -> SupportFact:
        return next(f for f in facts if f.key == pair)

    probes: list[ProbeSupport] = []
    cases: dict[str, list[ProbeCase]] = {k: [] for k in ("pre", "same_rule", "transfer", "composition", "specificity", "oracle_context")}

    def probe_id(kind: str) -> str:
        index = sum(1 for p in probes if p.kind == kind)
        return f"contextual_remap/{kind}/{index}"

    # Pre-learning baseline: unanswerable before teaching by design.
    for ctx, sym in ((contexts[0], symbols[0]), (contexts[-1], symbols[-1])):
        question = f"The room is {ctx}. What response follows {sym}?"
        cases["pre"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=rule[ctx][sym],
                kind=ProbeKind.PRE,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("pre"),
                kind="pre",
                category="pre_learning_baseline",
                question=question,
                expected_response=rule[ctx][sym],
                required_facts=(),
                derivation="pre-learning baseline: target is the hidden rule value; support is not required before teaching",
                minimal=False,
            )
        )

    # Same-rule: paraphrase taught mappings only.
    for ctx, sym in taught[:3]:
        question = f"You are in the {ctx} chamber. {sym} demands what token?"
        fact = fact_for((ctx, sym))
        cases["same_rule"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=fact.target,
                kind=ProbeKind.SAME_RULE,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("same_rule"),
                kind="same_rule",
                category="scored",
                question=question,
                expected_response=fact.target,
                required_facts=(fact,),
                derivation=f"lookup taught pair {fact.key} -> {fact.target}",
                minimal=True,
            )
        )

    # Transfer: novel phrasing over taught mappings only.
    transfer_prompts = (
        "Setting: {ctx} environment. Command word: {sym}. What is the correct response?",
        "Within the {ctx} domain, the signal {sym} elicits what?",
    )
    transfer_pairs = [taught[-1], taught[0]][: max(1, min(2, len(taught)))]
    for template, (ctx, sym) in zip(transfer_prompts, transfer_pairs, strict=False):
        question = template.format(ctx=ctx, sym=sym)
        fact = fact_for((ctx, sym))
        cases["transfer"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=fact.target,
                kind=ProbeKind.TRANSFER,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("transfer"),
                kind="transfer",
                category="scored",
                question=question,
                expected_response=fact.target,
                required_facts=(fact,),
                derivation=f"lookup taught pair {fact.key} -> {fact.target}",
                minimal=True,
            )
        )

    # Composition: typed chain over two taught lookups.
    question = (
        f"First, in the {first_ctx} room, respond to {first_sym}. "
        f"Then, in the {second_ctx} room, respond to {first_out}. "
        "Give both tokens in order."
    )
    expected = f"{first_out} then {second_out}"
    cases["composition"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=question),),
            expected_response=expected,
            kind=ProbeKind.COMPOSITION,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("composition"),
            kind="composition",
            category="scored",
            question=question,
            expected_response=expected,
            required_facts=(fact_for((first_ctx, first_sym)), fact_for((second_ctx, first_out))),
            derivation=(
                f"chain lookup 1: ({first_ctx}, {first_sym}) -> {first_out}; "
                f"chain lookup 2: ({second_ctx}, {first_out}) -> {second_out}"
            ),
            minimal=True,
        )
    )

    # Specificity: explicit unchanged scope, never a hidden category flip.
    unused = [c for c in _CONTEXTS if c not in contexts]
    spec_ctx = stream.choice(unused)
    spec_sym = stream.choice(_SYMBOLS)
    question = (
        f"Unrelated rule for the {spec_ctx} room: {spec_sym} answers with {spec_sym}. "
        f"What response follows {spec_sym}?"
    )
    cases["specificity"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=question),),
            expected_response=spec_sym,
            kind=ProbeKind.SPECIFICITY,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("specificity"),
            kind="specificity",
            category="explicit_scope",
            question=question,
            expected_response=spec_sym,
            required_facts=(),
            derivation="unchanged scope stated verbatim in the question; unrelated to the taught rule",
            minimal=False,
        )
    )

    # Oracle context: complete mapping stated in the probe itself.
    lines = ["Complete mapping:"]
    for ctx in sorted(contexts):
        for sym in sorted(symbols):
            lines.append(f"  {ctx} / {sym} -> {rule[ctx][sym]}")
    mapping_text = "\n".join(lines)
    oracle_question = (
        f"Given the above mapping, in the {first_ctx} room, what token follows {first_sym}?"
    )
    cases["oracle_context"].append(
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=mapping_text),
                DialogueMessage(role="user", content=oracle_question),
            ),
            expected_response=rule[first_ctx][first_sym],
            kind=ProbeKind.ORACLE_CONTEXT,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("oracle_context"),
            kind="oracle_context",
            category="explicit_scope",
            question=oracle_question,
            expected_response=rule[first_ctx][first_sym],
            required_facts=(),
            derivation="complete mapping is stated inside the probe",
            minimal=False,
        )
    )

    battery = ProbeBattery(
        pre=tuple(cases["pre"]),
        same_rule=tuple(cases["same_rule"]),
        transfer=tuple(cases["transfer"]),
        composition=tuple(cases["composition"]),
        specificity=tuple(cases["specificity"]),
        oracle_context=tuple(cases["oracle_context"]),
    )
    task = MetaTask(
        family=TaskFamily.CONTEXTUAL_REMAP,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=battery,
        rule_fingerprint=_canonical_sha256(rule_spec),
        assignment_fingerprint=_canonical_sha256(rule),
        composition_fingerprint=_canonical_sha256(composition_spec),
        paraphrase_group_fingerprint=_canonical_sha256(paraphrase_group),
    )
    return task, tuple(facts), tuple(probes)


# ---------------------------------------------------------------------------
# Family B: rule transformation (branch-covered, identifiable)
# ---------------------------------------------------------------------------

_RULE_TEMPLATES = ("permutation", "substitution", "conditional", "composition")


def transformation_grammar() -> tuple[tuple[str, Callable[[str], str]], ...]:
    """The complete public v1 transformation grammar (289 rules).

    Independent enumerations live in ``scripts/r20_task_support_check.py`` and
    ``scripts/audit_curricula_evals.py``; this copy exists so construction can
    refuse underdetermined targets before any model sees them.
    """
    candidates: list[tuple[str, Callable[[str], str]]] = [("reverse", lambda x: x[::-1])]
    for token in _SYMBOLS:
        candidates.append(
            (
                f"substitute:{token}",
                lambda x, t=token: "".join(t if c in _VOWELS else c for c in x),
            )
        )
        candidates.append(
            (
                f"reverse_substitute:{token}",
                lambda x, t=token: "".join(t if c in _VOWELS else c for c in x[::-1]),
            )
        )
    for suffix, prefix in itertools.product(_SYMBOLS, repeat=2):
        candidates.append(
            (
                f"conditional:{suffix}:{prefix}",
                lambda x, s=suffix, p=prefix: (
                    x + s if x and x[0] in _VOWELS else p + x
                ),
            )
        )
    return tuple(candidates)


def _build_rule_transformation_v2(
    stream: _HashStream, config: TaskGeneratorConfig
) -> tuple[MetaTask, tuple[SupportFact, ...], tuple[ProbeSupport, ...]]:
    template = stream.choice(_RULE_TEMPLATES)
    n_teach = stream.randint(config.min_events, config.max_events)
    n_operands = n_teach + 3

    vowel_operands = [o for o in _OPERANDS if o[0] in _VOWELS]
    consonant_operands = [o for o in _OPERANDS if o[0] not in _VOWELS]

    if template == "conditional":
        # Branch coverage: teaching must reach both branches the scored
        # probes can exercise, or the suffix/prefix are not determined.
        if n_teach < 2:
            raise SupportError("conditional tasks need >= 2 teaching events")
        branch_vowel = stream.choice(vowel_operands)
        branch_consonant = stream.choice(consonant_operands)
        rest = stream.sample(
            [o for o in _OPERANDS if o not in (branch_vowel, branch_consonant)],
            n_operands - 2,
        )
        operands = [branch_vowel, branch_consonant] + rest
    else:
        operands = stream.sample(_OPERANDS, n_operands)

    teaching_operands = operands[:n_teach]
    held_out_operands = operands[n_teach:]

    if template == "permutation":

        def apply_rule(op: str) -> str:
            return op[::-1]

        rule_spec = {"template": "permutation", "param1": "reverse", "param2": ""}
    elif template == "substitution":
        sub_char = stream.choice(_SYMBOLS)

        def apply_rule(op: str) -> str:
            return "".join(sub_char if c in _VOWELS else c for c in op)

        rule_spec = {"template": "substitution", "param1": sub_char, "param2": ""}
    elif template == "conditional":
        suffix = stream.choice(_SYMBOLS)
        prefix = stream.choice(_SYMBOLS)

        def apply_rule(op: str) -> str:
            if op and op[0] in _VOWELS:
                return op + suffix
            return prefix + op

        rule_spec = {"template": "conditional", "param1": suffix, "param2": prefix}
    else:
        sub_char = stream.choice(_SYMBOLS)

        def apply_rule(op: str) -> str:
            reversed_op = op[::-1]
            return "".join(sub_char if c in _VOWELS else c for c in reversed_op)

        rule_spec = {"template": "composition", "param1": "reverse", "param2": sub_char}

    assignment = {op: apply_rule(op) for op in operands}
    composition_spec = {
        "template": template,
        "first_primitive": rule_spec["param1"],
        "second_primitive": rule_spec["param2"],
        "composition_order": "first_then_second",
    }
    paraphrase_group = {
        "family": TaskFamily.RULE_TRANSFORMATION.value,
        "template": template,
        "operands": sorted(operands),
    }

    events: list[LearningEvent] = []
    facts: list[SupportFact] = []
    for op in teaching_operands:
        result = apply_rule(op)
        correction = _transformation_correction(op, result)
        events.append(_event(f"Apply the rule to: {op}", op, correction))
        facts.append(
            SupportFact(
                kind="transformation_pair",
                key=(op,),
                target=result,
                correction=correction,
            )
        )

    def check_identifiable(
        question: str, operand: str, expected: str, *, apply_twice: bool = False
    ) -> None:
        taught_pairs = {f.key[0]: f.target for f in facts}
        outputs = set()
        for _name, fn in transformation_grammar():
            if not all(fn(k) == v for k, v in taught_pairs.items()):
                continue
            value = fn(fn(operand)) if apply_twice else fn(operand)
            outputs.add(value)
        if outputs != {expected}:
            raise SupportError(
                f"transformation target not determined by teaching: {question!r} "
                f"expected {expected!r} possible {sorted(outputs)}"
            )

    probes: list[ProbeSupport] = []
    cases: dict[str, list[ProbeCase]] = {k: [] for k in ("pre", "same_rule", "transfer", "composition", "specificity", "oracle_context")}
    all_facts = tuple(facts)

    def probe_id(kind: str) -> str:
        index = sum(1 for p in probes if p.kind == kind)
        return f"rule_transformation/{kind}/{index}"

    for op in held_out_operands[:2]:
        question = f"Apply the rule to: {op}"
        expected = apply_rule(op)
        cases["pre"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=expected,
                kind=ProbeKind.PRE,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("pre"),
                kind="pre",
                category="pre_learning_baseline",
                question=question,
                expected_response=expected,
                required_facts=(),
                derivation="pre-learning baseline on a held-out operand",
                minimal=False,
            )
        )

    for op in teaching_operands[:2]:
        question = f"What is the transformed output for input {op}?"
        expected = apply_rule(op)
        check_identifiable(question, op, expected)
        cases["same_rule"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=expected,
                kind=ProbeKind.SAME_RULE,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("same_rule"),
                kind="same_rule",
                category="scored",
                question=question,
                expected_response=expected,
                required_facts=all_facts,
                derivation="unique teaching-consistent rule of the public grammar applied to the taught operand",
                minimal=False,
            )
        )

    for op in held_out_operands[:2]:
        question = f"Transform: {op}"
        expected = apply_rule(op)
        check_identifiable(question, op, expected)
        cases["transfer"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=expected,
                kind=ProbeKind.TRANSFER,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("transfer"),
                kind="transfer",
                category="scored",
                question=question,
                expected_response=expected,
                required_facts=all_facts,
                derivation="unique teaching-consistent rule applied twice-free to a held-out operand",
                minimal=False,
            )
        )

    if held_out_operands:
        op = held_out_operands[0]
        question = f"Apply the rule twice to: {op}"
        expected = apply_rule(apply_rule(op))
        check_identifiable(question, op, apply_rule(op))
        check_identifiable(question, op, expected, apply_twice=True)
        cases["composition"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=expected,
                kind=ProbeKind.COMPOSITION,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("composition"),
                kind="composition",
                category="scored",
                question=question,
                expected_response=expected,
                required_facts=all_facts,
                derivation="unique teaching-consistent rule applied twice to a held-out operand",
                minimal=False,
            )
        )

    # Specificity: explicit unchanged scope instead of a hidden category flip.
    spec_operand = stream.choice([o for o in _OPERANDS if o not in operands])
    question = (
        "Unrelated rule: the input is returned unchanged. "
        f"Apply the unrelated rule to: {spec_operand}"
    )
    cases["specificity"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=question),),
            expected_response=spec_operand,
            kind=ProbeKind.SPECIFICITY,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("specificity"),
            kind="specificity",
            category="explicit_scope",
            question=question,
            expected_response=spec_operand,
            required_facts=(),
            derivation="unchanged scope (identity) stated verbatim in the question",
            minimal=False,
        )
    )

    examples = [f"  {op} -> {apply_rule(op)}" for op in operands[:3]]
    rule_text = (
        f"Rule: {rule_spec['template']} with parameters "
        f"{rule_spec['param1']!r} and {rule_spec['param2']!r}.\n"
        "Worked examples:\n" + "\n".join(examples)
    )
    oracle_op = operands[-1]
    oracle_question = f"Given this rule, what is the output for: {oracle_op}?"
    cases["oracle_context"].append(
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=rule_text),
                DialogueMessage(role="user", content=oracle_question),
            ),
            expected_response=apply_rule(oracle_op),
            kind=ProbeKind.ORACLE_CONTEXT,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("oracle_context"),
            kind="oracle_context",
            category="explicit_scope",
            question=oracle_question,
            expected_response=apply_rule(oracle_op),
            required_facts=(),
            derivation="rule and worked examples are stated inside the probe",
            minimal=False,
        )
    )

    battery = ProbeBattery(
        pre=tuple(cases["pre"]),
        same_rule=tuple(cases["same_rule"]),
        transfer=tuple(cases["transfer"]),
        composition=tuple(cases["composition"]),
        specificity=tuple(cases["specificity"]),
        oracle_context=tuple(cases["oracle_context"]),
    )
    task = MetaTask(
        family=TaskFamily.RULE_TRANSFORMATION,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=battery,
        rule_fingerprint=_canonical_sha256(rule_spec),
        assignment_fingerprint=_canonical_sha256(assignment),
        composition_fingerprint=_canonical_sha256(composition_spec),
        paraphrase_group_fingerprint=_canonical_sha256(paraphrase_group),
    )
    return task, all_facts, tuple(probes)


# ---------------------------------------------------------------------------
# Family C: finite-state behavior (taught actions and goal)
# ---------------------------------------------------------------------------


def _build_finite_state_v2(
    stream: _HashStream, config: TaskGeneratorConfig
) -> tuple[MetaTask, tuple[SupportFact, ...], tuple[ProbeSupport, ...]]:
    states = list(_STATES)
    inputs = list(_INPUTS)
    graph: dict[str, dict[str, str]] = {
        s: {inp: stream.choice(states) for inp in inputs} for s in states
    }
    action_map: dict[str, str] = {
        s: stream.choice(_STATE_ACTIONS) for s in states
    }
    start_state = states[0]

    # Required path for the composition probe: two distinct taught edges.
    first_input = stream.choice(inputs)
    mid_state = graph[start_state][first_input]
    second_choices = [
        inp for inp in inputs if (mid_state, inp) != (start_state, first_input)
    ]
    if not second_choices:
        raise SupportError("cannot build a two-edge path from distinct taught edges")
    second_input = stream.choice(second_choices)
    final_state = graph[mid_state][second_input]

    # Goal strictly different from the final state so the scored probe
    # exercises the taught action; the paired counterfactual flips the goal to
    # the final state and the correct decision changes to GOAL_TERMINAL_ACTION.
    goal_state = stream.choice([s for s in states if s != final_state])
    counterfactual_goal = final_state

    rule_spec = {
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": {
            s: {inp: graph[s][inp] for inp in sorted(inputs)} for s in sorted(states)
        },
        "actions": {s: action_map[s] for s in sorted(states)},
        "start_state": start_state,
        "goal_state": goal_state,
        "goal_terminal_action": GOAL_TERMINAL_ACTION,
    }
    composition_spec = {
        "start": start_state,
        "input_1": first_input,
        "mid_state": mid_state,
        "input_2": second_input,
        "final_state": final_state,
        "final_action": action_map[final_state],
        "goal_state": goal_state,
    }
    paraphrase_group = {
        "family": TaskFamily.FINITE_STATE.value,
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": rule_spec["transitions"],
        "actions": rule_spec["actions"],
        "goal_state": goal_state,
    }

    required_edges = [
        (start_state, first_input, mid_state),
        (mid_state, second_input, final_state),
    ]
    required_event_specs: list[tuple[str, str, str]] = [
        ("edge", start_state, first_input),
        ("edge", mid_state, second_input),
        ("action", final_state, action_map[final_state]),
        ("goal", goal_state, goal_state),
    ]
    remaining_edges = [
        (s, inp, graph[s][inp])
        for s in states
        for inp in inputs
        if (s, inp) not in {(e[0], e[1]) for e in required_edges}
    ]
    low = max(config.min_events, len(required_event_specs))
    high = min(config.max_events, len(required_event_specs) + len(remaining_edges))
    if low > high:
        raise SupportError("cannot satisfy FSM teaching support within the event budget")
    n_events = stream.randint(low, high)
    extra_edges = stream.sample(remaining_edges, n_events - len(required_event_specs))

    events: list[LearningEvent] = []
    facts: list[SupportFact] = []
    for s, inp, nxt in required_edges:
        correction = _fsm_edge_correction(s, inp, nxt)
        events.append(
            _event(f"State: {s}. Input: {inp}. What is the next state?", s, correction)
        )
        facts.append(
            SupportFact(kind="fsm_edge", key=(s, inp), target=nxt, correction=correction)
        )
    action_correction = _fsm_action_correction(final_state, action_map[final_state])
    events.append(
        _event(
            f"State: {final_state}. Which action is required?",
            final_state,
            action_correction,
        )
    )
    facts.append(
        SupportFact(
            kind="fsm_action",
            key=(final_state,),
            target=action_map[final_state],
            correction=action_correction,
        )
    )
    goal_correction = _fsm_goal_correction(goal_state)
    events.append(
        _event("What is the goal state?", goal_state, goal_correction)
    )
    facts.append(
        SupportFact(
            kind="fsm_goal", key=("goal",), target=goal_state, correction=goal_correction
        )
    )
    for s, inp, nxt in extra_edges:
        correction = _fsm_edge_correction(s, inp, nxt)
        events.append(
            _event(f"State: {s}. Input: {inp}. What is the next state?", s, correction)
        )
        facts.append(
            SupportFact(kind="fsm_edge", key=(s, inp), target=nxt, correction=correction)
        )

    def fact_of(kind: str, key: tuple[str, ...]) -> SupportFact:
        return next(f for f in facts if f.kind == kind and f.key == key)

    def composition_expected(goal: str) -> str:
        action = GOAL_TERMINAL_ACTION if final_state == goal else action_map[final_state]
        return f"{final_state} {action}"

    probes: list[ProbeSupport] = []
    cases: dict[str, list[ProbeCase]] = {k: [] for k in ("pre", "same_rule", "transfer", "composition", "specificity", "oracle_context")}

    def probe_id(kind: str) -> str:
        index = sum(1 for p in probes if p.kind == kind)
        return f"finite_state/{kind}/{index}"

    pre_question = f"State: {start_state}. Input: {first_input}. What is the next state?"
    cases["pre"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=pre_question),),
            expected_response=mid_state,
            kind=ProbeKind.PRE,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("pre"),
            kind="pre",
            category="pre_learning_baseline",
            question=pre_question,
            expected_response=mid_state,
            required_facts=(),
            derivation="pre-learning baseline on the first path edge",
            minimal=False,
        )
    )

    for s, inp, nxt in required_edges:
        question = f"Given current state {s} and signal {inp}, which state follows?"
        cases["same_rule"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=nxt,
                kind=ProbeKind.SAME_RULE,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("same_rule"),
                kind="same_rule",
                category="scored",
                question=question,
                expected_response=nxt,
                required_facts=(fact_of("fsm_edge", (s, inp)),),
                derivation=f"taught transition ({s}, {inp}) -> {nxt}",
                minimal=True,
            )
        )

    for s, inp, nxt in required_edges:
        question = f"State: {s}. Signal: {inp}. Next state?"
        cases["transfer"].append(
            ProbeCase(
                messages=(DialogueMessage(role="user", content=question),),
                expected_response=nxt,
                kind=ProbeKind.TRANSFER,
            )
        )
        probes.append(
            ProbeSupport(
                probe_id=probe_id("transfer"),
                kind="transfer",
                category="scored",
                question=question,
                expected_response=nxt,
                required_facts=(fact_of("fsm_edge", (s, inp)),),
                derivation=f"taught transition ({s}, {inp}) -> {nxt} under novel phrasing",
                minimal=True,
            )
        )

    comp_question = (
        f"Start at {start_state}. First input: {first_input}. Then input: {second_input}. "
        f"Goal: reach {goal_state}. Report the final state and its action, using "
        f"{GOAL_TERMINAL_ACTION} for the action when the goal is reached."
    )
    expected = composition_expected(goal_state)
    cases["composition"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=comp_question),),
            expected_response=expected,
            kind=ProbeKind.COMPOSITION,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("composition"),
            kind="composition",
            category="scored",
            question=comp_question,
            expected_response=expected,
            required_facts=(
                fact_of("fsm_edge", (start_state, first_input)),
                fact_of("fsm_edge", (mid_state, second_input)),
                fact_of("fsm_action", (final_state,)),
                fact_of("fsm_goal", ("goal",)),
            ),
            derivation=(
                f"path ({start_state},{first_input})->{mid_state}->({mid_state},{second_input})->{final_state}; "
                f"taught action for {final_state}; goal {goal_state} selects "
                f"{GOAL_TERMINAL_ACTION if final_state == goal_state else action_map[final_state]}"
            ),
            minimal=True,
            paired_goal_counterfactual=(
                (counterfactual_goal, composition_expected(counterfactual_goal)),
            ),
        )
    )

    spec_question = (
        f"Unrelated machine: from state {states[-1]}, input {inputs[-1]} transitions to "
        f"{states[0]}. State: {states[-1]}. Signal: {inputs[-1]}. Next state?"
    )
    cases["specificity"].append(
        ProbeCase(
            messages=(DialogueMessage(role="user", content=spec_question),),
            expected_response=states[0],
            kind=ProbeKind.SPECIFICITY,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("specificity"),
            kind="specificity",
            category="explicit_scope",
            question=spec_question,
            expected_response=states[0],
            required_facts=(),
            derivation="unrelated machine's transition stated verbatim in the question",
            minimal=False,
        )
    )

    lines = ["Complete transition graph:"]
    for s in sorted(graph):
        for inp in sorted(graph[s]):
            lines.append(f"  {s} + {inp} -> {graph[s][inp]}")
    lines.append("Actions:")
    for s in sorted(action_map):
        lines.append(f"  {s}: {action_map[s]}")
    lines.append(f"Goal: reach {goal_state}.")
    graph_text = "\n".join(lines)
    oracle_question = (
        f"Given the above graph, from {start_state} with input {first_input}, "
        "what is the next state?"
    )
    cases["oracle_context"].append(
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=graph_text),
                DialogueMessage(role="user", content=oracle_question),
            ),
            expected_response=graph[start_state][first_input],
            kind=ProbeKind.ORACLE_CONTEXT,
        )
    )
    probes.append(
        ProbeSupport(
            probe_id=probe_id("oracle_context"),
            kind="oracle_context",
            category="explicit_scope",
            question=oracle_question,
            expected_response=graph[start_state][first_input],
            required_facts=(),
            derivation="complete graph, actions and goal are stated inside the probe",
            minimal=False,
        )
    )

    battery = ProbeBattery(
        pre=tuple(cases["pre"]),
        same_rule=tuple(cases["same_rule"]),
        transfer=tuple(cases["transfer"]),
        composition=tuple(cases["composition"]),
        specificity=tuple(cases["specificity"]),
        oracle_context=tuple(cases["oracle_context"]),
    )
    task = MetaTask(
        family=TaskFamily.FINITE_STATE,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=battery,
        rule_fingerprint=_canonical_sha256(rule_spec),
        assignment_fingerprint=_canonical_sha256(rule_spec["actions"]),
        composition_fingerprint=_canonical_sha256(composition_spec),
        paraphrase_group_fingerprint=_canonical_sha256(paraphrase_group),
    )
    return task, tuple(facts), tuple(probes)


# ---------------------------------------------------------------------------
# Task assembly, split firewall, public catalog builder
# ---------------------------------------------------------------------------

_FAMILY_ORDER = (
    TaskFamily.CONTEXTUAL_REMAP,
    TaskFamily.RULE_TRANSFORMATION,
    TaskFamily.FINITE_STATE,
)

_BUILDERS = {
    TaskFamily.CONTEXTUAL_REMAP: _build_contextual_remapping_v2,
    TaskFamily.RULE_TRANSFORMATION: _build_rule_transformation_v2,
    TaskFamily.FINITE_STATE: _build_finite_state_v2,
}


def _replace_split(task: MetaTask, split: DevSplit) -> MetaTask:
    return MetaTask(
        family=task.family,
        split=split,
        events=task.events,
        probes=task.probes,
        rule_fingerprint=task.rule_fingerprint,
        assignment_fingerprint=task.assignment_fingerprint,
        composition_fingerprint=task.composition_fingerprint,
        paraphrase_group_fingerprint=task.paraphrase_group_fingerprint,
    )


def _build_task_with_split(
    split: DevSplit,
    family: TaskFamily,
    task_index: int,
    config: TaskGeneratorConfig,
    collision_nonce: int = 0,
) -> tuple[MetaTask, tuple[SupportFact, ...], tuple[ProbeSupport, ...]]:
    stream = _HashStream(
        root_seed=config.root_seed,
        split=split,
        family=family,
        task_index=task_index,
        collision_nonce=collision_nonce,
    )
    task, facts, probes = _BUILDERS[family](stream, config)
    return _replace_split(task, split), facts, probes


def _fingerprints(task: MetaTask) -> set[str]:
    return {
        task.rule_fingerprint,
        task.assignment_fingerprint,
        task.composition_fingerprint,
        task.paraphrase_group_fingerprint,
    }


def _catalog_digest(
    train: Sequence[MetaTask], validation: Sequence[MetaTask]
) -> str:
    entries: list[dict[str, str]] = []
    for split, tasks in (
        (DevSplit.META_TRAIN, train),
        (DevSplit.META_VALIDATION, validation),
    ):
        for i, task in enumerate(tasks):
            entries.append(
                {
                    "split": split.value,
                    "index": str(i),
                    "family": task.family.value,
                    "rule_fingerprint": task.rule_fingerprint,
                    "assignment_fingerprint": task.assignment_fingerprint,
                    "composition_fingerprint": task.composition_fingerprint,
                    "paraphrase_group_fingerprint": task.paraphrase_group_fingerprint,
                }
            )
    return _canonical_sha256(entries)


def build_dev_catalog_v2(
    config: TaskGeneratorConfig,
) -> tuple[DevTaskCatalog, SupportBundle]:
    """Build both DEV splits of the v2 lineage plus their support certificates.

    Train is built first, then validation in fixed family/index order with
    deterministic collision rejection.  Construction fails loudly
    (``SupportError``) rather than emitting an unsupported task.
    """
    train_tasks: list[MetaTask] = []
    validation_tasks: list[MetaTask] = []
    support_entries: list[TaskSupport] = []

    def register(
        split: DevSplit,
        family: TaskFamily,
        index: int,
        task: MetaTask,
        facts: tuple[SupportFact, ...],
        probes: tuple[ProbeSupport, ...],
    ) -> None:
        support_entries.append(
            TaskSupport(
                split=split.value,
                family=family.value,
                task_index=index,
                rule_fingerprint=task.rule_fingerprint,
                facts=facts,
                probes=probes,
            )
        )

    seen: set[str] = set()
    for family in _FAMILY_ORDER:
        for i in range(config.train_tasks_per_family):
            # Train tasks are generated at nonce 0, exactly like the v1
            # lineage: the firewall guards train/validation overlap, not
            # train-internal fingerprint repetition.
            task, facts, probes = _build_task_with_split(
                DevSplit.META_TRAIN, family, i, config, collision_nonce=0
            )
            seen |= _fingerprints(task)
            train_tasks.append(task)
            register(DevSplit.META_TRAIN, family, i, task, facts, probes)

    for family in _FAMILY_ORDER:
        for i in range(config.validation_tasks_per_family):
            for nonce in range(MAX_COLLISION_NONCE + 1):
                task, facts, probes = _build_task_with_split(
                    DevSplit.META_VALIDATION, family, i, config, collision_nonce=nonce
                )
                if not (_fingerprints(task) & seen):
                    break
            else:  # pragma: no cover
                raise SplitFirewallError(
                    f"could not resolve validation collision for {family.value} index {i}"
                )
            seen |= _fingerprints(task)
            validation_tasks.append(task)
            register(DevSplit.META_VALIDATION, family, i, task, facts, probes)

    audit = audit_split_firewall(train_tasks, validation_tasks)
    assert_split_firewall(audit)
    catalog = DevTaskCatalog(
        meta_train=tuple(train_tasks),
        meta_validation=tuple(validation_tasks),
        catalog_sha256=_catalog_digest(train_tasks, validation_tasks),
        split_audit=audit,
    )
    bundle = SupportBundle(
        lineage=TASKGEN_SCHEMA_V2,
        catalog_sha256=catalog.catalog_sha256,
        tasks=tuple(support_entries),
    )
    return catalog, bundle
