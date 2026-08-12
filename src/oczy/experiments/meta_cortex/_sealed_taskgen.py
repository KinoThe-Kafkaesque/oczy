"""Private sealed meta-test task generation for the ``meta_cortex/v1`` instrument.

This module is NOT exported through the package ``__init__.py`` or CLI.
Only ``instrument.py`` (the candidate materializer) may import it.

It generates meta-test tasks using an independent 256-bit seed that is
not derived from the DEV root seed.  The generation algorithm mirrors
the DEV taskgen family builders but uses a domain-separated hash stream
with the private domain ``"meta_test"`` — this string never appears in
``DevSplit`` or any public API.

Design:
- ``_SealedHashStream`` uses counter-mode SHA-256 over canonical bytes:
  ``SEALED_TASKGEN_SCHEMA | test_seed | "meta_test" | family | task_index |
  collision_nonce | counter``.
- Family builders are duplicated here (not imported from taskgen.py) to
  keep the sealed generation path completely independent and avoid any
  dependency on taskgen.py internals.
- Collision rejection is within-domain and cross-domain (against DEV
  fingerprints supplied by the caller).
- The public DEV taskgen module is never modified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from .contracts import (
    DevSplit,
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

# Private schema — separate from the DEV taskgen schema
SEALED_TASKGEN_SCHEMA = "oczy/meta-cortex/sealed-taskgen/v1"

MAX_COLLISION_NONCE = 4096

# Token pools — identical to DEV taskgen (frozen vocabularies)
_SYMBOLS = (
    "dax", "fep", "grim", "hool", "jir", "kal", "lom", "nurp",
    "pex", "quob", "ral", "siv", "twem", "urb", "vol", "wix",
)
_CONTEXTS = (
    "amber", "azure", "bronze", "coral", "emerald", "indigo",
    "jade", "lavender", "marble", "obsidian", "pearl", "quartz",
    "ruby", "silver", "topaz", "violet",
)
_OUTPUTS = (
    "north", "south", "east", "west", "rise", "fall",
    "open", "close", "left", "right", "up", "down",
    "forward", "back", "halt", "turn",
)
_OPERANDS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "mu",
    "nu", "xi", "omicron", "pi", "rho", "sigma",
)
_STATES = ("q0", "q1", "q2")
_INPUTS = ("a", "b")
_ACTIONS = ("proceed", "wait", "yield", "hold")
_RULE_TEMPLATES = ("permutation", "substitution", "conditional", "composition")

# ---------------------------------------------------------------------------
# Sealed finite-state pools — expanded beyond DEV for semantic diversity.
# DEV uses 3 states / 2 inputs / 4 actions (Moore machine: action per state).
# Sealed uses up to 6 states / 3 inputs / 8 actions with Mealy-machine
# assignment (action per (state, input)).  This makes sealed fingerprints
# structurally distinct from DEV's and creates an assignment space of
# 8^(states×inputs) — far exceeding the 30-task requirement.
# ---------------------------------------------------------------------------
_SEALED_STATES = ("q0", "q1", "q2", "q3", "q4", "q5")
_SEALED_INPUTS = ("a", "b", "c")
_SEALED_ACTIONS = (
    "proceed", "wait", "yield", "hold",
    "advance", "retreat", "observe", "commit",
)


class _SealedHashStream:
    """Counter-mode SHA-256 over canonical bytes with sealed domain separation.

    Unlike the DEV ``_HashStream``, this uses the independent test seed
    and the private domain string ``"meta_test"`` — never ``DevSplit``.
    """

    __slots__ = ("_base", "_counter", "_buffer", "_pos")

    def __init__(
        self,
        test_seed: int,
        family: TaskFamily,
        task_index: int,
        collision_nonce: int = 0,
    ) -> None:
        base = "|".join(
            [
                SEALED_TASKGEN_SCHEMA,
                str(test_seed),
                "meta_test",
                family.value,
                str(task_index),
                str(collision_nonce),
            ]
        )
        self._base = base.encode("utf-8")
        self._counter = 0
        self._buffer = b""
        self._pos = 0

    def _refill(self) -> None:
        material = self._base + b"|" + str(self._counter).encode("utf-8")
        self._buffer = hashlib.sha256(material).digest()
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
            raw = self._read_bytes(byte_len)
            val = int.from_bytes(raw, "big")
            val &= (1 << bit_len) - 1
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

    def shuffle(self, seq: list[Any]) -> list[Any]:
        n = len(seq)
        pool = list(seq)
        for i in range(n - 1, 0, -1):
            j = self.randbelow(i + 1)
            pool[i], pool[j] = pool[j], pool[i]
        return pool


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _fingerprint(obj: Any) -> str:
    return _sha256_hex(_canonical_bytes(obj))


# ---------------------------------------------------------------------------
# Family A: Contextual remapping (mirrors DEV but with sealed stream)
# ---------------------------------------------------------------------------

def _build_contextual_remapping(
    stream: _SealedHashStream, config: TaskGeneratorConfig
) -> MetaTask:
    n_contexts = stream.randint(2, 3)
    n_symbols = stream.randint(2, 3)
    contexts = stream.sample(_CONTEXTS, n_contexts)
    symbols = stream.sample(_SYMBOLS, n_symbols)
    outputs = stream.sample(_OUTPUTS, n_symbols)

    rule: dict[str, dict[str, str]] = {}
    for ctx in contexts:
        rule[ctx] = {}
        for sym in symbols:
            rule[ctx][sym] = stream.choice(outputs)

    assignment = json.loads(json.dumps(rule, sort_keys=True))

    comp_ctx_a = contexts[0]
    comp_ctx_b = contexts[-1]
    comp_sym = symbols[0]
    comp_first = rule[comp_ctx_a][comp_sym]
    comp_second = rule[comp_ctx_b].get(comp_first, rule[comp_ctx_b][symbols[-1]])
    composition_spec = {
        "first_context": comp_ctx_a,
        "first_symbol": comp_sym,
        "first_output": comp_first,
        "second_context": comp_ctx_b,
        "second_input": comp_first,
        "second_output": comp_second,
    }

    paraphrase_group = {
        "family": TaskFamily.CONTEXTUAL_REMAP.value,
        "contexts": sorted(contexts),
        "symbols": sorted(symbols),
    }

    n_events = stream.randint(config.min_events, config.max_events)
    events: list[LearningEvent] = []
    taught_pairs: set[tuple[str, str, str]] = set()
    for _ in range(n_events):
        ctx = stream.choice(contexts)
        sym = stream.choice(symbols)
        correct = rule[ctx][sym]
        pair = (ctx, sym, correct)
        attempts = 0
        while pair in taught_pairs and attempts < 100:
            ctx = stream.choice(contexts)
            sym = stream.choice(symbols)
            correct = rule[ctx][sym]
            pair = (ctx, sym, correct)
            attempts += 1
        taught_pairs.add(pair)
        events.append(LearningEvent(
            observation_messages=(
                DialogueMessage(
                    role="user",
                    content=f"In the {ctx} room, respond to {sym}.",
                ),
            ),
            attempted_behavior=sym,
            correction=f"In the {ctx} room, {sym} requires the token {correct}.",
            outcome=OutcomeCode.CORRECTED,
        ))

    pre = _ctxremap_pre_probes(stream, contexts, symbols, rule)
    same_rule = _ctxremap_same_rule_probes(stream, contexts, symbols, rule)
    transfer = _ctxremap_transfer_probes(stream, contexts, symbols, rule)
    composition = _ctxremap_composition_probes(stream, composition_spec)
    specificity = _ctxremap_specificity_probes(stream, rule)
    oracle_context = _ctxremap_oracle_probes(stream, contexts, symbols, rule)

    probes = ProbeBattery(
        pre=pre, same_rule=same_rule, transfer=transfer,
        composition=composition, specificity=specificity,
        oracle_context=oracle_context,
    )

    return MetaTask(
        family=TaskFamily.CONTEXTUAL_REMAP,
        split=DevSplit.META_VALIDATION,  # sealed tasks carry META_VALIDATION
        events=tuple(events),
        probes=probes,
        rule_fingerprint=_fingerprint(rule),
        assignment_fingerprint=_fingerprint(assignment),
        composition_fingerprint=_fingerprint(composition_spec),
        paraphrase_group_fingerprint=_fingerprint(paraphrase_group),
    )


def _ctxremap_pre_probes(stream, contexts, symbols, rule):
    probes = []
    ctx, sym = contexts[0], symbols[0]
    probes.append(ProbeCase(
        messages=(DialogueMessage(role="user",
            content=f"The room is {ctx}. What response follows {sym}?"),),
        expected_response=rule[ctx][sym], kind=ProbeKind.PRE,
    ))
    if len(symbols) > 1:
        probes.append(ProbeCase(
            messages=(DialogueMessage(role="user",
                content=f"The room is {contexts[-1]}. What response follows {symbols[-1]}?"),),
            expected_response=rule[contexts[-1]][symbols[-1]], kind=ProbeKind.PRE,
        ))
    return tuple(probes)


def _ctxremap_same_rule_probes(stream, contexts, symbols, rule):
    probes = []
    for ctx in contexts:
        for sym in symbols:
            probes.append(ProbeCase(
                messages=(DialogueMessage(role="user",
                    content=f"You are in the {ctx} chamber. {sym} demands what token?"),),
                expected_response=rule[ctx][sym], kind=ProbeKind.SAME_RULE,
            ))
            break
    return tuple(probes)


def _ctxremap_transfer_probes(stream, contexts, symbols, rule):
    probes = []
    probes.append(ProbeCase(
        messages=(DialogueMessage(role="user",
            content=f"Setting: {contexts[0]} environment. Command word: {symbols[-1]}. What is the correct response?"),),
        expected_response=rule[contexts[0]][symbols[-1]], kind=ProbeKind.TRANSFER,
    ))
    if len(contexts) > 1:
        probes.append(ProbeCase(
            messages=(DialogueMessage(role="user",
                content=f"Within the {contexts[-1]} domain, the signal {symbols[0]} elicits what?"),),
            expected_response=rule[contexts[-1]][symbols[0]], kind=ProbeKind.TRANSFER,
        ))
    return tuple(probes)


def _ctxremap_composition_probes(stream, composition_spec):
    expected = f"{composition_spec['first_output']} then {composition_spec['second_output']}"
    return (ProbeCase(
        messages=(DialogueMessage(role="user",
            content=(f"First, in the {composition_spec['first_context']} room, "
                     f"respond to {composition_spec['first_symbol']}. "
                     f"Then, in the {composition_spec['second_context']} room, "
                     f"respond to {composition_spec['first_output']}. "
                     "Give both tokens in order.")),),
        expected_response=expected, kind=ProbeKind.COMPOSITION,
    ),)


def _ctxremap_specificity_probes(stream, rule):
    used = set(rule.keys())
    unused = [c for c in _CONTEXTS if c not in used] or list(_CONTEXTS)
    ctx = stream.choice(unused)
    sym = stream.choice(_SYMBOLS)
    return (ProbeCase(
        messages=(DialogueMessage(role="user",
            content=f"The room is {ctx}. What response follows {sym}?"),),
        expected_response=sym, kind=ProbeKind.SPECIFICITY,
    ),)


def _ctxremap_oracle_probes(stream, contexts, symbols, rule):
    lines = ["Complete mapping:"]
    for ctx in sorted(contexts):
        for sym in sorted(symbols):
            lines.append(f"  {ctx} / {sym} -> {rule[ctx][sym]}")
    return (ProbeCase(
        messages=(
            DialogueMessage(role="user", content="\n".join(lines)),
            DialogueMessage(role="user",
                content=f"Given the above mapping, in the {contexts[0]} room, what token follows {symbols[0]}?"),
        ),
        expected_response=rule[contexts[0]][symbols[0]],
        kind=ProbeKind.ORACLE_CONTEXT,
    ),)


# ---------------------------------------------------------------------------
# Family B: Rule transformation
# ---------------------------------------------------------------------------

def _build_rule_transformation(
    stream: _SealedHashStream, config: TaskGeneratorConfig
) -> MetaTask:
    template = stream.choice(_RULE_TEMPLATES)
    n_teach = stream.randint(config.min_events, config.max_events)
    n_operands = n_teach + 3
    operands = stream.sample(_OPERANDS, n_operands)
    teaching_operands = operands[:n_teach]
    held_out = operands[n_teach:]

    if template == "permutation":
        param1, param2 = "reverse", ""
        def apply_rule(op): return op[::-1]
    elif template == "substitution":
        sub_char = stream.choice(_SYMBOLS)
        param1, param2 = sub_char, ""
        def apply_rule(op):
            return "".join(sub_char if c in "aeiou" else c for c in op)
    elif template == "conditional":
        suffix = stream.choice(_SYMBOLS)
        prefix = stream.choice(_SYMBOLS)
        param1, param2 = suffix, prefix
        def apply_rule(op):
            return op + suffix if (op and op[0] in "aeiou") else prefix + op
    else:
        sub_char = stream.choice(_SYMBOLS)
        param1, param2 = "reverse", sub_char
        def apply_rule(op):
            return "".join(sub_char if c in "aeiou" else c for c in op[::-1])

    rule_spec = {"template": template, "param1": param1, "param2": param2}
    assignment = {op: apply_rule(op) for op in operands}
    composition_spec = {"template": template, "first_primitive": param1,
                        "second_primitive": param2, "composition_order": "first_then_second"}
    paraphrase_group = {"family": TaskFamily.RULE_TRANSFORMATION.value,
                        "template": template, "operands": sorted(operands)}

    events = []
    for op in teaching_operands:
        events.append(LearningEvent(
            observation_messages=(DialogueMessage(role="user", content=f"Apply the rule to: {op}"),),
            attempted_behavior=op,
            correction=f"The correct result for {op} is {apply_rule(op)}.",
            outcome=OutcomeCode.CORRECTED,
        ))

    pre = tuple(ProbeCase(
        messages=(DialogueMessage(role="user", content=f"Apply the rule to: {op}"),),
        expected_response=apply_rule(op), kind=ProbeKind.PRE,
    ) for op in held_out[:2])
    same_rule = tuple(ProbeCase(
        messages=(DialogueMessage(role="user", content=f"What is the transformed output for input {op}?"),),
        expected_response=apply_rule(op), kind=ProbeKind.SAME_RULE,
    ) for op in teaching_operands[:2])
    transfer = tuple(ProbeCase(
        messages=(DialogueMessage(role="user", content=f"Transform: {op}"),),
        expected_response=apply_rule(op), kind=ProbeKind.TRANSFER,
    ) for op in held_out[:2])
    composition = ()
    if held_out:
        op = held_out[0]
        composition = (ProbeCase(
            messages=(DialogueMessage(role="user", content=f"Apply the rule twice to: {op}"),),
            expected_response=apply_rule(apply_rule(op)), kind=ProbeKind.COMPOSITION,
        ),)
    specificity = (ProbeCase(
        messages=(DialogueMessage(role="user", content=f"Apply the rule to: {stream.choice(_OPERANDS)}"),),
        expected_response=stream.choice(_OPERANDS), kind=ProbeKind.SPECIFICITY,
    ),)
    examples = [f"  {op} -> {apply_rule(op)}" for op in operands[:3]]
    oracle_context = (ProbeCase(
        messages=(
            DialogueMessage(role="user",
                content=f"Rule: {template} with parameters {param1!r} and {param2!r}.\nWorked examples:\n" + "\n".join(examples)),
            DialogueMessage(role="user", content=f"Given this rule, what is the output for: {operands[-1]}?"),
        ),
        expected_response=apply_rule(operands[-1]), kind=ProbeKind.ORACLE_CONTEXT,
    ),)

    probes = ProbeBattery(pre=pre, same_rule=same_rule, transfer=transfer,
                          composition=composition, specificity=specificity,
                          oracle_context=oracle_context)

    return MetaTask(
        family=TaskFamily.RULE_TRANSFORMATION,
        split=DevSplit.META_VALIDATION,
        events=tuple(events), probes=probes,
        rule_fingerprint=_fingerprint(rule_spec),
        assignment_fingerprint=_fingerprint(assignment),
        composition_fingerprint=_fingerprint(composition_spec),
        paraphrase_group_fingerprint=_fingerprint(paraphrase_group),
    )


# ---------------------------------------------------------------------------
# Family C: Finite-state behavior
# ---------------------------------------------------------------------------

def _build_finite_state(
    stream: _SealedHashStream, config: TaskGeneratorConfig
) -> MetaTask:
    """Build a sealed finite-state task with expanded semantic diversity.

    Uses variable-size Mealy machines (action per (state, input)) drawn
    from expanded sealed pools — structurally distinct from DEV's
    fixed 3-state Moore machines.  The variable topology (3–5 states,
    2–3 inputs, 8 actions) and Mealy assignment create an assignment
    space of 8^(states×inputs), ensuring ≥30 disjoint sealed rules.
    """
    n_states = stream.randint(3, 5)
    n_inputs = stream.randint(2, 3)
    states = stream.sample(_SEALED_STATES, n_states)
    inputs = stream.sample(_SEALED_INPUTS, n_inputs)
    actions = list(_SEALED_ACTIONS)

    # Complete transition graph: {state: {input: next_state}}
    graph: dict[str, dict[str, str]] = {}
    for s in states:
        graph[s] = {}
        for inp in inputs:
            graph[s][inp] = stream.choice(states)

    # Mealy-machine action assignment: {state: {input: action}}
    # Structurally distinct from DEV's Moore {state: action}.
    action_map: dict[str, dict[str, str]] = {}
    for s in states:
        action_map[s] = {}
        for inp in inputs:
            action_map[s][inp] = stream.choice(actions)

    goal_state = stream.choice(states)
    start_state = states[0]

    rule_spec = {
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": {
            s: {inp: graph[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "actions": {
            s: {inp: action_map[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "start_state": start_state,
        "goal_state": goal_state,
    }
    assignment = {
        s: {inp: action_map[s][inp] for inp in sorted(inputs)}
        for s in sorted(states)
    }

    # Composition: follow at least two transitions
    comp_input_1 = stream.choice(inputs)
    comp_input_2 = stream.choice(inputs)
    comp_mid = graph[start_state][comp_input_1]
    comp_final = graph[comp_mid][comp_input_2]
    composition_spec = {
        "start": start_state,
        "input_1": comp_input_1,
        "mid_state": comp_mid,
        "input_2": comp_input_2,
        "final_state": comp_final,
        "final_action": action_map[comp_final][comp_input_2],
    }

    paraphrase_group = {
        "family": TaskFamily.FINITE_STATE.value,
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": {
            s: {inp: graph[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "actions": {
            s: {inp: action_map[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "goal_state": goal_state,
    }

    # Events: teach corrected transitions
    n_events = stream.randint(config.min_events, config.max_events)
    all_transitions = [(s, inp, graph[s][inp]) for s in states for inp in inputs]
    taught = stream.sample(all_transitions, min(n_events, len(all_transitions)))
    events: list[LearningEvent] = []
    for s, inp, next_s in taught:
        events.append(LearningEvent(
            observation_messages=(DialogueMessage(
                role="user",
                content=f"State: {s}. Input: {inp}. What is the next state?",
            ),),
            attempted_behavior=s,
            correction=f"From {s}, input {inp} transitions to {next_s}.",
            outcome=OutcomeCode.CORRECTED,
        ))

    # Probes
    pre = (ProbeCase(
        messages=(DialogueMessage(
            role="user",
            content=f"State: {states[0]}. Input: {inputs[0]}. What is the next state?",
        ),),
        expected_response=graph[states[0]][inputs[0]],
        kind=ProbeKind.PRE,
    ),)

    same_rule = tuple(ProbeCase(
        messages=(DialogueMessage(
            role="user",
            content=f"Given current state {s} and signal {inp}, which state follows?",
        ),),
        expected_response=next_s,
        kind=ProbeKind.SAME_RULE,
    ) for s, inp, next_s in taught[:2])

    taught_set = {(s, inp) for s, inp, _ in taught}
    unseen = [
        (s, inp, graph[s][inp])
        for s in states for inp in inputs
        if (s, inp) not in taught_set
    ]
    transfer = tuple(ProbeCase(
        messages=(DialogueMessage(
            role="user",
            content=f"State: {s}. Signal: {inp}. Next state?",
        ),),
        expected_response=next_s,
        kind=ProbeKind.TRANSFER,
    ) for s, inp, next_s in unseen[:2])

    composition = (ProbeCase(
        messages=(DialogueMessage(
            role="user",
            content=(
                f"Start at {start_state}. First input: {comp_input_1}. "
                f"Then input: {comp_input_2}. What is the result?"
            ),
        ),),
        expected_response=f"{comp_final} {action_map[comp_final][comp_input_2]}",
        kind=ProbeKind.COMPOSITION,
    ),)

    other_goals = [s for s in states if s != goal_state] or [goal_state]
    specificity = (ProbeCase(
        messages=(DialogueMessage(
            role="user",
            content=f"State: {states[0]}. Input: {inputs[0]}. What is the next state?",
        ),),
        expected_response=stream.choice(other_goals),
        kind=ProbeKind.SPECIFICITY,
    ),)

    lines = ["Complete transition graph:"]
    for s in sorted(graph.keys()):
        for inp in sorted(graph[s].keys()):
            lines.append(f"  {s} + {inp} -> {graph[s][inp]}")
    lines.append("Actions:")
    for s in sorted(action_map.keys()):
        for inp in sorted(action_map[s].keys()):
            lines.append(f"  {s} + {inp}: {action_map[s][inp]}")
    lines.append(f"Goal: reach {goal_state}.")
    oracle_context = (ProbeCase(
        messages=(
            DialogueMessage(role="user", content="\n".join(lines)),
            DialogueMessage(
                role="user",
                content=(
                    f"Given the above graph, from {sorted(graph.keys())[0]} "
                    f"with input {sorted(graph[sorted(graph.keys())[0]].keys())[0]}, "
                    "what is the next state?"
                ),
            ),
        ),
        expected_response=graph[sorted(graph.keys())[0]][sorted(graph[sorted(graph.keys())[0]].keys())[0]],
        kind=ProbeKind.ORACLE_CONTEXT,
    ),)

    probes = ProbeBattery(
        pre=pre, same_rule=same_rule, transfer=transfer,
        composition=composition, specificity=specificity,
        oracle_context=oracle_context,
    )

    return MetaTask(
        family=TaskFamily.FINITE_STATE,
        split=DevSplit.META_VALIDATION,
        events=tuple(events), probes=probes,
        rule_fingerprint=_fingerprint(rule_spec),
        assignment_fingerprint=_fingerprint(assignment),
        composition_fingerprint=_fingerprint(composition_spec),
        paraphrase_group_fingerprint=_fingerprint(paraphrase_group),
    )


# ---------------------------------------------------------------------------
# Public (module-internal) API
# ---------------------------------------------------------------------------

_FAMILY_BUILDERS = {
    TaskFamily.CONTEXTUAL_REMAP: _build_contextual_remapping,
    TaskFamily.RULE_TRANSFORMATION: _build_rule_transformation,
    TaskFamily.FINITE_STATE: _build_finite_state,
}


def generate_sealed_task(
    test_seed: int,
    family: TaskFamily,
    task_index: int,
    config: TaskGeneratorConfig,
    collision_nonce: int = 0,
) -> MetaTask:
    """Generate a single sealed meta-test task.

    Uses the independent test seed and private ``"meta_test"`` domain.
    The returned ``MetaTask`` has ``split=DevSplit.META_VALIDATION``
    (never ``META_TRAIN``); the caller tracks its sealed status.
    """
    stream = _SealedHashStream(
        test_seed=test_seed,
        family=family,
        task_index=task_index,
        collision_nonce=collision_nonce,
    )
    builder = _FAMILY_BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"Unknown family: {family}")
    return builder(stream, config)


def generate_sealed_tasks(
    test_seed: int,
    config: TaskGeneratorConfig,
    tasks_per_family: int,
    dev_fingerprints: set[str] | None = None,
) -> tuple[MetaTask, ...]:
    """Generate all sealed meta-test tasks with collision rejection.

    *dev_fingerprints* is the set of all DEV fingerprints (train +
    tuning + calibration) for cross-domain collision rejection.
    """
    family_order = (
        TaskFamily.CONTEXTUAL_REMAP,
        TaskFamily.RULE_TRANSFORMATION,
        TaskFamily.FINITE_STATE,
    )
    if dev_fingerprints is None:
        dev_fingerprints = set()

    all_tasks: list[MetaTask] = []
    seen_fps: set[str] = set(dev_fingerprints)

    for family in family_order:
        for i in range(tasks_per_family):
            nonce = 0
            while True:
                task = generate_sealed_task(
                    test_seed, family, i, config, collision_nonce=nonce
                )
                task_fps = {
                    task.rule_fingerprint,
                    task.assignment_fingerprint,
                    task.composition_fingerprint,
                    task.paraphrase_group_fingerprint,
                }
                if task_fps & seen_fps:
                    nonce += 1
                    if nonce > MAX_COLLISION_NONCE:
                        raise RuntimeError(
                            f"Sealed collision exhausted for {family.value} "
                            f"index {i} after {MAX_COLLISION_NONCE} attempts"
                        )
                    continue
                # Check within-domain
                within_fps: set[str] = set()
                for t in all_tasks:
                    within_fps.add(t.rule_fingerprint)
                    within_fps.add(t.assignment_fingerprint)
                    within_fps.add(t.composition_fingerprint)
                    within_fps.add(t.paraphrase_group_fingerprint)
                if task_fps & within_fps:
                    nonce += 1
                    if nonce > MAX_COLLISION_NONCE:
                        raise RuntimeError(
                            f"Sealed within-domain collision exhausted for "
                            f"{family.value} index {i}"
                        )
                    continue
                break
            all_tasks.append(task)
            seen_fps.update(task_fps)

    return tuple(all_tasks)
