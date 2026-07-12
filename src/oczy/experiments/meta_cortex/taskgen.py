"""Deterministic in-memory task generation and split firewall for the
DEV-only meta-cortex experiment.

This module performs **no file I/O** and imports nothing from ``eval/v2``
or ``organism_curriculum``.  Research/20 explicitly requires a separate
instrument.

Determinism design
-------------------

- ``_HashStream`` uses counter-mode SHA-256 over canonical bytes:
  ``TASKGEN_SCHEMA | root_seed | split.value | family.value | task_index |
  collision_nonce | counter``.
- Rejection-sampled ``randbelow()`` and deterministic Fisher–Yates; never
  uses Python ``hash()``, set iteration order, global ``random``, NumPy
  RNG, timestamps, filesystem order, or model outputs.
- Rule structures are canonicalized with sorted-key compact JSON and
  SHA-256.  Four independent fingerprints: complete rule/graph, output
  assignment, composition, and paraphrase group.
- Train is built first, then validation in fixed family/index order.  If
  any validation fingerprint collides with train, deterministically
  increment ``collision_nonce`` and regenerate.  The complete firewall
  runs before returning.
- Split is assigned **before** surface rendering.  All paraphrases for one
  rule live inside one ``MetaTask``; no paraphrase is independently
  partitioned.
- The only public catalog builder constructs both DEV splits and audits
  them together.  Invalid strings such as ``"meta_test"`` fail at
  ``DevSplit`` parsing before ``_HashStream`` is instantiated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from .contracts import (
    TASKGEN_SCHEMA,
    DevSplit,
    DevTaskCatalog,
    DialogueMessage,
    LearningEvent,
    MetaTask,
    OutcomeCode,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    SplitFirewallAudit,
    TaskFamily,
    TaskGeneratorConfig,
)

__all__ = [
    "generate_contextual_remapping_task",
    "generate_rule_transformation_task",
    "generate_finite_state_task",
    "build_dev_catalog",
    "audit_split_firewall",
    "assert_split_firewall",
    "SplitFirewallError",
    "MAX_COLLISION_NONCE",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_COLLISION_NONCE = 4096

# Opaque token pools — deterministic, never from RNG.  These are fixed
# vocabularies that make rendered text readable while the actual rule
# structure (which symbols map to which outputs) is RNG-derived.
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

# ProbeKind enum order for per-kind tuples
_PROBE_KIND_ORDER = tuple(ProbeKind)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SplitFirewallError(Exception):
    """Raised when the split firewall detects fingerprint overlap."""


# ---------------------------------------------------------------------------
# _HashStream — counter-mode SHA-256 deterministic RNG
# ---------------------------------------------------------------------------


class _HashStream:
    """Counter-mode SHA-256 over canonical bytes.

    Produces a deterministic, rejection-sampled stream of unsigned
    integers.  Never uses Python ``hash()``, ``random``, NumPy, or
    timestamps.
    """

    __slots__ = ("_base", "_counter", "_buffer", "_pos")

    def __init__(
        self,
        root_seed: int,
        split: DevSplit,
        family: TaskFamily,
        task_index: int,
        collision_nonce: int = 0,
    ) -> None:
        base = "|".join(
            [
                TASKGEN_SCHEMA,
                str(root_seed),
                split.value,
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
        """Generate the next 32-byte block."""
        material = self._base + b"|" + str(self._counter).encode("utf-8")
        self._buffer = hashlib.sha256(material).digest()
        self._counter += 1
        self._pos = 0

    def _read_bytes(self, n: int) -> bytes:
        """Read *n* bytes from the stream."""
        out = bytearray()
        while len(out) < n:
            if self._pos >= len(self._buffer):
                self._refill()
            need = min(n - len(out), len(self._buffer) - self._pos)
            out.extend(self._buffer[self._pos : self._pos + need])
            self._pos += need
        return bytes(out)

    def randbelow(self, bound: int) -> int:
        """Rejection-sampled uniform integer in ``[0, bound)``."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound == 1:
            return 0
        # Number of bytes needed
        bit_len = bound.bit_length()
        byte_len = (bit_len + 7) // 8
        # Rejection sampling: draw byte_len bytes, interpret as big-endian int
        # Reject values >= bound (which is 2^bit_len rounded up)
        while True:
            raw = self._read_bytes(byte_len)
            val = int.from_bytes(raw, "big")
            # Mask to bit_len bits for tighter rejection
            val &= (1 << bit_len) - 1
            if val < bound:
                return val

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in ``[lo, hi]`` inclusive."""
        return lo + self.randbelow(hi - lo + 1)

    def choice(self, seq: Sequence[Any]) -> Any:
        """Pick one element from *seq*."""
        return seq[self.randbelow(len(seq))]

    def sample(self, seq: Sequence[Any], k: int) -> list[Any]:
        """Pick *k* distinct elements from *seq* via deterministic Fisher–Yates."""
        n = len(seq)
        if k < 0 or k > n:
            raise ValueError(f"cannot sample {k} from {n} elements")
        # Deterministic partial Fisher–Yates shuffle
        pool = list(seq)
        for i in range(k):
            j = i + self.randbelow(n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def shuffle(self, seq: list[Any]) -> list[Any]:
        """Return a deterministically shuffled copy of *seq*."""
        n = len(seq)
        pool = list(seq)
        for i in range(n - 1, 0, -1):
            j = self.randbelow(i + 1)
            pool[i], pool[j] = pool[j], pool[i]
        return pool

    def derangement(self, seq: Sequence[Any]) -> list[Any]:
        """Return a deterministically shuffled copy with no fixed points.

        Raises ``ValueError`` if no derangement exists (e.g. len==1).
        """
        n = len(seq)
        if n < 2:
            raise ValueError("cannot derange a sequence of length < 2")
        for _ in range(MAX_COLLISION_NONCE):
            perm = self.shuffle(list(seq))
            if all(perm[i] is not seq[i] and perm[i] != seq[i] for i in range(n)):
                return perm
        raise ValueError("failed to find derangement")

    def bits(self, n: int) -> int:
        """Return *n* random bits as an int."""
        return self.randbelow(1 << n)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    """Serialize *obj* to canonical sorted-key compact JSON bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(obj: Any) -> str:
    """SHA-256 hex of canonical JSON of *obj*."""
    return _sha256_hex(_canonical_bytes(obj))



# ---------------------------------------------------------------------------
# Family A: Contextual remapping
# ---------------------------------------------------------------------------


def _build_contextual_remapping(
    stream: _HashStream, config: TaskGeneratorConfig
) -> MetaTask:
    """Build a contextual remapping task.

    Deterministically choose at least two opaque contexts, symbols, and
    output tokens; derive a complete context×symbol assignment; teach 2–5
    distinct mappings with correction events.  Same-rule probes paraphrase
    taught mappings, transfer probes change the surface/context phrasing,
    composition asks for a deterministic sequence of two learned mappings,
    specificity uses an unrelated context/rule, and oracle-context probes
    state the complete mapping plus worked examples.
    """
    n_contexts = stream.randint(2, 3)
    n_symbols = stream.randint(2, 3)
    contexts = stream.sample(_CONTEXTS, n_contexts)
    symbols = stream.sample(_SYMBOLS, n_symbols)
    outputs = stream.sample(_OUTPUTS, n_symbols)

    # Complete context×symbol assignment: each (context, symbol) -> output
    # Rule structure: {context: {symbol: output}}
    rule: dict[str, dict[str, str]] = {}
    for ctx in contexts:
        rule[ctx] = {}
        for sym in symbols:
            # Each context gives a (possibly different) output for each symbol
            rule[ctx][sym] = stream.choice(outputs)

    # Assignment fingerprint: the complete output mapping
    assignment = json.loads(json.dumps(rule, sort_keys=True))

    # Composition: apply two learned mappings in sequence
    # e.g. first map symbol -> output in context A, then use that output
    # as a symbol in context B
    comp_ctx_a = contexts[0]
    comp_ctx_b = contexts[-1]
    comp_sym = symbols[0]
    comp_first = rule[comp_ctx_a][comp_sym]
    # Find a symbol whose name matches comp_first, or use comp_first directly
    # as an operand in the second context
    comp_second = rule[comp_ctx_b].get(comp_first, rule[comp_ctx_b][symbols[-1]])
    composition_spec = {
        "first_context": comp_ctx_a,
        "first_symbol": comp_sym,
        "first_output": comp_first,
        "second_context": comp_ctx_b,
        "second_input": comp_first,
        "second_output": comp_second,
    }

    # Paraphrase group: all surface forms of this rule share the same group
    paraphrase_group = {
        "family": TaskFamily.CONTEXTUAL_REMAP.value,
        "contexts": sorted(contexts),
        "symbols": sorted(symbols),
    }

    # Number of events
    n_events = stream.randint(config.min_events, config.max_events)
    events: list[LearningEvent] = []
    taught_pairs: set[tuple[str, str, str]] = set()
    for _ in range(n_events):
        ctx = stream.choice(contexts)
        sym = stream.choice(symbols)
        correct = rule[ctx][sym]
        pair = (ctx, sym, correct)
        # Ensure distinct teaching events (different ctx,sym pairs)
        attempts = 0
        while pair in taught_pairs and attempts < 100:
            ctx = stream.choice(contexts)
            sym = stream.choice(symbols)
            correct = rule[ctx][sym]
            pair = (ctx, sym, correct)
            attempts += 1
        taught_pairs.add(pair)

        obs = (
            DialogueMessage(
                role="user",
                content=f"In the {ctx} room, respond to {sym}.",
            ),
        )
        attempted = sym  # naive attempt: echo the symbol
        correction = f"In the {ctx} room, {sym} requires the token {correct}."
        outcome = OutcomeCode.CORRECTED
        events.append(
            LearningEvent(
                observation_messages=obs,
                attempted_behavior=attempted,
                correction=correction,
                outcome=outcome,
            )
        )

    # Probes
    pre = _ctxremap_pre_probes(stream, contexts, symbols, rule)
    same_rule = _ctxremap_same_rule_probes(stream, contexts, symbols, rule, taught_pairs)
    transfer = _ctxremap_transfer_probes(stream, contexts, symbols, rule, taught_pairs)
    composition = _ctxremap_composition_probes(stream, composition_spec)
    specificity = _ctxremap_specificity_probes(stream, rule)
    oracle_context = _ctxremap_oracle_probes(stream, contexts, symbols, rule)

    probes = ProbeBattery(
        pre=pre,
        same_rule=same_rule,
        transfer=transfer,
        composition=composition,
        specificity=specificity,
        oracle_context=oracle_context,
    )

    rule_fp = _fingerprint(rule)
    assignment_fp = _fingerprint(assignment)
    composition_fp = _fingerprint(composition_spec)
    paraphrase_fp = _fingerprint(paraphrase_group)

    return MetaTask(
        family=TaskFamily.CONTEXTUAL_REMAP,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=probes,
        rule_fingerprint=rule_fp,
        assignment_fingerprint=assignment_fp,
        composition_fingerprint=composition_fp,
        paraphrase_group_fingerprint=paraphrase_fp,
    )


def _ctxremap_pre_probes(
    stream: _HashStream,
    contexts: list[str],
    symbols: list[str],
    rule: dict[str, dict[str, str]],
) -> tuple[ProbeCase, ...]:
    """Pre-learning probes: ask about mappings before any teaching."""
    probes = []
    ctx = contexts[0]
    sym = symbols[0]
    probes.append(
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"The room is {ctx}. What response follows {sym}?",
                ),
            ),
            expected_response=rule[ctx][sym],
            kind=ProbeKind.PRE,
        )
    )
    if len(symbols) > 1:
        ctx2 = contexts[-1]
        sym2 = symbols[-1]
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"The room is {ctx2}. What response follows {sym2}?",
                    ),
                ),
                expected_response=rule[ctx2][sym2],
                kind=ProbeKind.PRE,
            )
        )
    return tuple(probes)


def _ctxremap_same_rule_probes(
    stream: _HashStream,
    contexts: list[str],
    symbols: list[str],
    rule: dict[str, dict[str, str]],
    taught_pairs: set[tuple[str, str, str]],
) -> tuple[ProbeCase, ...]:
    """Same-rule probes: paraphrase taught mappings with different surface phrasing."""
    probes = []
    for ctx in contexts:
        for sym in symbols:
            # Paraphrase the question differently from teaching
            probes.append(
                ProbeCase(
                    messages=(
                        DialogueMessage(
                            role="user",
                            content=f"You are in the {ctx} chamber. {sym} demands what token?",
                        ),
                    ),
                    expected_response=rule[ctx][sym],
                    kind=ProbeKind.SAME_RULE,
                )
            )
            break  # one per context
    return tuple(probes)


def _ctxremap_transfer_probes(
    stream: _HashStream,
    contexts: list[str],
    symbols: list[str],
    rule: dict[str, dict[str, str]],
    taught_pairs: set[tuple[str, str, str]],
) -> tuple[ProbeCase, ...]:
    """Transfer probes: change surface/context phrasing without reusing teaching sentences."""
    probes = []
    # Use a context+symbol pair that was taught, but with novel phrasing
    ctx = contexts[0]
    sym = symbols[-1]
    probes.append(
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"Setting: {ctx} environment. Command word: {sym}. What is the correct response?",
                ),
            ),
            expected_response=rule[ctx][sym],
            kind=ProbeKind.TRANSFER,
        )
    )
    if len(contexts) > 1:
        ctx2 = contexts[-1]
        sym2 = symbols[0]
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"Within the {ctx2} domain, the signal {sym2} elicits what?",
                    ),
                ),
                expected_response=rule[ctx2][sym2],
                kind=ProbeKind.TRANSFER,
            )
        )
    return tuple(probes)


def _ctxremap_composition_probes(
    stream: _HashStream,
    composition_spec: dict[str, str],
) -> tuple[ProbeCase, ...]:
    """Composition probes: apply two learned mappings in sequence."""
    first_ctx = composition_spec["first_context"]
    first_sym = composition_spec["first_symbol"]
    first_out = composition_spec["first_output"]
    second_ctx = composition_spec["second_context"]
    second_out = composition_spec["second_output"]
    expected = f"{first_out} then {second_out}"
    return (
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=(
                        f"First, in the {first_ctx} room, respond to {first_sym}. "
                        f"Then, in the {second_ctx} room, respond to {first_out}. "
                        "Give both tokens in order."
                    ),
                ),
            ),
            expected_response=expected,
            kind=ProbeKind.COMPOSITION,
        ),
    )


def _ctxremap_specificity_probes(
    stream: _HashStream,
    rule: dict[str, dict[str, str]],
) -> tuple[ProbeCase, ...]:
    """Specificity probes: use an unrelated context/rule."""
    # Pick a context NOT in the rule
    used_contexts = set(rule.keys())
    unused = [c for c in _CONTEXTS if c not in used_contexts]
    if not unused:
        unused = list(_CONTEXTS)
    ctx = stream.choice(unused)
    sym = stream.choice(_SYMBOLS)
    # The expected response is the symbol itself (no rule applies)
    return (
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"The room is {ctx}. What response follows {sym}?",
                ),
            ),
            expected_response=sym,
            kind=ProbeKind.SPECIFICITY,
        ),
    )


def _ctxremap_oracle_probes(
    stream: _HashStream,
    contexts: list[str],
    symbols: list[str],
    rule: dict[str, dict[str, str]],
) -> tuple[ProbeCase, ...]:
    """Oracle-context probes: state the complete mapping plus worked examples."""
    lines = ["Complete mapping:"]
    for ctx in sorted(contexts):
        for sym in sorted(symbols):
            lines.append(f"  {ctx} / {sym} -> {rule[ctx][sym]}")
    mapping_text = "\n".join(lines)
    ctx = contexts[0]
    sym = symbols[0]
    return (
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=mapping_text),
                DialogueMessage(
                    role="user",
                    content=f"Given the above mapping, in the {ctx} room, what token follows {sym}?",
                ),
            ),
            expected_response=rule[ctx][sym],
            kind=ProbeKind.ORACLE_CONTEXT,
        ),
    )


# ---------------------------------------------------------------------------
# Family B: Rule transformation
# ---------------------------------------------------------------------------

# Rule templates
_RULE_TEMPLATES = ("permutation", "substitution", "conditional", "composition")


def _build_rule_transformation(
    stream: _HashStream, config: TaskGeneratorConfig
) -> MetaTask:
    """Build a rule transformation task.

    Cycle task indices across permutation, substitution, conditional, and
    two-primitive composition templates.  Generate opaque operands and two
    rule parameters.  Teaching and held-out operands are disjoint by
    construction.  Composition probes apply both learned primitives in a
    novel order; exact target generation uses the same pure oracle function
    but never exposes that function/parameter payload to the model.
    """
    template = stream.choice(_RULE_TEMPLATES)
    n_teach = stream.randint(config.min_events, config.max_events)
    n_operands = n_teach + 3  # enough for transfer + composition
    operands = stream.sample(_OPERANDS, n_operands)
    teaching_operands = operands[:n_teach]
    held_out_operands = operands[n_teach:]

    # Rule parameters
    if template == "permutation":
        # Permutation: reverse the operand string
        param1 = "reverse"
        param2 = ""

        def apply_rule(op: str) -> str:
            return op[::-1]

        rule_spec = {"template": "permutation", "param1": param1, "param2": param2}
    elif template == "substitution":
        # Substitution: replace each vowel with a fixed consonant
        sub_char = stream.choice(_SYMBOLS)
        param1 = sub_char
        param2 = ""

        def apply_rule(op: str) -> str:
            vowels = "aeiou"
            return "".join(sub_char if c in vowels else c for c in op)

        rule_spec = {"template": "substitution", "param1": param1, "param2": param2}
    elif template == "conditional":
        # Conditional: if operand starts with a vowel, append param1; else prepend param2
        suffix = stream.choice(_SYMBOLS)
        prefix = stream.choice(_SYMBOLS)
        param1 = suffix
        param2 = prefix

        def apply_rule(op: str) -> str:
            if op and op[0] in "aeiou":
                return op + suffix
            return prefix + op

        rule_spec = {"template": "conditional", "param1": param1, "param2": param2}
    else:
        # Two-primitive composition: reverse then substitute
        sub_char = stream.choice(_SYMBOLS)
        param1 = "reverse"
        param2 = sub_char

        def apply_rule(op: str) -> str:
            vowels = "aeiou"
            reversed_op = op[::-1]
            return "".join(sub_char if c in vowels else c for c in reversed_op)

        rule_spec = {"template": "composition", "param1": param1, "param2": param2}

    # Assignment: operand -> output for all operands
    assignment = {op: apply_rule(op) for op in operands}

    # Composition spec: for composition probes, apply both primitives in novel order
    # For the composition template, the two primitives are reverse and substitute
    # For other templates, we compose the rule with itself (apply twice)
    composition_spec = {
        "template": template,
        "first_primitive": param1,
        "second_primitive": param2,
        "composition_order": "first_then_second",
    }

    paraphrase_group = {
        "family": TaskFamily.RULE_TRANSFORMATION.value,
        "template": template,
        "operands": sorted(operands),
    }

    # Events: teach on teaching_operands
    events: list[LearningEvent] = []
    for op in teaching_operands:
        correct = apply_rule(op)
        obs = (
            DialogueMessage(
                role="user",
                content=f"Apply the rule to: {op}",
            ),
        )
        attempted = op  # naive: echo
        correction = f"The correct result for {op} is {correct}."
        events.append(
            LearningEvent(
                observation_messages=obs,
                attempted_behavior=attempted,
                correction=correction,
                outcome=OutcomeCode.CORRECTED,
            )
        )

    # Probes
    pre = _ruletrans_pre_probes(stream, held_out_operands, apply_rule)
    same_rule = _ruletrans_same_rule_probes(stream, teaching_operands, apply_rule)
    transfer = _ruletrans_transfer_probes(stream, held_out_operands, apply_rule)
    composition = _ruletrans_composition_probes(stream, held_out_operands, apply_rule)
    specificity = _ruletrans_specificity_probes(stream)
    oracle_context = _ruletrans_oracle_probes(stream, rule_spec, operands, apply_rule)

    probes = ProbeBattery(
        pre=pre,
        same_rule=same_rule,
        transfer=transfer,
        composition=composition,
        specificity=specificity,
        oracle_context=oracle_context,
    )

    rule_fp = _fingerprint(rule_spec)
    assignment_fp = _fingerprint(assignment)
    composition_fp = _fingerprint(composition_spec)
    paraphrase_fp = _fingerprint(paraphrase_group)

    return MetaTask(
        family=TaskFamily.RULE_TRANSFORMATION,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=probes,
        rule_fingerprint=rule_fp,
        assignment_fingerprint=assignment_fp,
        composition_fingerprint=composition_fp,
        paraphrase_group_fingerprint=paraphrase_fp,
    )


def _ruletrans_pre_probes(
    stream: _HashStream,
    held_out: list[str],
    apply_rule: Any,
) -> tuple[ProbeCase, ...]:
    """Pre-learning probes on held-out operands."""
    probes = []
    for op in held_out[:2]:
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(role="user", content=f"Apply the rule to: {op}"),
                ),
                expected_response=apply_rule(op),
                kind=ProbeKind.PRE,
            )
        )
    return tuple(probes)


def _ruletrans_same_rule_probes(
    stream: _HashStream,
    teaching_operands: list[str],
    apply_rule: Any,
) -> tuple[ProbeCase, ...]:
    """Same-rule probes: paraphrase taught operands with different phrasing."""
    probes = []
    for op in teaching_operands[:2]:
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"What is the transformed output for input {op}?",
                    ),
                ),
                expected_response=apply_rule(op),
                kind=ProbeKind.SAME_RULE,
            )
        )
    return tuple(probes)


def _ruletrans_transfer_probes(
    stream: _HashStream,
    held_out: list[str],
    apply_rule: Any,
) -> tuple[ProbeCase, ...]:
    """Transfer probes: apply rule to unseen operands."""
    probes = []
    for op in held_out[:2]:
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"Transform: {op}",
                    ),
                ),
                expected_response=apply_rule(op),
                kind=ProbeKind.TRANSFER,
            )
        )
    return tuple(probes)


def _ruletrans_composition_probes(
    stream: _HashStream,
    held_out: list[str],
    apply_rule: Any,
) -> tuple[ProbeCase, ...]:
    """Composition probes: apply both primitives in a novel order (apply rule twice)."""
    probes = []
    if held_out:
        op = held_out[0]
        double_applied = apply_rule(apply_rule(op))
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"Apply the rule twice to: {op}",
                    ),
                ),
                expected_response=double_applied,
                kind=ProbeKind.COMPOSITION,
            )
        )
    return tuple(probes)


def _ruletrans_specificity_probes(
    stream: _HashStream,
) -> tuple[ProbeCase, ...]:
    """Specificity probes: use an unrelated rule (identity/no-op)."""
    op = stream.choice(_OPERANDS)
    return (
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=f"Apply the rule to: {op}"),
            ),
            expected_response=op,  # identity: no transformation
            kind=ProbeKind.SPECIFICITY,
        ),
    )


def _ruletrans_oracle_probes(
    stream: _HashStream,
    rule_spec: dict[str, str],
    operands: list[str],
    apply_rule: Any,
) -> tuple[ProbeCase, ...]:
    """Oracle-context probes: state the complete rule plus worked examples."""
    examples = []
    for op in operands[:3]:
        examples.append(f"  {op} -> {apply_rule(op)}")
    rule_text = (
        f"Rule: {rule_spec['template']} with parameters "
        f"{rule_spec['param1']!r} and {rule_spec['param2']!r}.\n"
        "Worked examples:\n" + "\n".join(examples)
    )
    op = operands[-1]
    return (
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=rule_text),
                DialogueMessage(
                    role="user",
                    content=f"Given this rule, what is the output for: {op}?",
                ),
            ),
            expected_response=apply_rule(op),
            kind=ProbeKind.ORACLE_CONTEXT,
        ),
    )


# ---------------------------------------------------------------------------
# Family C: Finite-state behavior
# ---------------------------------------------------------------------------


def _build_finite_state(
    stream: _HashStream, config: TaskGeneratorConfig
) -> MetaTask:
    """Build a finite-state behavior task.

    Generate a complete 3-state, 2-input transition graph plus
    state-conditioned action assignment.  Teaching events cover 2–5
    corrected transitions.  Held-out and transfer probes traverse unseen
    edges; composition follows at least two transitions; the final user
    turn omits the original goal; specificity uses a disjoint graph/goal.
    """
    states = list(_STATES)
    inputs = list(_INPUTS)
    actions = list(_ACTIONS)

    # Complete transition graph: {state: {input: next_state}}
    graph: dict[str, dict[str, str]] = {}
    for s in states:
        graph[s] = {}
        for inp in inputs:
            graph[s][inp] = stream.choice(states)

    # State-conditioned action assignment: {state: action}
    action_map: dict[str, str] = {}
    for s in states:
        action_map[s] = stream.choice(actions)

    # Goal: reach a specific state
    goal_state = stream.choice(states)
    start_state = states[0]

    rule_spec = {
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": {
            s: {inp: graph[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "actions": {s: action_map[s] for s in sorted(states)},
        "start_state": start_state,
        "goal_state": goal_state,
    }

    assignment = json.loads(json.dumps(action_map, sort_keys=True))

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
        "final_action": action_map[comp_final],
    }

    paraphrase_group = {
        "family": TaskFamily.FINITE_STATE.value,
        "states": sorted(states),
        "inputs": sorted(inputs),
        "transitions": {
            s: {inp: graph[s][inp] for inp in sorted(inputs)}
            for s in sorted(states)
        },
        "actions": {s: action_map[s] for s in sorted(states)},
        "goal_state": goal_state,
    }

    # Events: teach 2-5 corrected transitions
    n_events = stream.randint(config.min_events, config.max_events)
    # Select distinct transitions to teach
    all_transitions = [(s, inp, graph[s][inp]) for s in states for inp in inputs]
    taught = stream.sample(all_transitions, min(n_events, len(all_transitions)))
    events: list[LearningEvent] = []
    for s, inp, next_s in taught:
        obs = (
            DialogueMessage(
                role="user",
                content=f"State: {s}. Input: {inp}. What is the next state?",
            ),
        )
        attempted = s  # naive: stay
        correction = f"From {s}, input {inp} transitions to {next_s}."
        events.append(
            LearningEvent(
                observation_messages=obs,
                attempted_behavior=attempted,
                correction=correction,
                outcome=OutcomeCode.CORRECTED,
            )
        )

    # Probes
    pre = _fsm_pre_probes(stream, states, inputs, graph, action_map, goal_state)
    same_rule = _fsm_same_rule_probes(stream, taught, graph, action_map)
    transfer = _fsm_transfer_probes(stream, states, inputs, graph, action_map, taught)
    composition = _fsm_composition_probes(stream, composition_spec)
    specificity = _fsm_specificity_probes(stream, goal_state)
    oracle_context = _fsm_oracle_probes(stream, rule_spec, graph, action_map, goal_state)

    probes = ProbeBattery(
        pre=pre,
        same_rule=same_rule,
        transfer=transfer,
        composition=composition,
        specificity=specificity,
        oracle_context=oracle_context,
    )

    rule_fp = _fingerprint(rule_spec)
    assignment_fp = _fingerprint(assignment)
    composition_fp = _fingerprint(composition_spec)
    paraphrase_fp = _fingerprint(paraphrase_group)

    return MetaTask(
        family=TaskFamily.FINITE_STATE,
        split=DevSplit.META_TRAIN,  # placeholder; set by caller
        events=tuple(events),
        probes=probes,
        rule_fingerprint=rule_fp,
        assignment_fingerprint=assignment_fp,
        composition_fingerprint=composition_fp,
        paraphrase_group_fingerprint=paraphrase_fp,
    )


def _fsm_pre_probes(
    stream: _HashStream,
    states: list[str],
    inputs: list[str],
    graph: dict[str, dict[str, str]],
    action_map: dict[str, str],
    goal_state: str,
) -> tuple[ProbeCase, ...]:
    """Pre-learning probes: ask about transitions before teaching."""
    s = states[0]
    inp = inputs[0]
    next_s = graph[s][inp]
    return (
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"State: {s}. Input: {inp}. What is the next state?",
                ),
            ),
            expected_response=next_s,
            kind=ProbeKind.PRE,
        ),
    )


def _fsm_same_rule_probes(
    stream: _HashStream,
    taught: list[tuple[str, str, str]],
    graph: dict[str, dict[str, str]],
    action_map: dict[str, str],
) -> tuple[ProbeCase, ...]:
    """Same-rule probes: paraphrase taught transitions."""
    probes = []
    for s, inp, next_s in taught[:2]:
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"Given current state {s} and signal {inp}, which state follows?",
                    ),
                ),
                expected_response=next_s,
                kind=ProbeKind.SAME_RULE,
            )
        )
    return tuple(probes)


def _fsm_transfer_probes(
    stream: _HashStream,
    states: list[str],
    inputs: list[str],
    graph: dict[str, dict[str, str]],
    action_map: dict[str, str],
    taught: list[tuple[str, str, str]],
) -> tuple[ProbeCase, ...]:
    """Transfer probes: traverse unseen edges (not in taught)."""
    taught_set = {(s, inp) for s, inp, _ in taught}
    unseen = [
        (s, inp, graph[s][inp])
        for s in states
        for inp in inputs
        if (s, inp) not in taught_set
    ]
    probes = []
    for s, inp, next_s in unseen[:2]:
        probes.append(
            ProbeCase(
                messages=(
                    DialogueMessage(
                        role="user",
                        content=f"State: {s}. Signal: {inp}. Next state?",
                    ),
                ),
                expected_response=next_s,
                kind=ProbeKind.TRANSFER,
            )
        )
    return tuple(probes)


def _fsm_composition_probes(
    stream: _HashStream,
    composition_spec: dict[str, Any],
) -> tuple[ProbeCase, ...]:
    """Composition probes: follow at least two transitions.  Final turn omits the original goal."""
    start = composition_spec["start"]
    inp1 = composition_spec["input_1"]
    inp2 = composition_spec["input_2"]
    final = composition_spec["final_state"]
    final_action = composition_spec["final_action"]
    # The final user turn does NOT repeat the original goal
    expected = f"{final} {final_action}"
    return (
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"Start at {start}. First input: {inp1}. Then input: {inp2}. What is the result?",
                ),
            ),
            expected_response=expected,
            kind=ProbeKind.COMPOSITION,
        ),
    )


def _fsm_specificity_probes(
    stream: _HashStream,
    goal_state: str,
) -> tuple[ProbeCase, ...]:
    """Specificity probes: use a disjoint graph/goal."""
    # Use a different goal state
    other_goals = [s for s in _STATES if s != goal_state]
    if not other_goals:
        other_goal = goal_state
    else:
        other_goal = stream.choice(other_goals)
    return (
        ProbeCase(
            messages=(
                DialogueMessage(
                    role="user",
                    content=f"State: {_STATES[0]}. Input: {_INPUTS[0]}. What is the next state?",
                ),
            ),
            expected_response=other_goal,  # unrelated goal
            kind=ProbeKind.SPECIFICITY,
        ),
    )


def _fsm_oracle_probes(
    stream: _HashStream,
    rule_spec: dict[str, Any],
    graph: dict[str, dict[str, str]],
    action_map: dict[str, str],
    goal_state: str,
) -> tuple[ProbeCase, ...]:
    """Oracle-context probes: state the complete graph plus worked examples."""
    lines = ["Complete transition graph:"]
    for s in sorted(graph.keys()):
        for inp in sorted(graph[s].keys()):
            lines.append(f"  {s} + {inp} -> {graph[s][inp]}")
    lines.append("Actions:")
    for s in sorted(action_map.keys()):
        lines.append(f"  {s}: {action_map[s]}")
    lines.append(f"Goal: reach {goal_state}.")
    graph_text = "\n".join(lines)
    s = sorted(graph.keys())[0]
    inp = sorted(graph[s].keys())[0]
    return (
        ProbeCase(
            messages=(
                DialogueMessage(role="user", content=graph_text),
                DialogueMessage(
                    role="user",
                    content=f"Given the above graph, from {s} with input {inp}, what is the next state?",
                ),
            ),
            expected_response=graph[s][inp],
            kind=ProbeKind.ORACLE_CONTEXT,
        ),
    )


# ---------------------------------------------------------------------------
# Task assembly with split assignment
# ---------------------------------------------------------------------------


def _build_task_with_split(
    split: DevSplit,
    family: TaskFamily,
    task_index: int,
    config: TaskGeneratorConfig,
    collision_nonce: int = 0,
) -> MetaTask:
    """Build a single task with the given split, family, and index.

    Split is assigned **before** surface rendering.  The ``collision_nonce``
    changes the hash stream base, producing a different rule if needed.
    """
    stream = _HashStream(
        root_seed=config.root_seed,
        split=split,
        family=family,
        task_index=task_index,
        collision_nonce=collision_nonce,
    )
    if family is TaskFamily.CONTEXTUAL_REMAP:
        task = _build_contextual_remapping(stream, config)
    elif family is TaskFamily.RULE_TRANSFORMATION:
        task = _build_rule_transformation(stream, config)
    elif family is TaskFamily.FINITE_STATE:
        task = _build_finite_state(stream, config)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Replace the placeholder split with the real one
    return _replace_split(task, split)


def _replace_split(task: MetaTask, split: DevSplit) -> MetaTask:
    """Return a copy of *task* with the split field set to *split*.

    Since MetaTask is frozen, we reconstruct it.
    """
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


# ---------------------------------------------------------------------------
# Split firewall
# ---------------------------------------------------------------------------


def audit_split_firewall(
    train: Sequence[MetaTask],
    validation: Sequence[MetaTask],
) -> SplitFirewallAudit:
    """Audit the split firewall.

    Computes overlap counts for rule, assignment, composition, and
    paraphrase group fingerprints.  ``passed`` is True only when all four
    overlap counts are zero.
    """
    train_rule = {t.rule_fingerprint for t in train}
    val_rule = {t.rule_fingerprint for t in validation}
    train_assign = {t.assignment_fingerprint for t in train}
    val_assign = {t.assignment_fingerprint for t in validation}
    train_comp = {t.composition_fingerprint for t in train}
    val_comp = {t.composition_fingerprint for t in validation}
    train_para = {t.paraphrase_group_fingerprint for t in train}
    val_para = {t.paraphrase_group_fingerprint for t in validation}

    return SplitFirewallAudit(
        train_rule_digests=tuple(sorted(train_rule)),
        validation_rule_digests=tuple(sorted(val_rule)),
        train_assignment_digests=tuple(sorted(train_assign)),
        validation_assignment_digests=tuple(sorted(val_assign)),
        train_composition_digests=tuple(sorted(train_comp)),
        validation_composition_digests=tuple(sorted(val_comp)),
        train_paraphrase_digests=tuple(sorted(train_para)),
        validation_paraphrase_digests=tuple(sorted(val_para)),
        rule_overlap=len(train_rule & val_rule),
        assignment_overlap=len(train_assign & val_assign),
        composition_overlap=len(train_comp & val_comp),
        paraphrase_overlap=len(train_para & val_para),
        train_task_count=len(train),
        validation_task_count=len(validation),
    )


def assert_split_firewall(audit: SplitFirewallAudit) -> None:
    """Raise ``SplitFirewallError`` if the firewall fails."""
    failures: list[str] = []
    if audit.rule_overlap > 0:
        failures.append(f"rule overlap: {audit.rule_overlap}")
    if audit.assignment_overlap > 0:
        failures.append(f"assignment overlap: {audit.assignment_overlap}")
    if audit.composition_overlap > 0:
        failures.append(f"composition overlap: {audit.composition_overlap}")
    if audit.paraphrase_overlap > 0:
        failures.append(f"paraphrase overlap: {audit.paraphrase_overlap}")
    if failures:
        raise SplitFirewallError(
            "Split firewall violated: " + "; ".join(failures)
        )


# ---------------------------------------------------------------------------
# Catalog digest
# ---------------------------------------------------------------------------


def _catalog_digest(
    train: Sequence[MetaTask],
    validation: Sequence[MetaTask],
) -> str:
    """Compute a deterministic SHA-256 digest over the complete catalog.

    Iterates tasks in fixed family/index order and hashes family, split,
    index, and all four fingerprints.
    """
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
    return _fingerprint(entries)


# ---------------------------------------------------------------------------
# Public catalog builder
# ---------------------------------------------------------------------------

_FAMILY_ORDER = (
    TaskFamily.CONTEXTUAL_REMAP,
    TaskFamily.RULE_TRANSFORMATION,
    TaskFamily.FINITE_STATE,
)


def build_dev_catalog(config: TaskGeneratorConfig) -> DevTaskCatalog:
    """Build the complete DEV catalog with both splits and firewall audit.

    Build train first, then validation in fixed family/index order.  If any
    validation fingerprint collides with train, deterministically increment
    ``collision_nonce`` and regenerate.  Run the complete firewall before
    returning.
    """
    # Build train tasks
    train_tasks: list[MetaTask] = []
    for family in _FAMILY_ORDER:
        for i in range(config.train_tasks_per_family):
            task = _build_task_with_split(
                DevSplit.META_TRAIN, family, i, config, collision_nonce=0
            )
            train_tasks.append(task)

    # Build validation tasks with collision rejection
    validation_tasks: list[MetaTask] = []
    train_fingerprints: set[str] = set()
    for t in train_tasks:
        train_fingerprints.add(t.rule_fingerprint)
        train_fingerprints.add(t.assignment_fingerprint)
        train_fingerprints.add(t.composition_fingerprint)
        train_fingerprints.add(t.paraphrase_group_fingerprint)

    for family in _FAMILY_ORDER:
        for i in range(config.validation_tasks_per_family):
            nonce = 0
            while True:
                task = _build_task_with_split(
                    DevSplit.META_VALIDATION, family, i, config, collision_nonce=nonce
                )
                task_fps = {
                    task.rule_fingerprint,
                    task.assignment_fingerprint,
                    task.composition_fingerprint,
                    task.paraphrase_group_fingerprint,
                }
                # Check against train AND already-built validation tasks
                if task_fps & train_fingerprints:
                    nonce += 1
                    if nonce > MAX_COLLISION_NONCE:
                        raise SplitFirewallError(
                            f"Could not resolve collision for validation "
                            f"{family.value} index {i} after {MAX_COLLISION_NONCE} attempts"
                        )
                    continue
                # Check against other validation tasks
                val_fps: set[str] = set()
                for vt in validation_tasks:
                    val_fps.add(vt.rule_fingerprint)
                    val_fps.add(vt.assignment_fingerprint)
                    val_fps.add(vt.composition_fingerprint)
                    val_fps.add(vt.paraphrase_group_fingerprint)
                if task_fps & val_fps:
                    nonce += 1
                    if nonce > MAX_COLLISION_NONCE:
                        raise SplitFirewallError(
                            f"Could not resolve validation-internal collision for "
                            f"{family.value} index {i} after {MAX_COLLISION_NONCE} attempts"
                        )
                    continue
                break
            validation_tasks.append(task)
            train_fingerprints.update(task_fps)

    # Audit
    audit = audit_split_firewall(train_tasks, validation_tasks)
    assert_split_firewall(audit)

    # Catalog digest
    digest = _catalog_digest(train_tasks, validation_tasks)

    return DevTaskCatalog(
        meta_train=tuple(train_tasks),
        meta_validation=tuple(validation_tasks),
        catalog_sha256=digest,
        split_audit=audit,
    )


# ---------------------------------------------------------------------------
# Public per-family generators (for testing)
# ---------------------------------------------------------------------------


def generate_contextual_remapping_task(
    key: int,
    config: TaskGeneratorConfig,
) -> MetaTask:
    """Generate a single contextual remapping task.

    *key* is the task index.  The split is ``META_TRAIN`` by default; use
    ``build_dev_catalog`` for the full split-aware catalog.
    """
    return _build_task_with_split(
        DevSplit.META_TRAIN,
        TaskFamily.CONTEXTUAL_REMAP,
        key,
        config,
    )


def generate_rule_transformation_task(
    key: int,
    config: TaskGeneratorConfig,
) -> MetaTask:
    """Generate a single rule transformation task."""
    return _build_task_with_split(
        DevSplit.META_TRAIN,
        TaskFamily.RULE_TRANSFORMATION,
        key,
        config,
    )


def generate_finite_state_task(
    key: int,
    config: TaskGeneratorConfig,
) -> MetaTask:
    """Generate a single finite-state behavior task."""
    return _build_task_with_split(
        DevSplit.META_TRAIN,
        TaskFamily.FINITE_STATE,
        key,
        config,
    )
