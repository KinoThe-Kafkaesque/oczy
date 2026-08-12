"""Deterministic base64 rotational-XOR task catalog for R24.

Every registered rule operates on a six-bit base64 alphabet index.  It rotates
that index left by zero, one, or two bits and XORs the result with a six-bit
mask.  The registered teaching-only alias maps query ``B`` to the same effective
index as ``A``; therefore A/B remains ambiguous and C is the first rotation
probe, while all held-out D..``/`` inputs use their literal indices.  The model
sees only byte records such as ``x=A`` and a one-byte base64 answer; catalog
fingerprints and split metadata never enter model-facing records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

BASE64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
SYMBOLS = BASE64_ALPHABET
BIT_WIDTH = 6
INDEX_MASK = (1 << BIT_WIDTH) - 1
ROTATIONS = (0, 1, 2)
TEACHING_INPUTS = (0, 1, 2)
HELDOUT_INPUTS = tuple(range(3, len(BASE64_ALPHABET)))
TEACHING_SYMBOLS = bytes(BASE64_ALPHABET[index] for index in TEACHING_INPUTS)
HELDOUT_SYMBOLS = bytes(BASE64_ALPHABET[index] for index in HELDOUT_INPUTS)

RULE_SCHEMA_VERSION = "oczy/r24-base64-rotational-xor-rule/v3"
CATALOG_SCHEMA_VERSION = "oczy/r24-base64-rotational-xor-catalog/v3"

DEFAULT_ROOT_SEED = 24_003
DEFAULT_CORTEX_META_TRAIN_SIZE = 50
DEFAULT_CORTEX_META_DEV_SIZE = 25
DEFAULT_SEALED_TEST_SIZE = 50
DEFAULT_ORGAN_DEV_SIZE = 25
DEFAULT_ORGAN_TRAIN_SIZE = 42
REGISTERED_CANDIDATE_COUNT = len(ROTATIONS) * len(BASE64_ALPHABET)


class ToyFamily(StrEnum):
    """Three registered transformation families, one per rotation geometry."""

    XOR = "base64_xor"
    ROTATE1_XOR = "base64_rotate1_xor"
    ROTATE2_XOR = "base64_rotate2_xor"


class ToySplit(StrEnum):
    """The five mutually disjoint protocol partitions."""

    CORTEX_META_TRAIN = "cortex_meta_train"
    CORTEX_META_DEV = "cortex_meta_dev"
    SEALED_TEST = "sealed_test"
    ORGAN_DEV = "organ_dev"
    ORGAN_TRAIN = "organ_train"


@dataclass(frozen=True, slots=True)
class ToyRule:
    """Apply registered B-to-A alias, rotate left, then XOR with a mask."""

    rotation: int
    mask: int

    def __post_init__(self) -> None:
        if isinstance(self.rotation, bool) or self.rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}")
        _validate_index(self.mask, name="mask")

    @property
    def family(self) -> ToyFamily:
        return {
            0: ToyFamily.XOR,
            1: ToyFamily.ROTATE1_XOR,
            2: ToyFamily.ROTATE2_XOR,
        }[self.rotation]

    def apply(self, input_value: int) -> int:
        """Return rotational XOR after applying the registered B-to-A alias."""

        _validate_index(input_value)
        # B is a registered teaching alias for A; C is the first rotation probe.
        effective_input = 0 if input_value == 1 else input_value
        return rotate_left_6(effective_input, self.rotation) ^ self.mask

    def canonical_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": RULE_SCHEMA_VERSION,
            "family": self.family.value,
            "alphabet_ascii": BASE64_ALPHABET.decode("ascii"),
            "bit_width": BIT_WIDTH,
            "rotation": self.rotation,
            "mask": self.mask,
            "teaching_alias_ascii": "B->A",
        }

    @property
    def fingerprint(self) -> str:
        """Process-independent SHA-256 identity of the mathematical rule."""

        return _sha256_json(self.canonical_dict())

    @property
    def rule_fingerprint(self) -> str:
        return self.fingerprint


@dataclass(frozen=True, slots=True)
class ToyInteraction:
    """One model query and its exact one-byte answer."""

    input_value: int
    output_value: int

    def __post_init__(self) -> None:
        _validate_index(self.input_value)
        _validate_index(self.output_value, name="output_value")

    @property
    def input_index(self) -> int:
        return self.input_value

    @property
    def output_index(self) -> int:
        return self.output_value

    @property
    def input_symbol(self) -> bytes:
        return bytes((BASE64_ALPHABET[self.input_value],))

    @property
    def output_symbol(self) -> bytes:
        return bytes((BASE64_ALPHABET[self.output_value],))

    @property
    def query_bytes(self) -> bytes:
        return b"x=" + self.input_symbol

    @property
    def answer_bytes(self) -> bytes:
        return self.output_symbol

    @property
    def query(self) -> bytes:
        return self.query_bytes

    @property
    def answer(self) -> bytes:
        return self.answer_bytes

    @property
    def query_text(self) -> str:
        return self.query_bytes.decode("ascii")

    @property
    def answer_text(self) -> str:
        return self.answer_bytes.decode("ascii")

    @property
    def feedback_bytes(self) -> bytes:
        return self.query_bytes + b"\nanswer=" + self.answer_bytes

    @property
    def model_record(self) -> dict[str, str]:
        """JSON-safe model payload, intentionally containing no task identity."""

        return {"query": self.query_text, "answer": self.answer_text}

    def canonical_dict(self) -> dict[str, int | str]:
        return {
            "input_index": self.input_value,
            "output_index": self.output_value,
            "query_ascii": self.query_text,
            "answer_ascii": self.answer_text,
        }


@dataclass(frozen=True, slots=True)
class ToyTask:
    """One rule materialized on the common teaching and held-out inputs."""

    split: ToySplit
    rule: ToyRule
    teaching_interactions: tuple[ToyInteraction, ...]
    heldout_interactions: tuple[ToyInteraction, ...]
    donor_rule_fingerprint: str

    @property
    def rule_fingerprint(self) -> str:
        return self.rule.fingerprint

    @property
    def fingerprint(self) -> str:
        """Administrative identity; it is never included in model records."""

        return _sha256_json(
            {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "split": self.split.value,
                "rule_fingerprint": self.rule_fingerprint,
                "teaching_inputs": [item.input_value for item in self.teaching],
                "heldout_inputs": [item.input_value for item in self.heldout],
            }
        )

    @property
    def teaching(self) -> tuple[ToyInteraction, ...]:
        return self.teaching_interactions

    @property
    def heldout(self) -> tuple[ToyInteraction, ...]:
        return self.heldout_interactions

    @property
    def teaching_feedback(self) -> bytes:
        return b"\n---\n".join(item.feedback_bytes for item in self.teaching)

    @property
    def teaching_records(self) -> tuple[dict[str, str], ...]:
        return tuple(item.model_record for item in self.teaching)

    @property
    def heldout_records(self) -> tuple[dict[str, str], ...]:
        return tuple(item.model_record for item in self.heldout)

    def model_facing_dict(self) -> dict[str, object]:
        """Return only query/answer records, with no IDs, split, or fingerprints."""

        return {
            "teaching": list(self.teaching_records),
            "heldout": list(self.heldout_records),
        }

    def canonical_dict(self) -> dict[str, object]:
        """Return administrative catalog data used for manifests and audits."""

        return {
            "split": self.split.value,
            "rule": self.rule.canonical_dict(),
            "rule_fingerprint": self.rule_fingerprint,
            "task_fingerprint": self.fingerprint,
            "donor_rule_fingerprint": self.donor_rule_fingerprint,
            "teaching": [item.canonical_dict() for item in self.teaching],
            "heldout": [item.canonical_dict() for item in self.heldout],
        }


@dataclass(frozen=True, slots=True)
class IdentifiabilityAudit:
    """Exhaustive matching result for the fixed three teaching interactions."""

    candidate_count: int
    teaching_inputs: tuple[int, ...]
    unique_after_one: int
    unique_after_two: int
    unique_after_three: int
    ambiguous_after_three: int
    minimum_matches_after_one: int
    maximum_matches_after_one: int
    minimum_matches_after_two: int
    maximum_matches_after_two: int
    minimum_matches_after_three: int
    maximum_matches_after_three: int
    audit_sha256: str

    @property
    def exactly_three(self) -> bool:
        """Whether the three-row teaching protocol identifies every candidate."""

        return (
            len(self.teaching_inputs) == 3
            and self.unique_after_one == 0
            and self.unique_after_two == 0
            and self.minimum_matches_after_two > 1
            and self.unique_after_three == self.candidate_count
            and self.ambiguous_after_three == 0
            and self.minimum_matches_after_three == 1
            and self.maximum_matches_after_three == 1
        )

    @property
    def all_identified(self) -> bool:
        return self.exactly_three

    def canonical_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "teaching_inputs": list(self.teaching_inputs),
            "unique_after_one": self.unique_after_one,
            "unique_after_two": self.unique_after_two,
            "unique_after_three": self.unique_after_three,
            "ambiguous_after_three": self.ambiguous_after_three,
            "minimum_matches_after_one": self.minimum_matches_after_one,
            "maximum_matches_after_one": self.maximum_matches_after_one,
            "minimum_matches_after_two": self.minimum_matches_after_two,
            "maximum_matches_after_two": self.maximum_matches_after_two,
            "minimum_matches_after_three": self.minimum_matches_after_three,
            "maximum_matches_after_three": self.maximum_matches_after_three,
            "audit_sha256": self.audit_sha256,
            "all_identified": self.all_identified,
        }


@dataclass(frozen=True, slots=True)
class ToyCatalog:
    """All candidates, disjoint partitions, donors, and reproducibility hashes."""

    root_seed: int
    candidates: tuple[ToyRule, ...]
    cortex_meta_train: tuple[ToyTask, ...]
    cortex_meta_dev: tuple[ToyTask, ...]
    sealed_test: tuple[ToyTask, ...]
    organ_dev: tuple[ToyTask, ...]
    organ_train: tuple[ToyTask, ...]
    identifiability: IdentifiabilityAudit
    registry_sha256: str
    split_sha256: Mapping[str, str]
    manifest_sha256: str

    @property
    def registered_candidates(self) -> tuple[ToyRule, ...]:
        return self.candidates

    @property
    def registered_rules(self) -> tuple[ToyRule, ...]:
        return self.candidates

    @property
    def all_tasks(self) -> tuple[ToyTask, ...]:
        return tuple(task for split in ToySplit for task in self.tasks_for_split(split))

    @property
    def non_test_tasks(self) -> tuple[ToyTask, ...]:
        return tuple(task for task in self.all_tasks if task.split is not ToySplit.SEALED_TEST)

    def tasks_for_split(self, split: ToySplit | str) -> tuple[ToyTask, ...]:
        resolved = ToySplit(split)
        return {
            ToySplit.CORTEX_META_TRAIN: self.cortex_meta_train,
            ToySplit.CORTEX_META_DEV: self.cortex_meta_dev,
            ToySplit.SEALED_TEST: self.sealed_test,
            ToySplit.ORGAN_DEV: self.organ_dev,
            ToySplit.ORGAN_TRAIN: self.organ_train,
        }[resolved]

    def phase_a_organ_corpus(
        self, *, include_all_non_test_development: bool = False
    ) -> tuple[ToyTask, ...]:
        """Return Phase A data without mutating catalog partitions.

        A later Phase A run can deliberately opt into the union of every
        non-test partition.  The sealed test is absent in both modes.
        """

        if not include_all_non_test_development:
            return self.organ_train
        return self.non_test_tasks

    def donor_pairs(
        self, split: ToySplit | str | None = None
    ) -> tuple[tuple[ToyTask, ToyTask], ...]:
        """Return each task and its deterministic in-split donor."""

        tasks = self.all_tasks if split is None else self.tasks_for_split(split)
        indexed = {task.rule_fingerprint: task for task in self.all_tasks}
        return tuple((task, indexed[task.donor_rule_fingerprint]) for task in tasks)

    def manifest_dict(self) -> dict[str, object]:
        return _manifest_payload(
            root_seed=self.root_seed,
            registry_sha256=self.registry_sha256,
            identifiability_sha256=self.identifiability.audit_sha256,
            split_sha256=self.split_sha256,
            split_counts={split.value: len(self.tasks_for_split(split)) for split in ToySplit},
        )


def rotate_left_6(value: int, rotation: int) -> int:
    """Rotate a validated six-bit integer left by 0, 1, or 2 places."""

    _validate_index(value)
    if isinstance(rotation, bool) or rotation not in ROTATIONS:
        raise ValueError(f"rotation must be one of {ROTATIONS}")
    if rotation == 0:
        return value
    return ((value << rotation) | (value >> (BIT_WIDTH - rotation))) & INDEX_MASK


def symbol_for_index(index: int) -> bytes:
    """Return the one-byte base64 symbol at a validated six-bit index."""

    _validate_index(index)
    return bytes((BASE64_ALPHABET[index],))


def index_for_symbol(symbol: bytes | str) -> int:
    """Return the six-bit index of exactly one ASCII base64 symbol."""

    if isinstance(symbol, str):
        try:
            raw = symbol.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("symbol must be ASCII base64") from exc
    elif isinstance(symbol, bytes):
        raw = symbol
    else:
        raise TypeError("symbol must be bytes or str")
    if len(raw) != 1 or raw[0] not in BASE64_ALPHABET:
        raise ValueError("symbol must be exactly one ASCII base64 byte")
    return BASE64_ALPHABET.index(raw)


def registered_candidates() -> tuple[ToyRule, ...]:
    """Return all 192 rules in canonical rotation/mask order."""

    return tuple(
        ToyRule(rotation=rotation, mask=mask)
        for rotation in ROTATIONS
        for mask in range(64)
    )


def make_interaction(rule: ToyRule, input_value: int) -> ToyInteraction:
    return ToyInteraction(input_value=input_value, output_value=rule.apply(input_value))


def matching_candidates(
    interactions: Sequence[ToyInteraction],
    *,
    candidates: Sequence[ToyRule] | None = None,
) -> tuple[ToyRule, ...]:
    """Brute-force registered rules consistent with every supplied row."""

    candidate_rules = registered_candidates() if candidates is None else tuple(candidates)
    return tuple(
        rule
        for rule in candidate_rules
        if all(rule.apply(item.input_value) == item.output_value for item in interactions)
    )


def identify_rule(
    teaching_interactions: Sequence[ToyInteraction],
    *,
    candidates: Sequence[ToyRule] | None = None,
) -> ToyRule:
    """Return the sole brute-force match, or fail if feedback is ambiguous."""

    matches = matching_candidates(teaching_interactions, candidates=candidates)
    if len(matches) != 1:
        raise ValueError(
            f"teaching feedback matched {len(matches)} candidates, expected exactly one"
        )
    return matches[0]


def brute_force_identifiability_audit(
    candidates: Sequence[ToyRule] | None = None,
    teaching_inputs: Sequence[int] = TEACHING_INPUTS,
) -> IdentifiabilityAudit:
    """Exhaustively verify that A/B remain ambiguous and A/B/C is unique."""

    candidate_rules = registered_candidates() if candidates is None else tuple(candidates)
    inputs = tuple(teaching_inputs)
    if inputs != TEACHING_INPUTS:
        raise ValueError(f"teaching inputs must be the fixed indices {TEACHING_INPUTS}")
    if len({rule.fingerprint for rule in candidate_rules}) != len(candidate_rules):
        raise ValueError("candidate registry contains duplicate rule fingerprints")

    prefix_match_counts: dict[int, list[int]] = {1: [], 2: [], 3: []}
    prefix_unique_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for rule in candidate_rules:
        teaching = tuple(make_interaction(rule, value) for value in inputs)
        for prefix in (1, 2, 3):
            matches = matching_candidates(
                teaching[:prefix], candidates=candidate_rules
            )
            prefix_match_counts[prefix].append(len(matches))
            prefix_unique_counts[prefix] += matches == (rule,)

    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "candidate_fingerprints": [rule.fingerprint for rule in candidate_rules],
        "teaching_inputs": list(inputs),
        "teaching_symbols_ascii": TEACHING_SYMBOLS.decode("ascii"),
        "prefix_match_counts": prefix_match_counts,
        "prefix_unique_counts": prefix_unique_counts,
    }
    audit = IdentifiabilityAudit(
        candidate_count=len(candidate_rules),
        teaching_inputs=inputs,
        unique_after_one=prefix_unique_counts[1],
        unique_after_two=prefix_unique_counts[2],
        unique_after_three=prefix_unique_counts[3],
        ambiguous_after_three=len(candidate_rules) - prefix_unique_counts[3],
        minimum_matches_after_one=min(prefix_match_counts[1], default=0),
        maximum_matches_after_one=max(prefix_match_counts[1], default=0),
        minimum_matches_after_two=min(prefix_match_counts[2], default=0),
        maximum_matches_after_two=max(prefix_match_counts[2], default=0),
        minimum_matches_after_three=min(prefix_match_counts[3], default=0),
        maximum_matches_after_three=max(prefix_match_counts[3], default=0),
        audit_sha256=_sha256_json(payload),
    )
    if not audit.all_identified:
        raise ValueError(
            f"candidate registry failed exact-three identification: {audit.canonical_dict()}"
        )
    return audit
def build_toy_catalog_v3(
    *,
    root_seed: int = DEFAULT_ROOT_SEED,
    cortex_meta_train_size: int = DEFAULT_CORTEX_META_TRAIN_SIZE,
    cortex_meta_dev_size: int = DEFAULT_CORTEX_META_DEV_SIZE,
    sealed_test_size: int = DEFAULT_SEALED_TEST_SIZE,
    organ_dev_size: int = DEFAULT_ORGAN_DEV_SIZE,
    organ_train_size: int = DEFAULT_ORGAN_TRAIN_SIZE,
) -> ToyCatalog:
    """Build the fixed 50/25/50/25/42 deterministic disjoint catalog."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise ValueError("root_seed must be an integer")
    counts = {
        ToySplit.CORTEX_META_TRAIN: cortex_meta_train_size,
        ToySplit.CORTEX_META_DEV: cortex_meta_dev_size,
        ToySplit.SEALED_TEST: sealed_test_size,
        ToySplit.ORGAN_DEV: organ_dev_size,
        ToySplit.ORGAN_TRAIN: organ_train_size,
    }
    expected_counts = {
        ToySplit.CORTEX_META_TRAIN: DEFAULT_CORTEX_META_TRAIN_SIZE,
        ToySplit.CORTEX_META_DEV: DEFAULT_CORTEX_META_DEV_SIZE,
        ToySplit.SEALED_TEST: DEFAULT_SEALED_TEST_SIZE,
        ToySplit.ORGAN_DEV: DEFAULT_ORGAN_DEV_SIZE,
        ToySplit.ORGAN_TRAIN: DEFAULT_ORGAN_TRAIN_SIZE,
    }
    if counts != expected_counts:
        rendered = {split.value: count for split, count in expected_counts.items()}
        raise ValueError(f"v3 split sizes are fixed: {rendered}")
    if sum(counts.values()) != REGISTERED_CANDIDATE_COUNT:
        raise AssertionError("fixed split sizes do not consume all candidates")

    candidates = registered_candidates()
    audit = brute_force_identifiability_audit(candidates)
    rotation_pools = _ranked_rotation_pools(candidates, root_seed=root_seed)
    cycle_index = int.from_bytes(
        hashlib.sha256(
            f"{CATALOG_SCHEMA_VERSION}|{root_seed}|rotation-cycle".encode("ascii")
        ).digest()[:8],
        "big",
    ) % len(ROTATIONS)

    def take_rules(count: int) -> tuple[ToyRule, ...]:
        nonlocal cycle_index
        selected: list[ToyRule] = []
        for _ in range(count):
            rotation = ROTATIONS[cycle_index % len(ROTATIONS)]
            selected.append(rotation_pools[rotation].pop())
            cycle_index += 1
        return tuple(selected)

    # Sealed test is allocated first; development consumers cannot alter it.
    allocation_order = (
        ToySplit.SEALED_TEST,
        ToySplit.CORTEX_META_TRAIN,
        ToySplit.CORTEX_META_DEV,
        ToySplit.ORGAN_DEV,
        ToySplit.ORGAN_TRAIN,
    )
    split_rules = {split: take_rules(counts[split]) for split in allocation_order}
    if any(rotation_pools.values()):
        raise AssertionError("split allocation left registered candidates unassigned")

    split_tasks = {split: _materialize_tasks(split, split_rules[split]) for split in ToySplit}
    _audit_split_firewall(candidates, split_tasks)
    _audit_donors(split_tasks)

    registry_hash = _sha256_json(
        {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "registered_candidates": [rule.canonical_dict() for rule in candidates],
        }
    )
    split_hashes = {
        split.value: _sha256_json(
            {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "split": split.value,
                "tasks": [task.canonical_dict() for task in split_tasks[split]],
            }
        )
        for split in ToySplit
    }
    manifest_payload = _manifest_payload(
        root_seed=root_seed,
        registry_sha256=registry_hash,
        identifiability_sha256=audit.audit_sha256,
        split_sha256=split_hashes,
        split_counts={split.value: len(split_tasks[split]) for split in ToySplit},
    )
    catalog = ToyCatalog(
        root_seed=root_seed,
        candidates=candidates,
        cortex_meta_train=split_tasks[ToySplit.CORTEX_META_TRAIN],
        cortex_meta_dev=split_tasks[ToySplit.CORTEX_META_DEV],
        sealed_test=split_tasks[ToySplit.SEALED_TEST],
        organ_dev=split_tasks[ToySplit.ORGAN_DEV],
        organ_train=split_tasks[ToySplit.ORGAN_TRAIN],
        identifiability=audit,
        registry_sha256=registry_hash,
        split_sha256=split_hashes,
        manifest_sha256=_sha256_json(manifest_payload),
    )
    if _sha256_json(catalog.manifest_dict()) != catalog.manifest_sha256:
        raise AssertionError("catalog manifest hash did not round-trip")
    return catalog


def build_catalog(**kwargs: int) -> ToyCatalog:
    """Concise alias for R24-local callers."""

    return build_toy_catalog_v3(**kwargs)


def _validate_index(value: int, *, name: str = "input_value") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= INDEX_MASK:
        raise ValueError(f"{name} must be an integer in 0..{INDEX_MASK}")


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _ranked_rotation_pools(
    candidates: Iterable[ToyRule], *, root_seed: int
) -> dict[int, list[ToyRule]]:
    grouped = {rotation: [] for rotation in ROTATIONS}
    for rule in candidates:
        grouped[rule.rotation].append(rule)
    for rotation, rules in grouped.items():
        rules.sort(
            key=lambda rule: hashlib.sha256(
                (
                    f"{CATALOG_SCHEMA_VERSION}|{root_seed}|rotation={rotation}|{rule.fingerprint}"
                ).encode("ascii")
            ).digest(),
            reverse=True,
        )
    return grouped


def _materialize_tasks(split: ToySplit, rules: Sequence[ToyRule]) -> tuple[ToyTask, ...]:
    grouped: dict[int, list[ToyRule]] = {rotation: [] for rotation in ROTATIONS}
    for rule in rules:
        grouped[rule.rotation].append(rule)

    donors: dict[str, str] = {}
    for rotation, rotation_rules in grouped.items():
        if len(rotation_rules) < 2:
            raise ValueError(f"{split.value} lacks two rotation={rotation} rules for donors")
        for index, rule in enumerate(rotation_rules):
            donor = next(
                (
                    rotation_rules[(index + offset) % len(rotation_rules)]
                    for offset in range(1, len(rotation_rules))
                    if all(
                        rotation_rules[(index + offset) % len(rotation_rules)].apply(value)
                        != rule.apply(value)
                        for value in TEACHING_INPUTS
                    )
                ),
                None,
            )
            if donor is None:
                raise ValueError(f"{split.value} lacks a valid donor for {rule.fingerprint}")
            donors[rule.fingerprint] = donor.fingerprint

    return tuple(
        ToyTask(
            split=split,
            rule=rule,
            teaching_interactions=tuple(make_interaction(rule, value) for value in TEACHING_INPUTS),
            heldout_interactions=tuple(make_interaction(rule, value) for value in HELDOUT_INPUTS),
            donor_rule_fingerprint=donors[rule.fingerprint],
        )
        for rule in rules
    )

def _audit_split_firewall(
    candidates: Sequence[ToyRule], split_tasks: Mapping[ToySplit, Sequence[ToyTask]]
) -> None:
    expected = {rule.fingerprint for rule in candidates}
    seen: set[str] = set()
    for split in ToySplit:
        fingerprints = {task.rule_fingerprint for task in split_tasks[split]}
        if len(fingerprints) != len(split_tasks[split]):
            raise ValueError(f"duplicate rule inside {split.value}")
        if overlap := seen & fingerprints:
            raise ValueError(f"split overlap entering {split.value}: {sorted(overlap)[:3]}")
        seen.update(fingerprints)
    if seen != expected:
        raise ValueError("split firewall did not assign every candidate exactly once")


def _audit_donors(split_tasks: Mapping[ToySplit, Sequence[ToyTask]]) -> None:
    for split in ToySplit:
        indexed = {task.rule_fingerprint: task for task in split_tasks[split]}
        for task in split_tasks[split]:
            donor = indexed.get(task.donor_rule_fingerprint)
            if donor is None:
                raise ValueError(f"{split.value} donor escaped its split")
            if donor.rule_fingerprint == task.rule_fingerprint:
                raise ValueError(f"{split.value} donor reused the target rule")
            if donor.rule.family is not task.rule.family:
                raise ValueError(f"{split.value} donor crossed rule families")
            if [item.query_bytes for item in donor.teaching] != [
                item.query_bytes for item in task.teaching
            ]:
                raise ValueError(f"{split.value} donor changed teaching inputs")
            for target_row, donor_row in zip(task.teaching, donor.teaching, strict=True):
                if target_row.answer_bytes == donor_row.answer_bytes:
                    raise ValueError(f"{split.value} donor did not change every teaching answer")
                if len(target_row.answer_bytes) != len(donor_row.answer_bytes):
                    raise ValueError(f"{split.value} donor changed answer length")


def _manifest_payload(
    *,
    root_seed: int,
    registry_sha256: str,
    identifiability_sha256: str,
    split_sha256: Mapping[str, str],
    split_counts: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "root_seed": root_seed,
        "alphabet_ascii": BASE64_ALPHABET.decode("ascii"),
        "bit_width": BIT_WIDTH,
        "rotations": list(ROTATIONS),
        "families": [family.value for family in ToyFamily],
        "teaching_inputs": list(TEACHING_INPUTS),
        "teaching_symbols_ascii": TEACHING_SYMBOLS.decode("ascii"),
        "teaching_alias_ascii": "B->A",
        "heldout_inputs": list(HELDOUT_INPUTS),
        "heldout_symbols_ascii": HELDOUT_SYMBOLS.decode("ascii"),
        "registry_sha256": registry_sha256,
        "identifiability_sha256": identifiability_sha256,
        "splits": {
            split.value: {
                "count": split_counts[split.value],
                "sha256": split_sha256[split.value],
            }
            for split in ToySplit
        },
    }
