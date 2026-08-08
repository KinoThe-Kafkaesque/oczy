"""Human-authorized R24 Phase-A v2 corpus and fail-closed semantic audit.

V1 copied every meta-cortex probe into byte-decoder supervision.  Specificity
probes can have the same rendered model input as a different target, and
contextual-remap composition asks for mappings absent from the complete table.
V2 excludes those two known-invalid classes and hashes the actual rendered
records rather than the task catalog index.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from .pretrain import PretrainExample, build_pretrain_corpus

CORPUS_SCHEMA_VERSION = "oczy/r24-phase-a-corpus/v2"


@dataclass(frozen=True, slots=True)
class CorpusAudit:
    schema_version: str
    raw_train_examples: int
    raw_validation_examples: int
    train_examples: int
    validation_examples: int
    excluded: dict[str, int]
    train_rules: int
    validation_rules: int
    rule_overlap: int
    train_input_conflicts: int
    validation_input_conflicts: int
    corpus_sha256: str
    split_sha256: str


def canonical_example(example: PretrainExample) -> dict[str, str]:
    return {
        "rule_fp": example.rule_fp,
        "oracle_text": example.oracle_text,
        "query_text": example.query_text,
        "answer_text": example.answer_text,
        "family": example.family,
        "kind": example.kind,
    }


def hash_examples(examples: Sequence[PretrainExample]) -> str:
    payload = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "examples": [canonical_example(example) for example in examples],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _filter_reason(example: PretrainExample) -> str | None:
    if example.kind == "specificity":
        return "specificity_rendered_input_conflict"
    if example.family == "contextual_remap" and example.kind == "composition":
        return "contextual_composition_undefined_mapping"
    return None


def _filter(examples: Sequence[PretrainExample], excluded: Counter[str]) -> list[PretrainExample]:
    kept: list[PretrainExample] = []
    for example in examples:
        reason = _filter_reason(example)
        if reason is None:
            kept.append(example)
        else:
            excluded[reason] += 1
    return kept


def input_conflicts(
    examples: Sequence[PretrainExample], *, oracle_mode: str
) -> dict[tuple[str, str], set[str]]:
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in examples:
        state_input = example.oracle_text if oracle_mode == "text" else example.rule_fp
        targets[(state_input, example.query_text)].add(example.answer_text)
    return {key: values for key, values in targets.items() if len(values) > 1}


class AuditDetails(TypedDict):
    conflicts: dict[str, object]
    train_rules: int
    validation_rules: int
    rule_overlap: int
    train_hash: str
    validation_hash: str


def audit_examples(
    train: Sequence[PretrainExample], validation: Sequence[PretrainExample]
) -> AuditDetails:
    conflicts: dict[str, object] = {}
    for split_name, examples in (("train", train), ("validation", validation)):
        for mode in ("text", "hash"):
            found = input_conflicts(examples, oracle_mode=mode)
            if found:
                conflicts[f"{split_name}_{mode}"] = [
                    {"state_query": list(key), "targets": sorted(values)}
                    for key, values in sorted(found.items())
                ]
    train_rules = {example.rule_fp for example in train}
    validation_rules = {example.rule_fp for example in validation}
    return {
        "conflicts": conflicts,
        "train_rules": len(train_rules),
        "validation_rules": len(validation_rules),
        "rule_overlap": len(train_rules & validation_rules),
        "train_hash": hash_examples(train),
        "validation_hash": hash_examples(validation),
    }


def build_phase_a_v2_corpus(
    *, root_seed: int, train_per_family: int, val_per_family: int
) -> tuple[list[PretrainExample], list[PretrainExample], CorpusAudit]:
    raw_train, raw_validation, _, _ = build_pretrain_corpus(
        root_seed=root_seed,
        train_per_family=train_per_family,
        val_per_family=val_per_family,
    )
    excluded: Counter[str] = Counter()
    train = _filter(raw_train, excluded)
    validation = _filter(raw_validation, excluded)
    details = audit_examples(train, validation)
    conflicts = details["conflicts"]
    if conflicts:
        raise ValueError(
            "R24 v2 corpus has conflicting targets for identical model inputs: "
            + json.dumps(conflicts, sort_keys=True)[:2000]
        )
    if details["rule_overlap"]:
        raise ValueError("R24 v2 train/validation rule firewall failed")
    split_payload = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "train_sha256": details["train_hash"],
        "validation_sha256": details["validation_hash"],
        "train_rules": details["train_rules"],
        "validation_rules": details["validation_rules"],
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    corpus_hash = hashlib.sha256(
        (str(details["train_hash"]) + str(details["validation_hash"])).encode()
    ).hexdigest()
    audit = CorpusAudit(
        schema_version=CORPUS_SCHEMA_VERSION,
        raw_train_examples=len(raw_train),
        raw_validation_examples=len(raw_validation),
        train_examples=len(train),
        validation_examples=len(validation),
        excluded=dict(sorted(excluded.items())),
        train_rules=int(details["train_rules"]),
        validation_rules=int(details["validation_rules"]),
        rule_overlap=int(details["rule_overlap"]),
        train_input_conflicts=0,
        validation_input_conflicts=0,
        corpus_sha256=corpus_hash,
        split_sha256=split_hash,
    )
    return train, validation, audit
