"""Frozen, ID-free feature and oracle-state encodings for R24-v3.

These encoders are deterministic protocol code, not learned parameters.  They
parse only the current six-bit query/interaction and never receive a rule,
fingerprint, split, or task identifier except the C9 structured oracle helper.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import torch

from .tiny_cortex_v3 import EVENT_WRITES, FEATURE_DIM, READOUT_DIM
from .toy_catalog_v3 import ToyInteraction, ToyRule, ToyTask

FEATURE_SCHEMA_VERSION = "oczy/r24-toy-fixed-features/v3"


def _signed_bits(value: int) -> torch.Tensor:
    return torch.tensor(
        [1.0 if bit == "1" else -1.0 for bit in f"{value:06b}"],
        dtype=torch.float32,
    )


def structured_rule_state(rule: ToyRule) -> torch.Tensor:
    """Return the registered C9/Phase-A oracle vector ``r[64]``."""
    state = torch.zeros(READOUT_DIM, dtype=torch.float32)
    state[rule.rotation] = 1.0
    state[3:9] = _signed_bits(rule.mask)
    return state


def interaction_feature(
    interaction: ToyInteraction,
    *,
    correction_output: int | None = None,
) -> torch.Tensor:
    """Encode observation, identity attempt, correction, and outcome.

    ``correction_output`` supports C4 wrong-feedback intervention while leaving
    the recipient observation and attempted behavior untouched.
    """
    input_value = interaction.input_value
    output_value = interaction.output_value if correction_output is None else correction_output
    if not 0 <= output_value < 64:
        raise ValueError("correction_output must be a six-bit value")
    input_bits = _signed_bits(input_value)
    output_bits = _signed_bits(output_value)
    feature = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    feature[0:6] = input_bits  # observation
    feature[6:12] = input_bits  # deterministic identity attempt
    feature[12:18] = output_bits  # correction
    feature[18] = 1.0  # corrected outcome
    feature[19:25] = input_bits * output_bits
    feature[25] = input_value / 63.0
    feature[26] = output_value / 63.0
    feature[27] = 1.0  # event rather than query
    return feature


def query_feature(interaction: ToyInteraction) -> torch.Tensor:
    """Encode only the current query; no gold answer or rule is read."""
    feature = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    feature[0:6] = _signed_bits(interaction.input_value)
    feature[28] = 1.0
    feature[29] = interaction.input_value / 63.0
    return feature


def teaching_features(task: ToyTask, *, donor: ToyTask | None = None) -> torch.Tensor:
    """Return exactly three event features, optionally with donor corrections."""
    if len(task.teaching) != EVENT_WRITES:
        raise ValueError(f"task must have exactly {EVENT_WRITES} teaching interactions")
    if donor is not None:
        if len(donor.teaching) != EVENT_WRITES:
            raise ValueError("donor must have exactly three teaching interactions")
        if donor.rule_fingerprint == task.rule_fingerprint:
            raise ValueError("donor rule must differ from recipient")
        if [row.input_value for row in donor.teaching] != [
            row.input_value for row in task.teaching
        ]:
            raise ValueError("donor teaching inputs must match recipient inputs")
        corrections = [row.output_value for row in donor.teaching]
        if any(
            correction == row.output_value
            for correction, row in zip(corrections, task.teaching, strict=True)
        ):
            raise ValueError("C4 donor correction must differ on every teaching row")
    else:
        corrections = [row.output_value for row in task.teaching]
    return torch.stack(
        [
            interaction_feature(row, correction_output=correction)
            for row, correction in zip(task.teaching, corrections, strict=True)
        ]
    )


def batch_teaching_features(
    tasks: Sequence[ToyTask], *, donors: Sequence[ToyTask] | None = None
) -> torch.Tensor:
    if not tasks:
        raise ValueError("tasks must be non-empty")
    if donors is not None and len(donors) != len(tasks):
        raise ValueError("donors must match task batch length")
    return torch.stack(
        [
            teaching_features(task, donor=None if donors is None else donors[index])
            for index, task in enumerate(tasks)
        ]
    )


def batch_query_features(interactions: Sequence[ToyInteraction]) -> torch.Tensor:
    if not interactions:
        raise ValueError("interactions must be non-empty")
    return torch.stack([query_feature(interaction) for interaction in interactions])


def feature_definition() -> dict[str, object]:
    golden_rule = ToyRule(rotation=2, mask=37)
    golden_interaction = ToyInteraction(input_value=19, output_value=42)
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "event_writes": EVENT_WRITES,
        "readout_dim": READOUT_DIM,
        "event_layout": {
            "observation_input_bits": [0, 6],
            "identity_attempt_bits": [6, 12],
            "correction_output_bits": [12, 18],
            "corrected_outcome": 18,
            "input_output_products": [19, 25],
            "normalized_input": 25,
            "normalized_correction": 26,
            "event_marker": 27,
        },
        "query_layout": {
            "input_bits": [0, 6],
            "query_marker": 28,
            "normalized_input": 29,
        },
        "oracle_layout": {
            "rotation_one_hot": [0, 3],
            "mask_signed_bits": [3, 9],
        },
        "golden_vectors": {
            "structured_rule_state": structured_rule_state(golden_rule).tolist(),
            "interaction_feature": interaction_feature(golden_interaction).tolist(),
            "query_feature": query_feature(golden_interaction).tolist(),
        },
    }


def feature_definition_sha256() -> str:
    return hashlib.sha256(
        json.dumps(feature_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
