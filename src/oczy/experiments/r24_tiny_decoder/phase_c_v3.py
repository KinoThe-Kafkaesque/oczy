"""R24-v3 exact-three cortex training and C1-C9 paired evaluation.

Phase-C development trains only a 6,995-parameter cortex through one shared,
frozen Phase-A decoder.  Online episodes contain exactly three writes and one
consolidation; no optimizer step occurs inside an episode.  Sealed TEST is a
separate checkpoint-only command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .decoder import TinyDecoderConfig, TinySharedDecoder
from .phase_a_toy_v3 import (
    BASE_CONFIG as PHASE_A_BASE_CONFIG,
)
from .phase_a_toy_v3 import (
    CASE_OVERRIDES as PHASE_A_CASE_OVERRIDES,
)
from .phase_a_toy_v3 import (
    PHASE_A_V3_SCHEMA,
)
from .phase_a_toy_v3 import (
    suite_sha256 as phase_a_suite_sha256,
)
from .tiny_cortex_v3 import CortexState, TinyCortexV3
from .toy_catalog_v3 import ToyCatalog, ToyInteraction, ToyTask, build_toy_catalog_v3
from .toy_features_v3 import (
    batch_query_features,
    batch_teaching_features,
    feature_definition_sha256,
    structured_rule_state,
)
from .vocab import EOS_ID, encode_bytes, encode_with_eos

PHASE_C_V3_SCHEMA = "oczy/r24-toy-phase-c/v3"
TEST_AUTHORIZATION_SCHEMA = "oczy/r24-toy-phase-c-test-authorization/v3"
CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9")
SEED_TUPLES: tuple[dict[str, int], ...] = (
    {"init_seed": 11, "batch_seed": 501},
    {"init_seed": 29, "batch_seed": 529},
    {"init_seed": 47, "batch_seed": 547},
    {"init_seed": 71, "batch_seed": 571},
    {"init_seed": 97, "batch_seed": 597},
)


@dataclass(frozen=True, slots=True)
class PhaseCV3Config:
    catalog_seed: int = 24_003
    steps: int = 8_000
    task_batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.task_batch_size < 1:
            raise ValueError("steps and task_batch_size must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid optimizer configuration")


BASE_CONFIG = PhaseCV3Config()


@dataclass(frozen=True, slots=True)
class FrozenOrgan:
    decoder: TinySharedDecoder
    artifact: Mapping[str, Any]
    artifact_dir: Path
    weight_hash: str
    conditioning: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_frozen_organ(
    artifact_dir: str | Path,
    *,
    expected_conditioning: Literal["film", "additive"] | None = None,
    device: str | torch.device = "cpu",
) -> FrozenOrgan:
    """Strictly load and hash-check a Phase-A v3 shared decoder."""
    directory = Path(artifact_dir)
    artifact_path = directory / "artifact.json"
    decoder_path = directory / "decoder.pt"
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != PHASE_A_V3_SCHEMA:
        raise ValueError("unsupported Phase-A organ schema")
    if artifact.get("sealed_test_accessed") is not False:
        raise ValueError("Phase-A organ must not have accessed sealed TEST")
    if artifact.get("feature_definition_sha256") != feature_definition_sha256():
        raise ValueError("feature definition hash mismatch")
    expected_phase_a_suite = phase_a_suite_sha256()
    if artifact.get("phase_a_suite_sha256") != expected_phase_a_suite:
        raise ValueError("Phase-A suite/source hash mismatch")
    conditioning = artifact.get("config", {}).get("conditioning")
    if conditioning not in PHASE_A_CASE_OVERRIDES:
        raise ValueError("Phase-A conditioning is not registered")
    expected_config = asdict(PHASE_A_BASE_CONFIG) | PHASE_A_CASE_OVERRIDES[conditioning]
    if artifact.get("config") != expected_config:
        raise ValueError("Phase-A organ does not use its registered case config")
    if artifact.get("suite") != {
        "case": conditioning,
        "suite_sha256": expected_phase_a_suite,
    }:
        raise ValueError("Phase-A case/suite registration mismatch")
    decoder_config_artifact = artifact.get("decoder_config", {})
    expected_decoder_fields = {
        "version": "r24-tiny-decoder/v3",
        "d_model": expected_config["d_model"],
        "n_layers": expected_config["n_layers"],
        "conditioning": conditioning,
        "deep_film": expected_config["deep_conditioning"],
        "dropout": expected_config["dropout"],
        "r_dim": 64,
    }
    if any(
        decoder_config_artifact.get(key) != value
        for key, value in expected_decoder_fields.items()
    ):
        raise ValueError("Phase-A decoder/config conditioning mismatch")
    expected_file = artifact.get("files", {}).get("decoder.pt", {}).get("sha256")
    if not isinstance(expected_file, str) or _sha256_file(decoder_path) != expected_file:
        raise ValueError("decoder.pt file hash mismatch")
    payload = torch.load(decoder_path, map_location=device, weights_only=True)
    if payload.get("schema_version") != PHASE_A_V3_SCHEMA:
        raise ValueError("decoder payload schema mismatch")
    if payload.get("phase_a_suite_sha256") != expected_phase_a_suite:
        raise ValueError("decoder payload suite/source mismatch")
    if payload.get("phase_a_config") != expected_config or payload.get("case") != conditioning:
        raise ValueError("decoder payload Phase-A case/config mismatch")
    if payload.get("catalog_manifest_sha256") != artifact.get("catalog_manifest_sha256"):
        raise ValueError("decoder catalog hash mismatch")
    if payload.get("feature_definition_sha256") != feature_definition_sha256():
        raise ValueError("decoder feature hash mismatch")
    if payload.get("decoder_config") != decoder_config_artifact:
        raise ValueError("decoder payload/artifact config mismatch")
    config = TinyDecoderConfig(**payload["decoder_config"])
    if config.version != "r24-tiny-decoder/v3":
        raise ValueError("decoder protocol version mismatch")
    if expected_conditioning is not None and config.conditioning != expected_conditioning:
        raise ValueError("decoder conditioning mismatch")
    decoder = TinySharedDecoder(config).to(device)
    decoder.load_state_dict(payload["decoder_state_dict"])
    decoder.eval()
    observed_hash = decoder.parameter_hash()
    if observed_hash != payload.get("weight_hash") or observed_hash != artifact.get("weight_hash"):
        raise ValueError("decoder parameter hash mismatch")
    frozen_hash = decoder.freeze()
    decoder.eval()
    if frozen_hash != observed_hash or any(p.requires_grad for p in decoder.parameters()):
        raise RuntimeError("decoder freeze failed")
    return FrozenOrgan(
        decoder=decoder,
        artifact=artifact,
        artifact_dir=directory,
        weight_hash=observed_hash,
        conditioning=config.conditioning,
    )


def _trim(ids: list[int]) -> list[int]:
    return ids[: ids.index(EOS_ID) + 1] if EOS_ID in ids else ids


def _donors_for(catalog: ToyCatalog, tasks: Sequence[ToyTask]) -> list[ToyTask]:
    indexed = {task.rule_fingerprint: task for task in catalog.all_tasks}
    return [indexed[task.donor_rule_fingerprint] for task in tasks]


def _consolidated_states(
    cortex: TinyCortexV3,
    tasks: Sequence[ToyTask],
    *,
    donors: Sequence[ToyTask] | None = None,
) -> tuple[CortexState, dict[str, Any]]:
    events = batch_teaching_features(tasks, donors=donors).to(
        next(cortex.parameters()).device
    )
    before_count = events.shape[1]
    state = cortex.initial_state(len(tasks))
    written = cortex.write_three(state, events)
    consolidated = cortex.consolidate(written.state).state
    if torch.count_nonzero(consolidated.fast).item() != 0:
        raise RuntimeError("consolidation did not clear fast state")
    # No raw interaction or feature tensor is returned or serialized.
    del events
    return consolidated, {
        "writer_calls_per_task": before_count,
        "consolidation_calls_per_task": 1,
        "raw_trace_count_after": 0,
        "feature_trace_count_after": 0,
        "persistent_bytes_per_task": cortex.post_consolidation_bytes_per_task(
            consolidated
        ),
    }


def _teacher_batch(
    tasks: Sequence[ToyTask], rng: random.Random
) -> tuple[list[ToyInteraction], torch.Tensor, torch.Tensor]:
    interactions = [task.heldout[rng.randrange(len(task.heldout))] for task in tasks]
    query = torch.tensor(
        [encode_bytes(item.query_text) for item in interactions], dtype=torch.long
    )
    answer = torch.tensor(
        [encode_with_eos(item.answer_text) for item in interactions], dtype=torch.long
    )
    return interactions, query, answer


def _condition_metrics(correct: int, total: int) -> dict[str, int | float]:
    return {"correct": correct, "total": total, "exact_accuracy": correct / total}


def evaluate_c1_c9(
    organ: FrozenOrgan,
    trained: TinyCortexV3,
    random_cortex: TinyCortexV3,
    tasks: Sequence[ToyTask],
    *,
    catalog: ToyCatalog,
    split: Literal["cortex_meta_dev", "sealed_test"],
) -> dict[str, Any]:
    """Evaluate paired C1-C9 without optimization or parameter mutation."""
    if not tasks:
        raise ValueError("evaluation tasks must be non-empty")
    decoder = organ.decoder
    trained.eval()
    random_cortex.eval()
    organ_before = decoder.parameter_hash()
    theta_before = trained.parameter_hash()
    theta0_before = random_cortex.parameter_hash()
    donors = _donors_for(catalog, tasks)
    with torch.inference_mode():
        c2_state, c2_trace = _consolidated_states(random_cortex, tasks)
        c3_state, c3_trace = _consolidated_states(trained, tasks)
        c4_state, c4_trace = _consolidated_states(trained, tasks, donors=donors)
        c5_state = trained.zero_state(c3_state)
        index = {task.rule_fingerprint: offset for offset, task in enumerate(tasks)}
        donor_indices = [index[task.donor_rule_fingerprint] for task in tasks]
        c6_state = CortexState(
            fast=torch.zeros_like(c3_state.fast),
            slow=c3_state.slow[donor_indices].clone(),
        )
        correct = {name: 0 for name in CONDITIONS}
        per_task = {
            task.rule_fingerprint: {name: 0 for name in CONDITIONS} for task in tasks
        }
        paired_rows: list[dict[str, bool]] = []
        c7_flips = 0
        c7_donor_correct = 0
        total = 0
        for heldout_index in range(len(tasks[0].heldout)):
            interactions = [task.heldout[heldout_index] for task in tasks]
            if any(
                item.input_value != interactions[0].input_value for item in interactions
            ):
                raise ValueError("paired evaluation requires common query bytes")
            query = torch.tensor(
                [encode_bytes(item.query_text) for item in interactions],
                dtype=torch.long,
                device=next(decoder.parameters()).device,
            )
            query_features = batch_query_features(interactions).to(query.device)
            oracle_state = torch.stack(
                [structured_rule_state(task.rule) for task in tasks]
            ).to(query.device)
            states = {
                "C1": torch.zeros((len(tasks), 64), device=query.device),
                "C2": random_cortex.read(c2_state, query_features),
                "C3": trained.read(c3_state, query_features),
                "C4": trained.read(c4_state, query_features),
                "C5": trained.read(c5_state, query_features),
                "C6": trained.read(c6_state, query_features),
                "C9": oracle_state,
            }
            generated = {
                name: decoder.generate_greedy(query, state, max_new_tokens=3).tolist()
                for name, state in states.items()
            }
            for task_index, (task, interaction) in enumerate(zip(tasks, interactions, strict=True)):
                gold = encode_with_eos(interaction.answer_text)
                outcomes: dict[str, bool] = {}
                for name in ("C1", "C2", "C3", "C4", "C5", "C6", "C9"):
                    outcome = _trim(generated[name][task_index]) == gold
                    outcomes[name] = outcome
                    correct[name] += int(outcome)
                    per_task[task.rule_fingerprint][name] += int(outcome)
                # C8: byte-only nearest-query row copy, deterministic ties; no rule solver.
                nearest = min(
                    task.teaching,
                    key=lambda row: (
                        sum(
                            abs(left - right)
                            for left, right in zip(
                                row.query_bytes, interaction.query_bytes, strict=True
                            )
                        ),
                        row.query_bytes,
                    ),
                )
                outcomes["C8"] = nearest.answer_text == interaction.answer_text
                correct["C8"] += int(outcomes["C8"])
                per_task[task.rule_fingerprint]["C8"] += int(outcomes["C8"])
                own = _trim(generated["C3"][task_index])
                donor = _trim(generated["C6"][task_index])
                c7_flips += int(own != donor)
                donor_task = donors[task_index]
                donor_gold = encode_with_eos(
                    donor_task.heldout[heldout_index].answer_text
                )
                c7_donor_correct += int(own != donor and donor == donor_gold)
                paired_rows.append(outcomes)
                total += 1
    metrics = {name: _condition_metrics(correct[name], total) for name in CONDITIONS}
    paired: dict[str, dict[str, int]] = {}
    for control in ("C1", "C2", "C4", "C5", "C6", "C8"):
        counts = {"c3_only": 0, "control_only": 0, "both": 0, "neither": 0}
        for row in paired_rows:
            c3_ok, control_ok = row["C3"], row[control]
            key = (
                "both"
                if c3_ok and control_ok
                else "c3_only"
                if c3_ok
                else "control_only"
                if control_ok
                else "neither"
            )
            counts[key] += 1
        paired[control] = counts
    denominator = len(tasks[0].heldout)
    per_task_records = {
        fingerprint: {
            name: {
                "correct": count,
                "total": denominator,
                "accuracy": count / denominator,
            }
            for name, count in values.items()
        }
        for fingerprint, values in sorted(per_task.items())
    }
    if decoder.parameter_hash() != organ_before:
        raise RuntimeError("frozen organ mutated during evaluation")
    if trained.parameter_hash() != theta_before or random_cortex.parameter_hash() != theta0_before:
        raise RuntimeError("cortex parameters mutated during evaluation")
    return {
        "schema_version": PHASE_C_V3_SCHEMA,
        "split": split,
        "conditions": metrics,
        "paired_controls": paired,
        "per_task": per_task_records,
        "C7": {
            "same_query_total": total,
            "state_conditioned_flip_count": c7_flips,
            "state_conditioned_flip_rate": c7_flips / total,
            "flip_to_donor_gold_count": c7_donor_correct,
            "flip_to_donor_gold_rate": c7_donor_correct / total,
        },
        "C8": {
            "stored_rows_per_task": 3,
            "logical_payload_bytes_max": max(
                sum(len(row.query_bytes) + len(row.answer_bytes) for row in task.teaching)
                for task in tasks
            ),
            "persistent_byte_budget": 1_024,
            "solver_access": False,
        },
        "trace_audits": {"C2": c2_trace, "C3": c3_trace, "C4": c4_trace},
        "organ_hash_before_after": [organ_before, decoder.parameter_hash()],
        "theta_hash_before_after": [theta_before, trained.parameter_hash()],
        "theta0_hash_before_after": [theta0_before, random_cortex.parameter_hash()],
    }


def train_dev(
    organ_dir: str | Path,
    *,
    seed_index: int,
    output_dir: str | Path,
    config: PhaseCV3Config = BASE_CONFIG,
) -> dict[str, Any]:
    if seed_index < 0 or seed_index >= len(SEED_TUPLES):
        raise IndexError("seed_index out of range")
    seed_tuple = SEED_TUPLES[seed_index]
    organ = load_frozen_organ(organ_dir, device=config.device)
    if organ.artifact.get("phase_c_organ_gate") is not True:
        raise ValueError("Phase-C is blocked because the Phase-A organ gate failed")
    catalog = build_toy_catalog_v3(root_seed=config.catalog_seed)
    if organ.artifact.get("catalog_manifest_sha256") != catalog.manifest_sha256:
        raise ValueError("organ/catalog manifest mismatch")
    trained = TinyCortexV3(init_seed=seed_tuple["init_seed"]).to(config.device)
    random_cortex = TinyCortexV3(init_seed=seed_tuple["init_seed"]).to(config.device)
    theta0_hash = trained.parameter_hash()
    if random_cortex.parameter_hash() != theta0_hash:
        raise RuntimeError("paired theta0 initialization mismatch")
    theta0_state = {name: value.detach().cpu().clone() for name, value in trained.state_dict().items()}
    optimizer = torch.optim.AdamW(
        trained.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} & {
        id(parameter) for parameter in organ.decoder.parameters()
    }:
        raise RuntimeError("optimizer includes frozen organ parameter")
    rng = random.Random(seed_tuple["batch_seed"])
    tasks = list(catalog.cortex_meta_train)
    organ_hash = organ.weight_hash
    trace: list[dict[str, float | int]] = []
    trace_audit: dict[str, Any] = {}
    order_hash = hashlib.sha256()
    for step in range(config.steps):
        trained.train()
        batch = rng.sample(tasks, config.task_batch_size)
        for task in batch:
            order_hash.update(task.rule_fingerprint.encode())
        state, trace_audit = _consolidated_states(trained, batch)
        interactions, query, answer = _teacher_batch(batch, rng)
        for interaction in interactions:
            order_hash.update(interaction.query_bytes)
            order_hash.update(b"\0")
            order_hash.update(interaction.answer_bytes)
            order_hash.update(b"\0")
        query = query.to(config.device)
        answer = answer.to(config.device)
        query_features = batch_query_features(interactions).to(config.device)
        readout = trained.read(state, query_features)
        loss = organ.decoder.teacher_forced_loss(query, readout, answer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trained.parameters(), config.grad_clip)
        if any(parameter.grad is not None for parameter in organ.decoder.parameters()):
            raise RuntimeError("frozen decoder accumulated a parameter gradient")
        optimizer.step()
        if organ.decoder.parameter_hash() != organ_hash:
            raise RuntimeError("frozen decoder changed during outer training")
        if step == 0 or (step + 1) % 1_000 == 0 or step + 1 == config.steps:
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
            }
            trace.append(row)
            print(
                f"R24-v3 Phase C {organ.conditioning} seed={seed_index} "
                f"{step + 1}/{config.steps} loss={loss.item():.6f}",
                flush=True,
            )
        del interactions, query_features, readout, state
    trained.eval()
    theta_hash = trained.parameter_hash()
    dev = evaluate_c1_c9(
        organ,
        trained,
        random_cortex,
        catalog.cortex_meta_dev,
        catalog=catalog,
        split="cortex_meta_dev",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "cortex.pt"
    torch.save(
        {
            "schema_version": PHASE_C_V3_SCHEMA,
            "phase_c_suite_sha256": suite_sha256(),
            "phase_c_config": asdict(config),
            "seed_index": seed_index,
            "seed_tuple": seed_tuple,
            "conditioning": organ.conditioning,
            "organ_weight_hash": organ.weight_hash,
            "catalog_manifest_sha256": catalog.manifest_sha256,
            "feature_definition_sha256": feature_definition_sha256(),
            "theta0_state_dict": theta0_state,
            "theta0_hash": theta0_hash,
            "theta_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in trained.state_dict().items()
            },
            "theta_hash": theta_hash,
        },
        checkpoint_path,
    )
    artifact: dict[str, Any] = {
        "schema_version": PHASE_C_V3_SCHEMA,
        "phase_c_suite_sha256": suite_sha256(),
        "stage": "train_dev",
        "config": asdict(config),
        "seed_index": seed_index,
        "seed_tuple": seed_tuple,
        "conditioning": organ.conditioning,
        "organ_weight_hash": organ.weight_hash,
        "organ_phase_c_gate": bool(organ.artifact.get("phase_c_organ_gate")),
        "catalog_manifest_sha256": catalog.manifest_sha256,
        "feature_definition_sha256": feature_definition_sha256(),
        "theta_parameter_count": trained.parameter_count(),
        "theta0_hash": theta0_hash,
        "theta_hash": theta_hash,
        "batch_order_hash": order_hash.hexdigest(),
        "trace": trace,
        "last_trace_audit": trace_audit,
        "dev": dev,
        "sealed_test_accessed": False,
        "files": {
            "cortex.pt": {
                "sha256": _sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            }
        },
    }
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_run_artifact(
    artifact: Mapping[str, Any], *, stage: Literal["dev", "test"]
) -> Mapping[str, Any]:
    if artifact.get("schema_version") != PHASE_C_V3_SCHEMA:
        raise ValueError("Phase-C artifact schema mismatch")
    if artifact.get("phase_c_suite_sha256") != suite_sha256():
        raise ValueError("Phase-C suite/source hash mismatch")
    expected_stage = "train_dev" if stage == "dev" else "test"
    if artifact.get("stage") != expected_stage:
        raise ValueError(f"expected {expected_stage} artifact")
    if artifact.get("config") != asdict(BASE_CONFIG):
        raise ValueError("artifact does not use the registered Phase-C config")
    seed_index = artifact.get("seed_index")
    if isinstance(seed_index, bool) or not isinstance(seed_index, int):
        raise ValueError("seed index must be an integer")
    if seed_index not in range(5) or artifact.get("seed_tuple") != SEED_TUPLES[seed_index]:
        raise ValueError("artifact seed tuple mismatch")
    if artifact.get("conditioning") not in ("film", "additive"):
        raise ValueError("artifact conditioning mismatch")
    catalog = build_toy_catalog_v3(root_seed=BASE_CONFIG.catalog_seed)
    if artifact.get("catalog_manifest_sha256") != catalog.manifest_sha256:
        raise ValueError("artifact catalog manifest mismatch")
    if artifact.get("feature_definition_sha256") != feature_definition_sha256():
        raise ValueError("artifact feature definition mismatch")
    if stage == "dev":
        if artifact.get("sealed_test_accessed") is not False:
            raise ValueError("DEV artifact touched sealed TEST")
        if artifact.get("organ_phase_c_gate") is not True:
            raise ValueError("Phase-A organ gate did not authorize Phase-C")
        result = artifact.get("dev")
        expected_split = "cortex_meta_dev"
        expected_task_count = len(catalog.cortex_meta_dev)
    else:
        if artifact.get("sealed_test_accessed") is not True:
            raise ValueError("TEST artifact access flag mismatch")
        if artifact.get("optimizer_steps") != 0 or artifact.get("backward_calls") != 0:
            raise ValueError("TEST artifact reports optimization or backward")
        result = artifact.get("test")
        expected_split = "sealed_test"
        expected_task_count = len(catalog.sealed_test)
    if not isinstance(result, Mapping):
        raise ValueError("artifact result is missing")
    if result.get("schema_version") != PHASE_C_V3_SCHEMA:
        raise ValueError("result schema mismatch")
    if result.get("split") != expected_split:
        raise ValueError("result split mismatch")
    expected_total = expected_task_count * len(catalog.cortex_meta_dev[0].heldout)
    conditions = result.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        raise ValueError("condition key mismatch")
    for name in CONDITIONS:
        metric = conditions[name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"{name} metric is malformed")
        correct = metric.get("correct")
        total = metric.get("total")
        accuracy = metric.get("exact_accuracy")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or not isinstance(total, int)
            or total != expected_total
            or correct < 0
            or correct > total
            or not isinstance(accuracy, (int, float))
            or abs(float(accuracy) - correct / total) > 1e-12
        ):
            raise ValueError(f"{name} numerator/denominator mismatch")
    per_task = result.get("per_task")
    if not isinstance(per_task, Mapping) or len(per_task) != expected_task_count:
        raise ValueError("per-task record count mismatch")
    heldout_count = len(catalog.cortex_meta_dev[0].heldout)
    per_task_sums = {name: 0 for name in CONDITIONS}
    for record in per_task.values():
        if not isinstance(record, Mapping) or set(record) != set(CONDITIONS):
            raise ValueError("per-task condition mismatch")
        for metric in record.values():
            if not isinstance(metric, Mapping) or metric.get("total") != heldout_count:
                raise ValueError("per-task denominator mismatch")
            correct = metric.get("correct")
            accuracy = metric.get("accuracy")
            if (
                isinstance(correct, bool)
                or not isinstance(correct, int)
                or not 0 <= correct <= heldout_count
                or not isinstance(accuracy, (int, float))
                or not math.isfinite(float(accuracy))
                or abs(float(accuracy) - correct / heldout_count) > 1e-12
            ):
                raise ValueError("per-task numerator mismatch")
        for name in CONDITIONS:
            per_task_sums[name] += int(record[name]["correct"])
    if any(
        per_task_sums[name] != conditions[name]["correct"] for name in CONDITIONS
    ):
        raise ValueError("per-task and aggregate numerators disagree")
    paired = result.get("paired_controls")
    if not isinstance(paired, Mapping):
        raise ValueError("paired controls missing")
    for control in ("C1", "C2", "C4", "C5", "C6", "C8"):
        counts = paired.get(control)
        expected_paired_keys = {"c3_only", "control_only", "both", "neither"}
        if (
            not isinstance(counts, Mapping)
            or set(counts) != expected_paired_keys
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            )
            or sum(counts.values()) != expected_total
            or counts["both"] + counts["c3_only"] != conditions["C3"]["correct"]
            or counts["both"] + counts["control_only"]
            != conditions[control]["correct"]
        ):
            raise ValueError(f"paired {control} numerator/denominator mismatch")
    c7 = result.get("C7")
    if not isinstance(c7, Mapping) or c7.get("same_query_total") != expected_total:
        raise ValueError("C7 denominator mismatch")
    for count_key, rate_key in (
        ("state_conditioned_flip_count", "state_conditioned_flip_rate"),
        ("flip_to_donor_gold_count", "flip_to_donor_gold_rate"),
    ):
        count = c7.get(count_key)
        rate = c7.get(rate_key)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= expected_total
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or abs(float(rate) - count / expected_total) > 1e-12
        ):
            raise ValueError(f"C7 {count_key} numerator/rate mismatch")
    return result


def aggregate_runs(
    artifacts: Sequence[Mapping[str, Any]], *, stage: Literal["dev", "test"]
) -> dict[str, Any]:
    if len(artifacts) != 5:
        raise ValueError("aggregation requires exactly five registered seeds")
    indexes = {artifact.get("seed_index") for artifact in artifacts}
    if indexes != set(range(5)):
        raise ValueError("seed indexes must be exactly 0..4")
    results = [_validate_run_artifact(artifact, stage=stage) for artifact in artifacts]
    organ_hashes = {str(artifact["organ_weight_hash"]) for artifact in artifacts}
    theta_hashes = {str(artifact["theta_hash"]) for artifact in artifacts}
    conditionings = {str(artifact["conditioning"]) for artifact in artifacts}
    catalog_hashes = {str(artifact["catalog_manifest_sha256"]) for artifact in artifacts}
    feature_hashes = {str(artifact["feature_definition_sha256"]) for artifact in artifacts}
    if len(organ_hashes) != 1 or len(theta_hashes) != 5:
        raise ValueError("organ must be shared and theta seeds must be distinct")
    if len(conditionings) != 1 or len(catalog_hashes) != 1 or len(feature_hashes) != 1:
        raise ValueError("aggregation mixed conditioning or protocol identities")
    if stage == "test":
        authorization_hashes = {
            str(artifact.get("test_authorization_sha256")) for artifact in artifacts
        }
        dev_decision_hashes = {
            str(artifact.get("dev_decision_sha256")) for artifact in artifacts
        }
        if (
            len(authorization_hashes) != 1
            or len(dev_decision_hashes) != 1
            or any(len(value) != 64 for value in authorization_hashes | dev_decision_hashes)
        ):
            raise ValueError("TEST aggregation mixed authorization or DEV decisions")
    means = {
        name: sum(float(result["conditions"][name]["exact_accuracy"]) for result in results)
        / len(results)
        for name in CONDITIONS
    }
    deltas = {
        control: means["C3"] - means[control]
        for control in ("C1", "C2", "C4", "C5", "C6", "C8")
    }
    positive_seed_counts = {
        control: sum(
            float(result["conditions"]["C3"]["exact_accuracy"])
            > float(result["conditions"][control]["exact_accuracy"])
            for result in results
        )
        for control in ("C1", "C2", "C4", "C5", "C6", "C8")
    }
    mean_flip_rate = sum(
        float(result["C7"]["state_conditioned_flip_rate"]) for result in results
    ) / len(results)
    organ_gate = means["C9"] >= 0.95 and (
        stage == "test" or all(artifact.get("organ_phase_c_gate") is True for artifact in artifacts)
    )
    primary = deltas["C2"] >= 0.02 and deltas["C1"] >= 0.02
    causal = all(deltas[name] >= 0.02 for name in ("C4", "C5", "C6", "C8"))
    robust = positive_seed_counts["C1"] == 5 and positive_seed_counts["C2"] == 5
    passed = bool(organ_gate and primary and causal and robust and mean_flip_rate >= 0.02)
    if stage == "dev":
        decision = "authorize_sealed_test" if passed else "do_not_authorize_sealed_test"
    else:
        decision = "accept_toy_existence" if passed else "do_not_accept"
    return {
        "schema_version": PHASE_C_V3_SCHEMA,
        "phase_c_suite_sha256": suite_sha256(),
        "stage": stage,
        "conditioning": next(iter(conditionings)),
        "organ_weight_hash": next(iter(organ_hashes)),
        "catalog_manifest_sha256": next(iter(catalog_hashes)),
        "feature_definition_sha256": next(iter(feature_hashes)),
        "mean_condition_accuracy": means,
        "mean_c3_deltas": deltas,
        "positive_seed_counts": positive_seed_counts,
        "mean_c7_state_conditioned_flip_rate": mean_flip_rate,
        "organ_gate": organ_gate,
        "primary_gate": primary,
        "causal_controls_gate": causal,
        "seed_robustness_gate": robust,
        "passed": passed,
        "decision": decision,
    }


def _validate_checkpoint_payload(
    directory: Path, artifact: Mapping[str, Any]
) -> Mapping[str, Any]:
    payload = torch.load(directory / "cortex.pt", map_location="cpu", weights_only=True)
    exact_fields = {
        "schema_version": PHASE_C_V3_SCHEMA,
        "phase_c_suite_sha256": suite_sha256(),
        "phase_c_config": artifact.get("config"),
        "seed_index": artifact.get("seed_index"),
        "seed_tuple": artifact.get("seed_tuple"),
        "conditioning": artifact.get("conditioning"),
        "organ_weight_hash": artifact.get("organ_weight_hash"),
        "catalog_manifest_sha256": artifact.get("catalog_manifest_sha256"),
        "feature_definition_sha256": artifact.get("feature_definition_sha256"),
        "theta0_hash": artifact.get("theta0_hash"),
        "theta_hash": artifact.get("theta_hash"),
    }
    for key, expected in exact_fields.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint/artifact {key} mismatch")
    trained = TinyCortexV3(init_seed=int(payload["seed_tuple"]["init_seed"]))
    trained.load_state_dict(payload["theta_state_dict"])
    theta0 = TinyCortexV3(init_seed=int(payload["seed_tuple"]["init_seed"]))
    theta0.load_state_dict(payload["theta0_state_dict"])
    if trained.parameter_hash() != payload["theta_hash"]:
        raise ValueError("checkpoint trained parameter hash mismatch")
    if theta0.parameter_hash() != payload["theta0_hash"]:
        raise ValueError("checkpoint theta0 parameter hash mismatch")
    return payload


def create_test_authorization(
    artifacts: Sequence[Mapping[str, Any]],
    checkpoint_dirs: Sequence[str | Path],
    *,
    signoff_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a passing five-seed DEV gate to five on-disk artifacts/checkpoints."""
    if len(checkpoint_dirs) != 5 or len(artifacts) != 5:
        raise ValueError("TEST authorization requires five artifacts and checkpoints")
    decision = aggregate_runs(artifacts, stage="dev")
    if decision["passed"] is not True:
        raise ValueError("cannot authorize TEST because the DEV gate failed")
    if not signoff_id:
        raise ValueError("signoff_id must be non-empty")
    output = Path(output_path).resolve()
    root = output.parent
    checkpoint_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    relative_dirs: dict[str, str] = {}
    for artifact, directory_value in zip(artifacts, checkpoint_dirs, strict=True):
        directory = Path(directory_value).resolve()
        try:
            relative = directory.relative_to(root)
        except ValueError as error:
            raise ValueError("checkpoint directories must be below authorization root") from error
        disk_artifact_path = directory / "artifact.json"
        disk_artifact = json.loads(disk_artifact_path.read_text())
        if disk_artifact != dict(artifact):
            raise ValueError("passed DEV artifact differs from artifact.json")
        checkpoint_path = directory / "cortex.pt"
        observed = _sha256_file(checkpoint_path)
        expected = artifact["files"]["cortex.pt"]["sha256"]
        if observed != expected:
            raise ValueError("DEV checkpoint hash does not match its artifact")
        _validate_checkpoint_payload(directory, artifact)
        seed_key = str(int(artifact["seed_index"]))
        if seed_key in checkpoint_hashes:
            raise ValueError("duplicate checkpoint seed")
        checkpoint_hashes[seed_key] = observed
        artifact_hashes[seed_key] = _sha256_file(disk_artifact_path)
        relative_dirs[seed_key] = relative.as_posix()
    if set(checkpoint_hashes) != {str(index) for index in range(5)}:
        raise ValueError("checkpoint seeds must be exactly 0..4")
    authorization = {
        "schema_version": TEST_AUTHORIZATION_SCHEMA,
        "phase_c_suite_sha256": suite_sha256(),
        "dev_decision": decision,
        "dev_decision_sha256": _canonical_sha256(decision),
        "conditioning": decision["conditioning"],
        "organ_weight_hash": decision["organ_weight_hash"],
        "checkpoint_dir_by_seed": relative_dirs,
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "artifact_sha256_by_seed": artifact_hashes,
        "signoff_id": signoff_id,
    }
    output.write_text(json.dumps(authorization, indent=2, sort_keys=True))
    return authorization


def _authorized_dev_runs(
    authorization_path: Path, authorization: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Path]]:
    root = authorization_path.resolve().parent
    directories = authorization.get("checkpoint_dir_by_seed")
    checkpoint_hashes = authorization.get("checkpoint_sha256_by_seed")
    artifact_hashes = authorization.get("artifact_sha256_by_seed")
    expected_keys = {str(index) for index in range(5)}
    if (
        not isinstance(directories, Mapping)
        or not isinstance(checkpoint_hashes, Mapping)
        or not isinstance(artifact_hashes, Mapping)
        or set(directories) != expected_keys
        or set(checkpoint_hashes) != expected_keys
        or set(artifact_hashes) != expected_keys
    ):
        raise ValueError("TEST authorization must bind five complete DEV runs")
    artifacts: list[Mapping[str, Any]] = []
    resolved: dict[str, Path] = {}
    for seed_key in sorted(expected_keys, key=int):
        relative = Path(str(directories[seed_key]))
        if relative.is_absolute():
            raise ValueError("authorized checkpoint path must be relative")
        directory = (root / relative).resolve()
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise ValueError("authorized checkpoint escaped authorization root") from error
        artifact_path = directory / "artifact.json"
        checkpoint_path = directory / "cortex.pt"
        if _sha256_file(artifact_path) != artifact_hashes[seed_key]:
            raise ValueError("authorized DEV artifact file hash mismatch")
        if _sha256_file(checkpoint_path) != checkpoint_hashes[seed_key]:
            raise ValueError("authorized DEV checkpoint file hash mismatch")
        artifact = json.loads(artifact_path.read_text())
        if str(artifact.get("seed_index")) != seed_key:
            raise ValueError("authorized DEV seed mismatch")
        if artifact.get("files", {}).get("cortex.pt", {}).get("sha256") != checkpoint_hashes[seed_key]:
            raise ValueError("DEV artifact/checkpoint hash mismatch")
        _validate_checkpoint_payload(directory, artifact)
        artifacts.append(artifact)
        resolved[seed_key] = directory
    return artifacts, resolved


def evaluate_test(
    organ_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    authorization_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Checkpoint-only TEST after reaggregating five hash-bound DEV runs."""
    organ = load_frozen_organ(organ_dir)
    authorization_file = Path(authorization_path)
    authorization = json.loads(authorization_file.read_text())
    if authorization.get("schema_version") != TEST_AUTHORIZATION_SCHEMA:
        raise ValueError("TEST authorization schema mismatch")
    if authorization.get("phase_c_suite_sha256") != suite_sha256():
        raise ValueError("TEST authorization suite mismatch")
    if not authorization.get("signoff_id"):
        raise ValueError("TEST authorization signoff is missing")
    dev_artifacts, authorized_dirs = _authorized_dev_runs(
        authorization_file, authorization
    )
    decision = aggregate_runs(dev_artifacts, stage="dev")
    if decision.get("passed") is not True or decision.get("decision") != "authorize_sealed_test":
        raise ValueError("five-seed DEV gate did not authorize TEST")
    if authorization.get("dev_decision") != decision:
        raise ValueError("recorded DEV decision mismatch")
    if authorization.get("dev_decision_sha256") != _canonical_sha256(decision):
        raise ValueError("DEV decision hash mismatch")
    if authorization.get("conditioning") != organ.conditioning:
        raise ValueError("TEST authorization conditioning mismatch")
    if authorization.get("organ_weight_hash") != organ.weight_hash:
        raise ValueError("TEST authorization organ mismatch")
    directory = Path(checkpoint_dir).resolve()
    dev_artifact = json.loads((directory / "artifact.json").read_text())
    seed_key = str(dev_artifact.get("seed_index"))
    if seed_key not in authorized_dirs or directory != authorized_dirs[seed_key]:
        raise ValueError("checkpoint directory is absent from TEST authorization")
    payload = _validate_checkpoint_payload(directory, dev_artifact)
    if payload.get("organ_weight_hash") != organ.weight_hash:
        raise ValueError("checkpoint/organ mismatch")
    trained = TinyCortexV3(init_seed=int(payload["seed_tuple"]["init_seed"]))
    trained.load_state_dict(payload["theta_state_dict"])
    random_cortex = TinyCortexV3(init_seed=int(payload["seed_tuple"]["init_seed"]))
    random_cortex.load_state_dict(payload["theta0_state_dict"])
    if trained.parameter_hash() != payload["theta_hash"]:
        raise ValueError("trained theta hash mismatch")
    if random_cortex.parameter_hash() != payload["theta0_hash"]:
        raise ValueError("theta0 hash mismatch")
    catalog = build_toy_catalog_v3()
    if payload.get("catalog_manifest_sha256") != catalog.manifest_sha256:
        raise ValueError("checkpoint catalog mismatch")
    # This is the first and only point at which this command accesses sealed tasks.
    result = evaluate_c1_c9(
        organ,
        trained,
        random_cortex,
        catalog.sealed_test,
        catalog=catalog,
        split="sealed_test",
    )
    artifact = {
        "schema_version": PHASE_C_V3_SCHEMA,
        "phase_c_suite_sha256": suite_sha256(),
        "stage": "test",
        "config": asdict(BASE_CONFIG),
        "seed_index": int(payload["seed_index"]),
        "seed_tuple": payload["seed_tuple"],
        "conditioning": organ.conditioning,
        "organ_weight_hash": organ.weight_hash,
        "feature_definition_sha256": feature_definition_sha256(),
        "theta0_hash": random_cortex.parameter_hash(),
        "theta_hash": trained.parameter_hash(),
        "catalog_manifest_sha256": catalog.manifest_sha256,
        "test_authorization_sha256": _sha256_file(authorization_file),
        "dev_decision_sha256": _canonical_sha256(decision),
        "test": result,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "sealed_test_accessed": True,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact


def implementation_file_sha256() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "vocab.py",
        "decoder.py",
        "phase_a_toy_v3.py",
        "phase_c_v3.py",
        "tiny_cortex_v3.py",
        "toy_catalog_v3.py",
        "toy_features_v3.py",
    )
    return {name: _sha256_file(root / name) for name in names}


def suite_definition() -> dict[str, Any]:
    return {
        "schema_version": PHASE_C_V3_SCHEMA,
        "implementation_file_sha256": implementation_file_sha256(),
        "config": asdict(BASE_CONFIG),
        "seed_tuples": list(SEED_TUPLES),
        "conditions": {
            "C1": "decoder with r=zeros",
            "C2": "paired theta0 after the same three correct interactions",
            "C3": "meta-trained theta after exactly three correct interactions",
            "C4": "recipient observations/attempts with in-split donor corrections",
            "C5": "trained cortex with post-consolidation F/S zeroed",
            "C6": "trained cortex with donor consolidated state",
            "C7": "identical query bytes under own versus donor state",
            "C8": "nearest-query-byte teaching-row copy retrieval, no solver",
            "C9": "structured full-rule oracle state",
        },
        "dev_gate": {
            "C9_min": 0.95,
            "mean_C3_minus_C2_min": 0.02,
            "mean_C3_minus_C1_min": 0.02,
            "mean_C3_minus_each_C4_C5_C6_C8_min": 0.02,
            "C3_gt_C1_and_C2_seeds": 5,
            "mean_C7_flip_rate_min": 0.02,
        },
        "test_policy": "checkpoint-only sealed TEST after a separately collected five-seed DEV gate",
        "test_authorization": {
            "schema_version": TEST_AUTHORIZATION_SCHEMA,
            "requires_passing_dev_gate": True,
            "binds_suite_organ_and_five_checkpoint_hashes": True,
        },
        "online_contract": {
            "writes": 3,
            "consolidations": 1,
            "optimizer_steps_inside_episode": 0,
            "persistent_bytes_per_task": 1024,
            "theta_parameter_budget": 10240,
        },
    }


def suite_sha256() -> str:
    return hashlib.sha256(
        json.dumps(suite_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _emit(artifact: Mapping[str, Any], key: str) -> None:
    result = artifact[key]
    for condition in CONDITIONS:
        print(
            f"METRIC {condition.lower()}_accuracy="
            f"{float(result['conditions'][condition]['exact_accuracy']):.6f}",
            flush=True,
        )
    print(
        "METRIC c7_flip_rate="
        f"{float(result['C7']['state_conditioned_flip_rate']):.6f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    subparsers = parser.add_subparsers(dest="command")
    dev_parser = subparsers.add_parser("train-dev")
    dev_parser.add_argument("--organ-dir", required=True)
    dev_parser.add_argument("--seed-index", required=True, type=int, choices=range(5))
    dev_parser.add_argument("--output", required=True)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--organ-dir", required=True)
    test_parser.add_argument("--checkpoint-dir", required=True)
    test_parser.add_argument("--authorization-json", required=True)
    test_parser.add_argument("--output", required=True)
    parser.add_argument("--print-definition", action="store_true")
    args = parser.parse_args(argv)
    if args.print_definition:
        print(json.dumps(suite_definition(), indent=2, sort_keys=True))
        return 0
    if args.command is None:
        parser.error("a command is required unless --print-definition is used")
    if args.command == "train-dev":
        artifact = train_dev(
            args.organ_dir, seed_index=args.seed_index, output_dir=args.output
        )
        _emit(artifact, "dev")
    else:
        artifact = evaluate_test(
            args.organ_dir,
            args.checkpoint_dir,
            authorization_path=args.authorization_json,
            output_dir=args.output,
        )
        _emit(artifact, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
