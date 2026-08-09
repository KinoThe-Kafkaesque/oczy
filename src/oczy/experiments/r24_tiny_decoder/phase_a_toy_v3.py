"""R24-v3 Phase A/B: structured-oracle training of one shared tiny decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from .decoder import TinyDecoderConfig, TinySharedDecoder
from .toy_catalog_v3 import ToyCatalog, ToyInteraction, ToyTask, build_toy_catalog_v3
from .toy_features_v3 import feature_definition_sha256, structured_rule_state
from .vocab import EOS_ID, encode_bytes, encode_with_eos

PHASE_A_V3_SCHEMA = "oczy/r24-toy-phase-a/v3"


@dataclass(frozen=True, slots=True)
class PhaseAToyV3Config:
    catalog_seed: int = 24_003
    init_seed: int = 7
    batch_seed: int = 101
    dropout_seed: int = 202
    control_seed: int = 303
    d_model: int = 64
    n_layers: int = 2
    conditioning: Literal["film", "additive"] = "film"
    deep_conditioning: bool = True
    steps: int = 5_000
    lr: float = 2e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    dropout: float = 0.1
    device: str = "cpu"
    protocol_version: str = "r24-tiny-decoder/v3"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if self.conditioning not in ("film", "additive"):
            raise ValueError("v3 registers only film and additive")
        if not self.deep_conditioning:
            raise ValueError("v3 registers deep conditioning for both arms")


BASE_CONFIG = PhaseAToyV3Config()
CASE_OVERRIDES: dict[str, dict[str, object]] = {
    "film": {},
    "additive": {"conditioning": "additive"},
}


def phase_a_tasks(catalog: ToyCatalog) -> tuple[tuple[ToyTask, ...], tuple[ToyTask, ...]]:
    """Return frozen train/dev tasks; sealed TEST is absent."""
    train = catalog.organ_train + catalog.cortex_meta_train
    validation = catalog.organ_dev + catalog.cortex_meta_dev
    test_fingerprints = {task.rule_fingerprint for task in catalog.sealed_test}
    if test_fingerprints & {task.rule_fingerprint for task in train + validation}:
        raise ValueError("sealed TEST leaked into Phase A")
    if {task.rule_fingerprint for task in train} & {
        task.rule_fingerprint for task in validation
    }:
        raise ValueError("Phase A train/dev firewall failed")
    return train, validation


def _rows(tasks: Sequence[ToyTask]) -> list[tuple[ToyTask, ToyInteraction]]:
    return [
        (task, interaction)
        for task in tasks
        for interaction in task.teaching + task.heldout
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trim(row: list[int]) -> list[int]:
    return row[: row.index(EOS_ID) + 1] if EOS_ID in row else row


def _donor_index(catalog: ToyCatalog) -> dict[str, ToyTask]:
    return {task.rule_fingerprint: donor for task, donor in catalog.donor_pairs()}


def evaluate_decoder(
    decoder: TinySharedDecoder,
    tasks: Sequence[ToyTask],
    *,
    catalog: ToyCatalog,
    batch_size: int,
) -> dict[str, Any]:
    decoder.eval()
    rows = _rows(tasks)
    donors = _donor_index(catalog)
    totals = {name: 0 for name in ("oracle", "zero", "swapped")}
    per_rule: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[bool]] = {name: [] for name in totals}
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            query = torch.tensor(
                [encode_bytes(interaction.query_text) for _, interaction in batch],
                dtype=torch.long,
                device=next(decoder.parameters()).device,
            )
            oracle = torch.stack(
                [structured_rule_state(task.rule) for task, _ in batch]
            ).to(query.device)
            states = {
                "oracle": oracle,
                "zero": torch.zeros_like(oracle),
                "swapped": torch.stack(
                    [
                        structured_rule_state(donors[task.rule_fingerprint].rule)
                        for task, _ in batch
                    ]
                ).to(query.device),
            }
            generated = {
                name: decoder.generate_greedy(query, state, max_new_tokens=3).tolist()
                for name, state in states.items()
            }
            for index, (task, interaction) in enumerate(batch):
                gold = encode_with_eos(interaction.answer_text)
                record = per_rule.setdefault(
                    task.rule_fingerprint,
                    {
                        "rotation": task.rule.rotation,
                        "total": 0,
                        "controls": {
                            name: {"correct": 0} for name in totals
                        },
                    },
                )
                record["total"] += 1
                for name in totals:
                    correct = _trim(generated[name][index]) == gold
                    totals[name] += int(correct)
                    record["controls"][name]["correct"] += int(correct)
                    predictions[name].append(correct)
    total = len(rows)
    controls = {
        name: {"correct": correct, "total": total, "exact_accuracy": correct / total}
        for name, correct in totals.items()
    }
    for record in per_rule.values():
        for metrics in record["controls"].values():
            metrics["accuracy"] = metrics["correct"] / record["total"]
    paired: dict[str, dict[str, int]] = {}
    for control in ("zero", "swapped"):
        counts = {"oracle_only": 0, "control_only": 0, "both": 0, "neither": 0}
        for oracle_ok, control_ok in zip(
            predictions["oracle"], predictions[control], strict=True
        ):
            key = (
                "both"
                if oracle_ok and control_ok
                else "oracle_only"
                if oracle_ok
                else "control_only"
                if control_ok
                else "neither"
            )
            counts[key] += 1
        paired[control] = counts
    return {
        "controls": controls,
        "paired_controls": paired,
        "per_rule_controls": dict(sorted(per_rule.items())),
    }


def train_phase_a_toy_v3(
    config: PhaseAToyV3Config,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    catalog = build_toy_catalog_v3(root_seed=config.catalog_seed)
    train_tasks, validation_tasks = phase_a_tasks(catalog)
    train_rows = _rows(train_tasks)
    device = torch.device(config.device)
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    decoder_config = TinyDecoderConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        conditioning=config.conditioning,
        deep_film=config.deep_conditioning,
        dropout=config.dropout,
        version=config.protocol_version,
    )
    decoder = TinySharedDecoder(decoder_config).to(device)
    initial_weight_hash = decoder.parameter_hash()
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    rng = random.Random(config.batch_seed)
    order_hash = hashlib.sha256()
    torch.manual_seed(config.dropout_seed)
    trace: list[dict[str, float | int]] = []
    for step in range(config.steps):
        decoder.train()
        batch = rng.sample(train_rows, config.batch_size)
        for task, interaction in batch:
            order_hash.update(task.rule_fingerprint.encode())
            order_hash.update(bytes((interaction.input_value,)))
        query = torch.tensor(
            [encode_bytes(interaction.query_text) for _, interaction in batch],
            dtype=torch.long,
            device=device,
        )
        answer = torch.tensor(
            [encode_with_eos(interaction.answer_text) for _, interaction in batch],
            dtype=torch.long,
            device=device,
        )
        state = torch.stack([structured_rule_state(task.rule) for task, _ in batch]).to(
            device
        )
        loss = decoder.teacher_forced_loss(query, state, answer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 1_000 == 0 or step + 1 == config.steps:
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
            }
            trace.append(row)
            print(
                f"R24-v3 Phase A {config.conditioning} {step + 1}/{config.steps} "
                f"loss={loss.item():.6f}",
                flush=True,
            )
    validation = evaluate_decoder(
        decoder, validation_tasks, catalog=catalog, batch_size=config.batch_size
    )
    weight_hash = decoder.parameter_hash()
    frozen_hash = decoder.freeze()
    if weight_hash != frozen_hash:
        raise RuntimeError("decoder hash changed while freezing")
    oracle = validation["controls"]["oracle"]
    zero = validation["controls"]["zero"]
    swapped = validation["controls"]["swapped"]
    artifact: dict[str, Any] = {
        "schema_version": PHASE_A_V3_SCHEMA,
        "phase_a_suite_sha256": suite_sha256(),
        "protocol_version": config.protocol_version,
        "config": asdict(config),
        "decoder_config": asdict(decoder_config),
        "catalog_manifest": catalog.manifest_dict(),
        "catalog_manifest_sha256": catalog.manifest_sha256,
        "feature_definition_sha256": feature_definition_sha256(),
        "train_task_count": len(train_tasks),
        "validation_task_count": len(validation_tasks),
        "train_example_count": len(train_rows),
        "validation_example_count": sum(
            len(task.teaching) + len(task.heldout) for task in validation_tasks
        ),
        "sealed_test_task_count": len(catalog.sealed_test),
        "sealed_test_accessed": False,
        "initial_weight_hash": initial_weight_hash,
        "weight_hash": weight_hash,
        "frozen_hash": frozen_hash,
        "batch_order_hash": order_hash.hexdigest(),
        "trace": trace,
        "validation": validation,
        "oracle_dev_accuracy": oracle["exact_accuracy"],
        "zero_dev_accuracy": zero["exact_accuracy"],
        "swapped_dev_accuracy": swapped["exact_accuracy"],
        "zero_delta": oracle["exact_accuracy"] - zero["exact_accuracy"],
        "swapped_delta": oracle["exact_accuracy"] - swapped["exact_accuracy"],
        "phase_c_organ_gate": bool(
            oracle["exact_accuracy"] >= 0.95
            and oracle["exact_accuracy"] - swapped["exact_accuracy"] >= 0.50
        ),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        decoder_path = output / "decoder.pt"
        torch.save(
            {
                "schema_version": PHASE_A_V3_SCHEMA,
                "phase_a_suite_sha256": suite_sha256(),
                "phase_a_config": asdict(config),
                "case": config.conditioning,
                "decoder_config": asdict(decoder_config),
                "decoder_state_dict": decoder.state_dict(),
                "weight_hash": weight_hash,
                "catalog_manifest_sha256": catalog.manifest_sha256,
                "feature_definition_sha256": feature_definition_sha256(),
            },
            decoder_path,
        )
        artifact["files"] = {
            "decoder.pt": {
                "sha256": _sha256_file(decoder_path),
                "bytes": decoder_path.stat().st_size,
            }
        }
        (output / "artifact.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True)
        )
        # Strict reload and output parity.
        payload = torch.load(decoder_path, map_location=device, weights_only=True)
        reloaded = TinySharedDecoder(TinyDecoderConfig(**payload["decoder_config"])).to(
            device
        )
        reloaded.load_state_dict(payload["decoder_state_dict"])
        reloaded.eval()
        if reloaded.parameter_hash() != weight_hash:
            raise RuntimeError("decoder reload hash mismatch")
        probe_task = validation_tasks[0]
        probe = probe_task.heldout[0]
        query = torch.tensor([encode_bytes(probe.query_text)], device=device)
        state = structured_rule_state(probe_task.rule).unsqueeze(0).to(device)
        with torch.inference_mode():
            equal = torch.equal(
                decoder(query, state), reloaded(query, state)
            )
        if not equal:
            raise RuntimeError("decoder reload output mismatch")
        artifact["reload_verification"] = {
            "weight_hash": reloaded.parameter_hash(),
            "output_bit_equal": True,
        }
        (output / "artifact.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True)
        )
    return artifact


def implementation_file_sha256() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "decoder.py",
        "vocab.py",
        "phase_a_toy_v3.py",
        "toy_catalog_v3.py",
        "toy_features_v3.py",
    )
    return {name: _sha256_file(root / name) for name in names}


def suite_definition() -> dict[str, Any]:
    return {
        "schema_version": PHASE_A_V3_SCHEMA,
        "implementation_file_sha256": implementation_file_sha256(),
        "base_config": asdict(BASE_CONFIG),
        "case_overrides": CASE_OVERRIDES,
        "organ_gate": {
            "oracle_dev_accuracy_min": 0.95,
            "oracle_minus_swapped_min": 0.50,
            "additive_failure_policy": "run and report BLOCKED; do not call it a cortex null",
        },
        "train_splits": ["organ_train", "cortex_meta_train"],
        "validation_splits": ["organ_dev", "cortex_meta_dev"],
        "sealed_test_access": False,
    }


def suite_sha256() -> str:
    return hashlib.sha256(
        json.dumps(suite_definition(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_case(case: str, output: str | Path) -> dict[str, Any]:
    if case not in CASE_OVERRIDES:
        raise KeyError(case)
    config = replace(BASE_CONFIG, **CASE_OVERRIDES[case])
    artifact = train_phase_a_toy_v3(config, output_dir=output)
    artifact["suite"] = {"case": case, "suite_sha256": suite_sha256()}
    Path(output, "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--case", choices=sorted(CASE_OVERRIDES))
    parser.add_argument("--output")
    parser.add_argument("--print-definition", action="store_true")
    args = parser.parse_args(argv)
    if args.print_definition:
        print(json.dumps(suite_definition(), indent=2, sort_keys=True))
        return 0
    if args.case is None or args.output is None:
        parser.error("--case and --output are required unless --print-definition is used")
    artifact = run_case(args.case, args.output)
    for name in (
        "oracle_dev_accuracy",
        "zero_dev_accuracy",
        "swapped_dev_accuracy",
        "zero_delta",
        "swapped_delta",
    ):
        print(f"METRIC {name}={float(artifact[name]):.6f}", flush=True)
    print(f"METRIC phase_c_organ_gate={int(artifact['phase_c_organ_gate'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
