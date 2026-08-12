"""R24 Phase-A v2: reproducible, diagnostic oracle-to-decoder training.

V2 is a new experimental protocol.  It preserves the frozen outcome metric
(exact byte-sequence match) while repairing padded-query evaluation, seeding
before construction, and adding pre-registered representation/training
ablations and causal state controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
import torch.nn.functional as F

from .corpus_v2 import audit_examples, build_phase_a_v2_corpus, hash_examples
from .decoder import TinyDecoderConfig, TinySharedDecoder
from .oracle_v2 import LearnedRuleOracleEncoder, OraclePooling, PoolingOracleEncoder
from .pretrain import PretrainExample, fixed_hash_z
from .vocab import EOS_ID, PAD_ID, encode_bytes, encode_with_eos

SchedulerName = Literal["constant", "cosine"]
OracleMode = Literal["text", "hash", "learned"]


@dataclass(frozen=True, slots=True)
class PhaseAV2Config:
    # root_seed is a compatibility fallback; v2 artifacts resolve and record each stream.
    root_seed: int = 123
    catalog_seed: int | None = None
    init_seed: int | None = None
    batch_seed: int | None = None
    dropout_seed: int | None = None
    control_seed: int | None = None
    train_per_family: int = 20
    val_per_family: int = 10
    d_model: int = 64
    n_layers: int = 2
    conditioning: Literal["film", "additive", "prefix", "none"] = "film"
    deep_film: bool = True
    n_prefix_tokens: int = 4
    encoder_pooling: OraclePooling = "mean"
    oracle_mode: OracleMode = "text"
    steps: int = 800
    lr: float = 3e-3
    encoder_lr_multiplier: float = 1.0
    weight_decay: float = 0.01
    batch_size: int = 32
    scheduler: SchedulerName = "constant"
    warmup_steps: int = 0
    counterfactual_weight: float = 0.0
    dropout: float = 0.1
    device: str = "cpu"
    max_train_eval_examples: int = 256
    protocol_version: str = "r24-tiny-decoder/v2"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.lr <= 0 or self.encoder_lr_multiplier <= 0:
            raise ValueError("lr and encoder_lr_multiplier must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.counterfactual_weight < 0:
            raise ValueError("counterfactual_weight must be non-negative")
        if self.conditioning == "prefix" and self.deep_film:
            raise ValueError("prefix conditioning requires deep_film=False")

    def resolved_seeds(self) -> dict[str, int]:
        return {
            "catalog": self.root_seed if self.catalog_seed is None else self.catalog_seed,
            "init": self.root_seed if self.init_seed is None else self.init_seed,
            "batch": self.root_seed if self.batch_seed is None else self.batch_seed,
            "dropout": self.root_seed if self.dropout_seed is None else self.dropout_seed,
            "control": self.root_seed if self.control_seed is None else self.control_seed,
        }


def seed_everything(seed: int) -> None:
    """Seed before constructing either model (the v1 loop seeded too late)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_examples(
    examples: Sequence[PretrainExample], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    query_ids = [encode_bytes(example.query_text) for example in examples]
    query_lengths = {len(ids) for ids in query_ids}
    if len(query_lengths) != 1:
        raise ValueError("v2 batches must have one query length; padding would change alignment")
    answer_ids = [encode_with_eos(example.answer_text) for example in examples]
    query = torch.tensor(query_ids, dtype=torch.long, device=device)
    max_answer = max(map(len, answer_ids))
    answer = torch.full((len(examples), max_answer), PAD_ID, dtype=torch.long, device=device)
    for index, ids in enumerate(answer_ids):
        answer[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return query, answer


def _encode_texts(
    texts: Sequence[str], encoder: PoolingOracleEncoder, device: torch.device
) -> torch.Tensor:
    encoded = [encode_bytes(text)[: encoder.max_len] for text in texts]
    max_len = max(map(len, encoded))
    tokens = torch.full((len(encoded), max_len), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(encoded), max_len), dtype=torch.bool, device=device)
    for index, ids in enumerate(encoded):
        tokens[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, : len(ids)] = True
    return encoder(tokens, mask)


def _states_for(
    examples: Sequence[PretrainExample],
    oracle: PoolingOracleEncoder | LearnedRuleOracleEncoder | None,
    device: torch.device,
) -> torch.Tensor:
    if oracle is None:
        return fixed_hash_z([example.rule_fp for example in examples], device)
    if isinstance(oracle, LearnedRuleOracleEncoder):
        return oracle([example.rule_fp for example in examples])
    return _encode_texts([example.oracle_text for example in examples], oracle, device)


def _buckets(examples: Sequence[PretrainExample]) -> dict[int, list[PretrainExample]]:
    grouped: dict[int, list[PretrainExample]] = defaultdict(list)
    for example in examples:
        grouped[len(encode_bytes(example.query_text))].append(example)
    return dict(grouped)


def _draw_batch(
    grouped: dict[int, list[PretrainExample]],
    batch_size: int,
    rng: random.Random,
) -> list[PretrainExample]:
    lengths = sorted(grouped)
    weights = [len(grouped[length]) for length in lengths]
    length = rng.choices(lengths, weights=weights, k=1)[0]
    return rng.choices(grouped[length], k=batch_size)


def _wrong_rule_examples(
    examples: Sequence[PretrainExample], universe: Sequence[PretrainExample]
) -> list[PretrainExample]:
    by_rule: dict[str, PretrainExample] = {}
    for candidate in universe:
        by_rule.setdefault(candidate.rule_fp, candidate)
    ordered = [by_rule[key] for key in sorted(by_rule)]
    if len(ordered) < 2:
        return list(examples)
    wrong: list[PretrainExample] = []
    for example in examples:
        index = next(
            i for i, candidate in enumerate(ordered) if candidate.rule_fp == example.rule_fp
        )
        wrong.append(ordered[(index + 1) % len(ordered)])
    return wrong


def _scheduler(
    optimizer: torch.optim.Optimizer, config: PhaseAV2Config
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-8, (step + 1) / config.warmup_steps)
        if config.scheduler == "constant":
            return 1.0
        progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _trim_prediction(tokens: Sequence[int]) -> list[int]:
    result = list(tokens)
    if EOS_ID in result:
        result = result[: result.index(EOS_ID) + 1]
    return result


class _MetricAccumulator(TypedDict):
    correct: int
    total: int
    aligned_tokens_correct: int
    gold_tokens: int
    edit_similarity_sum: float
    predictions: list[list[int]]


def _metric_accumulator() -> _MetricAccumulator:
    return {
        "correct": 0,
        "total": 0,
        "aligned_tokens_correct": 0,
        "gold_tokens": 0,
        "edit_similarity_sum": 0.0,
        "predictions": [],
    }


def _update_metrics(
    accumulator: _MetricAccumulator,
    examples: Sequence[PretrainExample],
    generated: torch.Tensor,
) -> None:
    for row, example in zip(generated.tolist(), examples, strict=True):
        predicted = _trim_prediction(row)
        gold = encode_with_eos(example.answer_text)
        accumulator["correct"] += int(predicted == gold)
        accumulator["total"] += 1
        aligned = sum(a == b for a, b in zip(predicted, gold, strict=False))
        accumulator["aligned_tokens_correct"] += aligned
        accumulator["gold_tokens"] += len(gold)
        distance = _edit_distance(predicted, gold)
        denominator = max(1, len(predicted), len(gold))
        accumulator["edit_similarity_sum"] += 1.0 - distance / denominator
        accumulator["predictions"].append(predicted)


def _finalize_metrics(accumulator: _MetricAccumulator) -> dict[str, float | int]:
    total = accumulator["total"]
    gold_tokens = accumulator["gold_tokens"]
    return {
        "correct": accumulator["correct"],
        "total": total,
        "exact_accuracy": accumulator["correct"] / max(1, total),
        "aligned_token_accuracy": accumulator["aligned_tokens_correct"] / max(1, gold_tokens),
        "edit_similarity": accumulator["edit_similarity_sum"] / max(1, total),
    }


def _teacher_forced_metrics(
    decoder: TinySharedDecoder,
    state: torch.Tensor,
    query: torch.Tensor,
    answer: torch.Tensor,
) -> tuple[float, int, int]:
    query_len = query.shape[1]
    answer_len = answer.shape[1]
    logits = decoder(query, state, answer)
    target_logits = logits[:, query_len - 1 : query_len + answer_len - 1]
    loss = F.cross_entropy(
        target_logits.transpose(1, 2), answer, ignore_index=PAD_ID, reduction="sum"
    )
    valid = answer != PAD_ID
    correct = ((target_logits.argmax(dim=-1) == answer) & valid).sum().item()
    return float(loss.item()), int(correct), int(valid.sum().item())


def evaluate_phase_a_v2(
    decoder: TinySharedDecoder,
    oracle: PoolingOracleEncoder | LearnedRuleOracleEncoder | None,
    examples: Sequence[PretrainExample],
    *,
    seed: int,
    batch_size: int,
    include_controls: bool = True,
    sample_limit: int = 12,
) -> dict[str, Any]:
    """Evaluate without right-padding queries; v1's padded batching was invalid."""
    decoder.eval()
    if oracle is not None:
        oracle.eval()
    device = next(decoder.parameters()).device
    grouped = _buckets(examples)
    controls = ["oracle", "zero", "random", "swapped"] if include_controls else ["oracle"]
    accumulators = {name: _metric_accumulator() for name in controls}
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kind_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    teacher_loss = 0.0
    teacher_correct = 0
    teacher_tokens = 0
    samples: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(seed + 9173)
    universe = list(examples)

    with torch.inference_mode():
        for query_length in sorted(grouped):
            bucket = grouped[query_length]
            for offset in range(0, len(bucket), batch_size):
                batch = bucket[offset : offset + batch_size]
                query, answer = _pad_examples(batch, device)
                state = _states_for(batch, oracle, device)
                states: dict[str, torch.Tensor] = {"oracle": state}
                if include_controls:
                    states["zero"] = torch.zeros_like(state)
                    states["random"] = torch.randn(
                        state.shape, generator=generator, device=device, dtype=state.dtype
                    ) * state.std().clamp_min(0.1)
                    wrong_examples = _wrong_rule_examples(batch, universe)
                    states["swapped"] = _states_for(wrong_examples, oracle, device)
                batch_predictions: dict[str, list[list[int]]] = {}
                for name, control_state in states.items():
                    generated = decoder.generate_greedy(
                        query, control_state, max_new_tokens=answer.shape[1] + 5
                    )
                    _update_metrics(accumulators[name], batch, generated)
                    batch_predictions[name] = [_trim_prediction(row) for row in generated.tolist()]
                loss, correct, tokens = _teacher_forced_metrics(decoder, state, query, answer)
                teacher_loss += loss
                teacher_correct += correct
                teacher_tokens += tokens
                oracle_predictions = batch_predictions["oracle"]
                for index, example in enumerate(batch):
                    is_correct = oracle_predictions[index] == encode_with_eos(example.answer_text)
                    family_counts[example.family][0] += int(is_correct)
                    family_counts[example.family][1] += 1
                    kind_counts[example.kind][0] += int(is_correct)
                    kind_counts[example.kind][1] += 1
                    if len(samples) < sample_limit:
                        samples.append(
                            {
                                "family": example.family,
                                "kind": example.kind,
                                "query": example.query_text,
                                "gold_ids": encode_with_eos(example.answer_text),
                                "prediction_ids": oracle_predictions[index],
                            }
                        )

    finalized = {name: _finalize_metrics(value) for name, value in accumulators.items()}
    result: dict[str, Any] = {
        "controls": finalized,
        "teacher_forced_loss": teacher_loss / max(1, teacher_tokens),
        "teacher_forced_token_accuracy": teacher_correct / max(1, teacher_tokens),
        "by_family": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(family_counts.items())
        },
        "by_kind": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(kind_counts.items())
        },
        "samples": samples,
    }
    if include_controls:
        oracle_predictions = accumulators["oracle"]["predictions"]
        assert isinstance(oracle_predictions, list)
        ordered_examples = [
            example for length in sorted(grouped) for example in grouped[length]
        ]
        predictions_by_control: dict[str, list[list[int]]] = {}
        for control in controls:
            predictions = accumulators[control]["predictions"]
            assert isinstance(predictions, list)
            predictions_by_control[control] = predictions

        paired: dict[str, dict[str, int]] = {}
        for control in ("zero", "random", "swapped"):
            control_predictions = predictions_by_control[control]
            counts = {"oracle_only": 0, "control_only": 0, "both": 0, "neither": 0}
            for oracle_pred, control_pred, example in zip(
                oracle_predictions, control_predictions, ordered_examples, strict=True
            ):
                gold = encode_with_eos(example.answer_text)
                oracle_ok = oracle_pred == gold
                control_ok = control_pred == gold
                if oracle_ok and control_ok:
                    counts["both"] += 1
                elif oracle_ok:
                    counts["oracle_only"] += 1
                elif control_ok:
                    counts["control_only"] += 1
                else:
                    counts["neither"] += 1
            paired[control] = counts
        result["paired_controls"] = paired

        # Rows within a rule are repeated measures.  Persist aggregate paired
        # correctness by held-out rule so confirmation can report equal-rule
        # macro summaries instead of treating every rendered probe as iid.
        per_rule: dict[str, dict[str, Any]] = {}
        for index, example in enumerate(ordered_examples):
            record = per_rule.setdefault(
                example.rule_fp,
                {
                    "family": example.family,
                    "total": 0,
                    "controls": {
                        name: {"correct": 0} for name in controls
                    },
                    "paired_swapped": {
                        "oracle_only": 0,
                        "control_only": 0,
                        "both": 0,
                        "neither": 0,
                    },
                },
            )
            record["total"] += 1
            gold = encode_with_eos(example.answer_text)
            outcomes = {
                name: predictions_by_control[name][index] == gold for name in controls
            }
            for name, is_correct in outcomes.items():
                record["controls"][name]["correct"] += int(is_correct)
            oracle_ok = outcomes["oracle"]
            swapped_ok = outcomes["swapped"]
            if oracle_ok and swapped_ok:
                paired_key = "both"
            elif oracle_ok:
                paired_key = "oracle_only"
            elif swapped_ok:
                paired_key = "control_only"
            else:
                paired_key = "neither"
            record["paired_swapped"][paired_key] += 1
        for record in per_rule.values():
            total = int(record["total"])
            for metrics in record["controls"].values():
                metrics["accuracy"] = int(metrics["correct"]) / total
        result["per_rule_controls"] = dict(sorted(per_rule.items()))
    return result


def evaluate_retrieval_baselines(
    train_examples: Sequence[PretrainExample],
    validation_examples: Sequence[PretrainExample],
) -> dict[str, Any]:
    """Explicit non-parametric bars retained alongside changed dynamics."""
    answer_counts = Counter(example.answer_text for example in train_examples)
    majority_answer = answer_counts.most_common(1)[0][0]
    exact_query: dict[str, Counter[str]] = defaultdict(Counter)
    for example in train_examples:
        exact_query[example.query_text][example.answer_text] += 1

    def ngrams(text: str) -> set[bytes]:
        raw = text.encode()
        if len(raw) < 3:
            return {raw}
        return {raw[index : index + 3] for index in range(len(raw) - 2)}

    retrieval_rows: list[tuple[str, str, set[bytes]]] = []
    seen: set[tuple[str, str]] = set()
    for example in train_examples:
        key = (example.query_text, example.answer_text)
        if key not in seen:
            seen.add(key)
            retrieval_rows.append((key[0], key[1], ngrams(key[0])))
    retrieval_rows.sort(key=lambda row: (row[0], row[1]))

    counts = {
        "majority_answer": 0,
        "exact_query_lookup": 0,
        "nearest_query_3gram": 0,
    }
    exact_query_coverage = 0
    for example in validation_examples:
        if majority_answer == example.answer_text:
            counts["majority_answer"] += 1
        if example.query_text in exact_query:
            exact_query_coverage += 1
            lookup_answer = exact_query[example.query_text].most_common(1)[0][0]
        else:
            lookup_answer = majority_answer
        if lookup_answer == example.answer_text:
            counts["exact_query_lookup"] += 1
        query_grams = ngrams(example.query_text)
        best_score = -1.0
        best_answer = majority_answer
        for _, candidate_answer, candidate_grams in retrieval_rows:
            union = len(query_grams | candidate_grams)
            score = len(query_grams & candidate_grams) / max(1, union)
            if score > best_score:
                best_score = score
                best_answer = candidate_answer
        if best_answer == example.answer_text:
            counts["nearest_query_3gram"] += 1
    total = len(validation_examples)
    return {
        "total": total,
        "exact_query_coverage": exact_query_coverage,
        "exact_query_coverage_rate": exact_query_coverage / max(1, total),
        **{
            name: {"correct": correct, "accuracy": correct / max(1, total)}
            for name, correct in counts.items()
        },
    }


def _stable_subset(examples: Sequence[PretrainExample], limit: int) -> list[PretrainExample]:
    if len(examples) <= limit:
        return list(examples)
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{example.rule_fp}|{example.kind}|{example.query_text}".encode()
        ).hexdigest(),
    )[:limit]


def _latent_diagnostics(
    oracle: PoolingOracleEncoder | LearnedRuleOracleEncoder | None,
    examples: Sequence[PretrainExample],
    device: torch.device,
) -> dict[str, float | int]:
    by_rule: dict[str, PretrainExample] = {}
    for example in examples:
        by_rule.setdefault(example.rule_fp, example)
    unique = [by_rule[key] for key in sorted(by_rule)]
    states: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(unique), 64):
            states.append(_states_for(unique[offset : offset + 64], oracle, device).cpu())
    matrix = torch.cat(states, dim=0)
    return {
        "rules": len(unique),
        "mean_norm": float(matrix.norm(dim=1).mean().item()),
        "dimension_std_mean": float(matrix.std(dim=0, unbiased=False).mean().item()),
        "global_std": float(matrix.std(unbiased=False).item()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_phase_a_v2(
    config: PhaseAV2Config,
    *,
    train_examples: Sequence[PretrainExample] | None = None,
    val_examples: Sequence[PretrainExample] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate one v2 configuration."""
    seeds = config.resolved_seeds()
    device = torch.device(config.device)
    if (train_examples is None) != (val_examples is None):
        raise ValueError("train_examples and val_examples must be overridden together")
    if train_examples is None:
        built_train, built_val, corpus_audit = build_phase_a_v2_corpus(
            root_seed=seeds["catalog"],
            train_per_family=config.train_per_family,
            val_per_family=config.val_per_family,
        )
        train_examples = built_train
        val_examples = built_val
        corpus_hash = corpus_audit.corpus_sha256
        split_hash = corpus_audit.split_sha256
        corpus_audit_record: dict[str, Any] = asdict(corpus_audit)
    else:
        assert val_examples is not None
        train_examples = list(train_examples)
        val_examples = list(val_examples)
        custom_audit = audit_examples(train_examples, val_examples)
        if custom_audit["conflicts"]:
            raise ValueError(
                "custom R24 v2 corpus has conflicting targets for identical inputs: "
                + json.dumps(custom_audit["conflicts"], sort_keys=True)[:2000]
            )
        corpus_hash = hashlib.sha256(
            (hash_examples(train_examples) + hash_examples(val_examples)).encode()
        ).hexdigest()
        split_hash = hashlib.sha256(
            json.dumps(custom_audit, sort_keys=True, default=list).encode()
        ).hexdigest()
        corpus_audit_record = {
            "schema_version": "oczy/r24-phase-a-custom-corpus/v2",
            **custom_audit,
            "corpus_sha256": corpus_hash,
            "split_sha256": split_hash,
        }
    assert train_examples is not None and val_examples is not None
    train_examples = list(train_examples)
    val_examples = list(val_examples)
    if not train_examples or not val_examples:
        raise ValueError("train and validation examples must be non-empty")

    seed_everything(seeds["init"])
    decoder_config = TinyDecoderConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        conditioning=config.conditioning,
        deep_film=config.deep_film,
        n_prefix_tokens=config.n_prefix_tokens,
        dropout=config.dropout,
        version=config.protocol_version,
    )
    decoder = TinySharedDecoder(decoder_config).to(device)
    initial_weight_hash = decoder.parameter_hash()
    # Encoder initialization must not depend on conditioner parameter count.
    torch.manual_seed(seeds["init"] + 1_000_003)
    oracle: PoolingOracleEncoder | LearnedRuleOracleEncoder | None
    if config.oracle_mode == "text":
        oracle = PoolingOracleEncoder(
            d_model=config.d_model,
            n_layers=2,
            n_heads=2,
            r_dim=64,
            pooling=config.encoder_pooling,
            dropout=config.dropout,
        ).to(device)
    elif config.oracle_mode == "learned":
        oracle = LearnedRuleOracleEncoder(
            [example.rule_fp for example in train_examples], r_dim=64
        ).to(device)
    else:
        oracle = None
    decoder_parameters = list(decoder.parameters())
    oracle_parameters = list(oracle.parameters()) if oracle is not None else []
    parameters = decoder_parameters + oracle_parameters
    parameter_groups: list[dict[str, Any]] = [{"params": decoder_parameters, "lr": config.lr}]
    if oracle_parameters:
        parameter_groups.append(
            {
                "params": oracle_parameters,
                "lr": config.lr * config.encoder_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config)
    rng = random.Random(seeds["batch"])
    grouped = _buckets(train_examples)
    trace: list[dict[str, float | int]] = []
    batch_order_hash = hashlib.sha256()
    torch.manual_seed(seeds["dropout"])

    for step in range(config.steps):
        decoder.train()
        if oracle is not None:
            oracle.train()
        batch = _draw_batch(grouped, config.batch_size, rng)
        for example in batch:
            batch_order_hash.update(
                f"{example.rule_fp}|{example.kind}|{example.query_text}\n".encode()
            )
        query, answer = _pad_examples(batch, device)
        state = _states_for(batch, oracle, device)
        per_example_loss = decoder.teacher_forced_loss_per_example(query, state, answer)
        primary_loss = per_example_loss.mean()
        rank_loss = primary_loss.new_zeros(())
        if config.counterfactual_weight > 0:
            wrong_examples = _wrong_rule_examples(batch, train_examples)
            wrong_state = _states_for(wrong_examples, oracle, device)
            wrong_loss = decoder.teacher_forced_loss_per_example(query, wrong_state, answer)
            rank_loss = F.softplus(per_example_loss - wrong_loss).mean()
        loss = primary_loss + config.counterfactual_weight * rank_loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % max(1, config.steps // 8) == 0:
            row = {
                "step": step + 1,
                "loss": float(primary_loss.detach().item()),
                "rank_loss": float(rank_loss.detach().item()),
                "grad_norm": float(grad_norm),
                "decoder_lr": float(optimizer.param_groups[0]["lr"]),
                "encoder_lr": float(
                    optimizer.param_groups[-1]["lr"]
                    if oracle_parameters
                    else optimizer.param_groups[0]["lr"]
                ),
            }
            trace.append(row)
            print(
                "step {step}/{total} loss {loss:.4f} rank {rank:.4f} "
                "grad {grad:.3f} lr {lr:.6g}".format(
                    step=row["step"],
                    total=config.steps,
                    loss=row["loss"],
                    rank=row["rank_loss"],
                    grad=row["grad_norm"],
                    lr=row["decoder_lr"],
                ),
                flush=True,
            )

    decoder.eval()
    if oracle is not None:
        oracle.eval()
    val_metrics = evaluate_phase_a_v2(
        decoder,
        oracle,
        val_examples,
        seed=seeds["control"],
        batch_size=config.batch_size,
        include_controls=True,
    )
    train_subset = _stable_subset(train_examples, config.max_train_eval_examples)
    train_metrics = evaluate_phase_a_v2(
        decoder,
        oracle,
        train_subset,
        seed=seeds["control"] + 1,
        batch_size=config.batch_size,
        include_controls=False,
        sample_limit=0,
    )
    latent = _latent_diagnostics(oracle, val_examples, device)
    retrieval_baselines = evaluate_retrieval_baselines(train_examples, val_examples)
    weight_hash = decoder.parameter_hash()
    frozen_hash = decoder.freeze()
    if weight_hash != frozen_hash:
        raise RuntimeError("decoder hash changed while freezing")

    controls = val_metrics["controls"]
    assert isinstance(controls, dict)
    oracle_metrics = controls["oracle"]
    zero_metrics = controls["zero"]
    swapped_metrics = controls["swapped"]
    assert isinstance(oracle_metrics, dict)
    assert isinstance(zero_metrics, dict)
    assert isinstance(swapped_metrics, dict)
    train_rules = {example.rule_fp for example in train_examples}
    validation_rules = {example.rule_fp for example in val_examples}
    conditioner_parameters = sum(
        parameter.numel()
        for name, parameter in decoder.named_parameters()
        if any(part in name for part in ("films", "additive_projs", "prefix_proj"))
    )
    artifact: dict[str, Any] = {
        "schema_version": "oczy/r24-phase-a-artifact/v2",
        "protocol_version": config.protocol_version,
        "config": asdict(config),
        "resolved_seeds": seeds,
        "corpus_hash": corpus_hash,
        "split_hash": split_hash,
        "corpus_audit": corpus_audit_record,
        "batch_order_hash": batch_order_hash.hexdigest(),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_rules": len(train_rules),
        "validation_rules": len(validation_rules),
        "rule_overlap": len(train_rules & validation_rules),
        "parameter_counts": {
            "decoder": sum(parameter.numel() for parameter in decoder.parameters()),
            "conditioner": conditioner_parameters,
            "oracle": sum(parameter.numel() for parameter in oracle.parameters())
            if oracle is not None
            else 0,
        },
        "initial_weight_hash": initial_weight_hash,
        "weight_hash": weight_hash,
        "frozen_hash": frozen_hash,
        "trace": trace,
        "train": train_metrics,
        "validation": val_metrics,
        "latent": latent,
        "retrieval_baselines": retrieval_baselines,
        # Compatibility metrics consumed by the remote runner.
        "oracle_dev_accuracy": oracle_metrics["exact_accuracy"],
        "query_only_dev_accuracy": zero_metrics["exact_accuracy"],
        "swapped_dev_accuracy": swapped_metrics["exact_accuracy"],
        "random_dev_accuracy": controls["random"]["exact_accuracy"],
        "zero_state_delta": float(oracle_metrics["exact_accuracy"])
        - float(zero_metrics["exact_accuracy"]),
        "swapped_delta": float(oracle_metrics["exact_accuracy"])
        - float(swapped_metrics["exact_accuracy"]),
        "random_delta": float(oracle_metrics["exact_accuracy"])
        - float(controls["random"]["exact_accuracy"]),
        # Deprecated v1 compatibility name; scientific v2 gates use swapped_delta.
        "delta": float(oracle_metrics["exact_accuracy"]) - float(zero_metrics["exact_accuracy"]),
    }

    # Reload is checked even for in-memory diagnostics.  Phase C must consume a
    # loadable frozen artifact rather than a metadata-only hash.
    reloaded = TinySharedDecoder(decoder_config).to(device)
    reloaded.load_state_dict(decoder.state_dict())
    reloaded.eval()
    reloaded_hash = reloaded.parameter_hash()
    if reloaded_hash != frozen_hash:
        raise RuntimeError("decoder reload hash mismatch")
    probe = [val_examples[0]]
    probe_query, probe_answer = _pad_examples(probe, device)
    with torch.inference_mode():
        probe_state = _states_for(probe, oracle, device)
        original_logits = decoder(probe_query, probe_state, probe_answer)
        reloaded_logits = reloaded(probe_query, probe_state, probe_answer)
    reload_output_equal = torch.equal(original_logits, reloaded_logits)
    if not reload_output_equal:
        raise RuntimeError("decoder reload output mismatch")
    artifact["reload_verification"] = {
        "hash": reloaded_hash,
        "output_bit_equal": reload_output_equal,
    }

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        decoder_path = output / "decoder.pt"
        torch.save(
            {
                "decoder_config": asdict(decoder_config),
                "decoder_state_dict": decoder.state_dict(),
                "weight_hash": weight_hash,
            },
            decoder_path,
        )
        files: dict[str, Any] = {
            "decoder.pt": {
                "sha256": _sha256_file(decoder_path),
                "bytes": decoder_path.stat().st_size,
            }
        }
        if oracle is not None:
            oracle_path = output / "oracle.pt"
            oracle_payload: dict[str, Any] = {
                "mode": config.oracle_mode,
                "state_dict": oracle.state_dict(),
            }
            if isinstance(oracle, LearnedRuleOracleEncoder):
                oracle_payload["rule_to_index"] = oracle.rule_to_index
            else:
                oracle_payload["pooling"] = config.encoder_pooling
            torch.save(oracle_payload, oracle_path)
            files["oracle.pt"] = {
                "sha256": _sha256_file(oracle_path),
                "bytes": oracle_path.stat().st_size,
            }
        artifact["files"] = files
        (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact
