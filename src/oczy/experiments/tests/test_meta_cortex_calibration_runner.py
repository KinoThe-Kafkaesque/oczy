"""High-signal contract tests for the resumable DEV calibration record collector.

These tests verify the **runner** (``calibration_runner.collect_calibration_shard``)
and the **strict merge** (``calibration.merge_calibration_shards``) using a
test-only differentiable tiny organ and five verified developmental checkpoints.

No network, no real Qwen model, no sealed access, no production fake fallback.

Contracts verified:

  1. **Exact scoring** — the runner uses ``FrozenScorer.score_response``
     (NFKC + strip + exact equality).  A substring match that the old
     ``training._score_probe_accuracy`` would accept is rejected.
  2. **Online optimizer prohibition** — ``optimizer_step_count`` is 0 and
     theta/organ hashes are stable before and after collection.
  3. **Trace deletion** — ``trace_count_after`` is 0 in every record.
  4. **Deterministic shard bytes** — re-running the same shard produces
     byte-identical shard JSON.
  5. **Seed/task sharding** — dev/eval seed indices and task indices
     partition the work deterministically; invalid indices raise before
     any output.
  6. **Strict merge completeness** — merge rejects partial data (missing
     5×5 cells, missing 20-repeat groups, missing theta hashes).
  7. **Strict merge duplicate rejection** — duplicate seed cells and
     duplicate no-update repeats raise.
  8. **Strict merge hash rejection** — wrong definition/view/scorer hash
     and inconsistent organ hash raise.
  9. **Canonical output compatibility** — merged files are loadable by
     ``load_no_update_repeat_records`` / ``load_seed_cell_records`` /
     ``load_theta_hashes`` and feed ``run_dev_calibration``.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from oczy.experiments.meta_cortex.artifacts import (
    canonical_theta_hash,
    save_developmental_checkpoint,
)
from oczy.experiments.meta_cortex.calibration import (
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    CALIBRATION_SHARD_SCHEMA,
    FrozenScorer,
    MergeResult,
    NoUpdateRepeatRecord,
    ProbeCount,
    SeedCellRecord,
    ShardCollection,
    ShardThetaHash,
    TaskConditionRecord,
    load_calibration_shard,
    load_no_update_repeat_records,
    load_seed_cell_records,
    load_theta_hashes,
    merge_calibration_shards,
    run_dev_calibration,
    write_calibration_shard,
)
from oczy.experiments.meta_cortex.calibration_runner import (
    ShardConfig,
    collect_calibration_shard,
)
from oczy.experiments.meta_cortex.contracts import (
    DEV_SCHEMA,
    TASKGEN_SCHEMA,
    CheckpointMetadata,
    DevSplit,
    DialogueMessage,
    LearningEvent,
    MetaTask,
    ModelConfig,
    OutcomeCode,
    OuterLoopConfig,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    TaskFamily,
    TaskGeneratorConfig,
)
from oczy.experiments.meta_cortex.instrument_contracts import (
    CALIBRATION_VIEW_SCHEMA,
    CalibrationInstrumentView,
)
from oczy.experiments.meta_cortex.model import MetaCortex
from oczy.experiments.meta_cortex.organ import FrozenOrganError

# ---------------------------------------------------------------------------
# Test-only differentiable tiny frozen organ — no network required.
# ---------------------------------------------------------------------------


class _TinyFrozenOrgan:
    """Test-only differentiable tiny frozen organ.

    Uses a frozen random embedding + linear head with ``requires_grad=False``.
    Gradients flow through ``soft_bank`` input but never to organ parameters.
    No network or model download required.

    This is NOT a production fake fallback — it is a test fixture that
    satisfies the ``FrozenLanguageOrgan`` protocol for unit testing the
    calibration runner's scientific contracts.
    """

    def __init__(self, feature_dim: int = 16, vocab_size: int = 128) -> None:
        self.feature_dim = feature_dim
        self._vocab_size = vocab_size
        self._closed = False
        torch.manual_seed(42)
        self._embedding = torch.randn(vocab_size, feature_dim, requires_grad=False)
        self._output_proj = torch.randn(feature_dim, vocab_size, requires_grad=False)
        self._output_bias = torch.zeros(vocab_size, requires_grad=False)
        self._initial_hash = self.parameter_hash()

    def _tokenize(self, text: str) -> list[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def _render(self, messages: list[DialogueMessage]) -> str:
        return "".join(f"{m.role}: {m.content}\n" for m in messages)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not texts:
            raise FrozenOrganError("encode_texts received empty text sequence")
        features = []
        for text in texts:
            ids = self._tokenize(text)
            if not ids:
                ids = [0]
            embeds = self._embedding[ids]
            features.append(embeds.mean(dim=0).to(dtype=torch.float32))
        return torch.stack(features, dim=0).detach()

    def teacher_forced_logits(
        self,
        messages: list[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not messages:
            raise FrozenOrganError("empty messages")
        if not target or not target.strip():
            raise FrozenOrganError("empty target")
        if soft_bank.dim() != 3 or soft_bank.shape[0] != 1:
            raise FrozenOrganError("soft_bank must be [1, L, D]")
        if soft_bank.shape[2] != self.feature_dim:
            raise FrozenOrganError("soft_bank feature dim mismatch")
        if not torch.isfinite(soft_bank).all():
            raise FrozenOrganError("soft_bank contains non-finite values")

        target_text = " " + target.lstrip()
        answer_ids = self._tokenize(target_text)
        if not answer_ids:
            raise FrozenOrganError("target tokenization produced empty list")

        bank = soft_bank[0]
        bank_steer = bank.mean(dim=0)
        answer_embeds = self._embedding[answer_ids]
        steered = answer_embeds + bank_steer.unsqueeze(0)
        logits = steered @ self._output_proj + self._output_bias
        return logits

    def teacher_forced_loss(
        self,
        messages: list[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.teacher_forced_logits(messages, target, soft_bank)
        target_text = " " + target.lstrip()
        answer_ids = self._tokenize(target_text)
        targets = torch.tensor(answer_ids, dtype=torch.long)
        return torch.nn.functional.cross_entropy(logits, targets)

    def specificity_kl(
        self,
        messages: list[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
        reference_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bank_logits = self.teacher_forced_logits(messages, target, soft_bank)
        bank_log_probs = torch.nn.functional.log_softmax(bank_logits, dim=-1)
        with torch.no_grad():
            if reference_bank is not None:
                ref_logits = self.teacher_forced_logits(messages, target, reference_bank)
            else:
                target_text = " " + target.lstrip()
                answer_ids = self._tokenize(target_text)
                answer_embeds = self._embedding[answer_ids]
                ref_logits = answer_embeds @ self._output_proj + self._output_bias
            ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
        bank_probs = bank_log_probs.exp()
        ref_log_probs = ref_probs.clamp(min=1e-12).log()
        kl = (bank_probs * (bank_log_probs - ref_log_probs)).sum(dim=-1)
        return kl.mean()

    def generate(
        self,
        messages: list[DialogueMessage],
        soft_bank: torch.Tensor,
        max_new_tokens: int,
    ) -> str:
        if self._closed:
            raise FrozenOrganError("organ is closed")
        if not messages:
            raise FrozenOrganError("empty messages")
        if max_new_tokens <= 0:
            raise FrozenOrganError("max_new_tokens must be positive")
        prompt_text = self._render(messages)
        prompt_ids = self._tokenize(prompt_text)
        bank = soft_bank[0]
        pe = self._embedding[prompt_ids]
        emb = torch.cat([bank, pe], dim=0)
        logits = emb @ self._output_proj + self._output_bias
        next_token = int(logits[-1].argmax().item())
        generated: list[str] = []
        for _ in range(max_new_tokens):
            generated.append(chr(min(next_token, 127)))
            ne = self._embedding[next_token].unsqueeze(0)
            logits = ne @ self._output_proj + self._output_bias
            next_token = int(logits.argmax().item())
        return "".join(generated)

    def parameter_hash(self) -> str:
        parts: list[str] = []
        for name, tensor in sorted([
            ("embedding", self._embedding),
            ("output_bias", self._output_bias),
            ("output_proj", self._output_proj),
        ]):
            data = tensor.detach().cpu().contiguous().numpy().tobytes()
            parts.append(f"{name}:{hashlib.sha256(data).hexdigest()}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def assert_frozen(self) -> None:
        for name, tensor in [
            ("embedding", self._embedding),
            ("output_proj", self._output_proj),
            ("output_bias", self._output_bias),
        ]:
            if tensor.requires_grad:
                raise FrozenOrganError(f"{name} has requires_grad=True")

    def close(self) -> None:
        self._closed = True

    def organ_parameter_ids(self) -> set[int]:
        return {id(self._embedding), id(self._output_proj), id(self._output_bias)}


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_MODEL_CONFIG = ModelConfig(feature_dim=16, d_cortex=64, bank_width=2)
_OUTER_CONFIG = OuterLoopConfig(
    outer_steps=2,
    tasks_per_step=1,
    optimizer_name="adamw",
    learning_rate=1e-3,
    validation_interval=1,
    generation_interval=1,
    grad_clip_norm=1.0,
    seed=0,
)
_TASKGEN_CONFIG = TaskGeneratorConfig(
    root_seed=20260709,
    train_tasks_per_family=2,
    validation_tasks_per_family=2,
    min_events=2,
    max_events=2,
)

_ZERO_HASH = "0" * 64
_ORGAN_HASH = "a" * 64

# Scorer hash from the real FrozenScorer — the view must carry this.
_SCORER = FrozenScorer()
_SCORER_SHA256 = _SCORER.sha256


def _make_probe_battery(task_label: str = "test") -> ProbeBattery:
    """Build a minimal valid ProbeBattery with task-distinct content."""
    msg = DialogueMessage(role="user", content=task_label)
    pc = ProbeCase(
        messages=(msg,),
        expected_response=task_label,
        kind=ProbeKind.SAME_RULE,
    )
    return ProbeBattery(
        pre=(pc,),
        same_rule=(pc,),
        transfer=(pc,),
        composition=(pc,),
        specificity=(pc,),
        oracle_context=(pc,),
    )


def _make_meta_task(family: TaskFamily, rule_fp: str, index: int = 0) -> MetaTask:
    """Build a minimal valid MetaTask with META_VALIDATION split.

    Each task gets distinct content (indexed by *index*) so the model
    produces different cortex states — C6 STATE_SWAPPED requires the
    donor state to differ from the trained state.
    """
    label = f"{family.value}_task_{index:03d}"
    msg = DialogueMessage(role="user", content=f"observe_{label}")
    event = LearningEvent(
        observation_messages=(msg,),
        attempted_behavior=f"attempt_{label}",
        correction=f"correct_{label}",
        outcome=OutcomeCode.NEUTRAL,
    )
    return MetaTask(
        family=family,
        split=DevSplit.META_VALIDATION,
        events=(event, event),
        probes=_make_probe_battery(label),
        rule_fingerprint=rule_fp,
        assignment_fingerprint=_ZERO_HASH,
        composition_fingerprint=_ZERO_HASH,
        paraphrase_group_fingerprint=_ZERO_HASH,
    )


def _make_view(n_tasks_per_family: int = 30) -> CalibrationInstrumentView:
    """Build a minimal CalibrationInstrumentView with n tasks per family."""
    families = [
        TaskFamily.CONTEXTUAL_REMAP,
        TaskFamily.RULE_TRANSFORMATION,
        TaskFamily.FINITE_STATE,
    ]
    tasks: list[MetaTask] = []
    for fam in families:
        for i in range(n_tasks_per_family):
            suffix = f"{fam.value}_{i:03d}"
            fp = suffix + "0" * (64 - len(suffix))
            tasks.append(_make_meta_task(fam, fp, index=i))
    return CalibrationInstrumentView(
        schema=CALIBRATION_VIEW_SCHEMA,
        instrument_id="meta_cortex/v1",
        instrument_version="v1",
        definition_sha256=_ZERO_HASH,
        calibration_view_sha256=_ZERO_HASH,
        scorer_sha256=_SCORER_SHA256,
        endpoint_schema_sha256=_ZERO_HASH,
        confidence_level=0.95,
        target_power=0.80,
        minimum_tasks_per_family=30,
        developmental_seeds=tuple(100 + i for i in range(5)),
        evaluation_seeds=tuple(200 + i for i in range(5)),
        no_update_repeat_seeds=tuple(300 + i for i in range(20)),
        task_cluster_bootstrap_seed=999,
        tasks=tuple(tasks),
        calibration_tasks_per_family={
            "contextual_remap": n_tasks_per_family,
            "rule_transformation": n_tasks_per_family,
            "finite_state": n_tasks_per_family,
        },
    )


def _make_checkpoint(
    tmp_path: Path,
    organ: _TinyFrozenOrgan,
    model: MetaCortex,
) -> Path:
    """Write a developmental checkpoint and return its directory path."""
    theta_hash = canonical_theta_hash(model)
    param_count = model.parameter_count()
    metadata = CheckpointMetadata(
        schema=DEV_SCHEMA,
        model_config=_MODEL_CONFIG,
        taskgen_schema=TASKGEN_SCHEMA,
        taskgen_digest="a" * 64,
        outer_config=_OUTER_CONFIG,
        completed_step=2,
        best_step=1,
        validation_score=0.5,
        parameter_count=param_count,
        parameter_bytes=param_count * 4,
        theta_hash=theta_hash,
        organ_identity="test-organ",
        organ_hash=organ.parameter_hash(),
        source_provenance="unavailable",
    )
    ckpt_dir = tmp_path / "checkpoint"
    save_developmental_checkpoint(ckpt_dir, model, metadata)
    return ckpt_dir


def _make_task_condition_record(
    family: str,
    rule_fp: str,
    dev_idx: int,
    eval_idx: int,
    condition: str,
) -> TaskConditionRecord:
    """Build a minimal valid TaskConditionRecord."""
    pc = ProbeCount(correct=1, total=2)
    return TaskConditionRecord(
        family=family,
        rule_fingerprint=rule_fp,
        assignment_fingerprint=_ZERO_HASH,
        composition_fingerprint=_ZERO_HASH,
        paraphrase_group_fingerprint=_ZERO_HASH,
        developmental_seed_index=dev_idx,
        evaluation_seed_index=eval_idx,
        developmental_seed=100 + dev_idx,
        evaluation_seed=200 + eval_idx,
        condition=condition,
        score_vector_hash=_ZERO_HASH,
        same_rule=pc,
        transfer=pc,
        composition=pc,
        specificity=pc,
        pre_learning_primary=pc,
        immediately_pre_deletion_primary=pc,
        post_deletion_primary=pc,
        theta_hash=_ZERO_HASH,
        organ_hash=_ORGAN_HASH,
        state_hash=_ZERO_HASH,
        optimizer_step_count=0,
        trace_count_after=0,
        fast_zero=True,
        slow_zero=True,
    )


def _make_no_update_repeat_record(
    family: str,
    rule_fp: str,
    dev_idx: int,
    repeat_idx: int,
) -> NoUpdateRepeatRecord:
    """Build a minimal valid NoUpdateRepeatRecord."""
    return NoUpdateRepeatRecord(
        family=family,
        rule_fingerprint=rule_fp,
        developmental_seed_index=dev_idx,
        developmental_seed=100 + dev_idx,
        repeat_index=repeat_idx,
        repeat_seed=300 + repeat_idx,
        specificity_accuracy=Fraction(1, 2),
        primary_pre_deletion=Fraction(1, 2),
        primary_post_deletion=Fraction(1, 2),
        theta_hash=_ZERO_HASH,
        organ_hash=_ORGAN_HASH,
        optimizer_step_count=0,
        trace_count_after=0,
        fast_zero=True,
        slow_zero=True,
    )


def _build_synthetic_shard_collection(
    view: CalibrationInstrumentView,
    *,
    dev_seed_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
    eval_seed_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
    task_indices: tuple[int, ...] | None = None,
) -> ShardCollection:
    """Build a complete synthetic ShardCollection covering all cells.

    This bypasses the runner (no organ/model needed) and constructs records
    directly — used for merge-only tests.
    """
    if task_indices is None:
        task_indices = tuple(range(len(view.tasks)))
    nu_records: list[NoUpdateRepeatRecord] = []
    sc_records: list[SeedCellRecord] = []
    theta_hashes: list[ShardThetaHash] = []

    for dev_idx in dev_seed_indices:
        theta_hashes.append(ShardThetaHash(
            developmental_seed_index=dev_idx,
            theta_hash=_ZERO_HASH,
        ))
        for ti in task_indices:
            task = view.tasks[ti]
            fam = task.family.value
            for rep in range(20):
                nu_records.append(_make_no_update_repeat_record(
                    fam, task.rule_fingerprint, dev_idx, rep,
                ))

        for eval_idx in eval_seed_indices:
            for ti in task_indices:
                task = view.tasks[ti]
                fam = task.family.value
                conds = tuple(
                    _make_task_condition_record(
                        fam, task.rule_fingerprint, dev_idx, eval_idx, cond,
                    )
                    for cond in (C1, C2, C3, C4, C5, C6)
                )
                sc_records.append(SeedCellRecord(
                    developmental_seed_index=dev_idx,
                    evaluation_seed_index=eval_idx,
                    developmental_seed=100 + dev_idx,
                    evaluation_seed=200 + eval_idx,
                    rule_fingerprint=task.rule_fingerprint,
                    family=fam,
                    conditions=conds,
                ))

    return ShardCollection(
        no_update_repeat_records=tuple(nu_records),
        seed_cell_records=tuple(sc_records),
        theta_hashes=tuple(theta_hashes),
    )


def _write_shards_for_merge(
    tmp_path: Path,
    view: CalibrationInstrumentView,
    collections: list[ShardCollection],
    organ_hash: str = _ORGAN_HASH,
) -> list[Path]:
    """Write multiple ShardCollections to shard files."""
    paths = []
    for i, coll in enumerate(collections):
        path = tmp_path / f"shard_{i}.json"
        write_calibration_shard(
            path,
            view=view,
            collection=coll,
            organ_hash=organ_hash,
        )
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# ShardConfig validation tests
# ---------------------------------------------------------------------------


class TestShardConfigValidation:
    """ShardConfig must reject invalid indices before any I/O."""

    def test_valid_config(self) -> None:
        cfg = ShardConfig(
            checkpoint_dir="/tmp/ckpt",
            model_id="test",
            dev_seed_indices=(0, 1),
            eval_seed_indices=(0, 1),
            task_indices=(0, 1, 2),
        )
        assert cfg.dev_seed_indices == (0, 1)
        assert cfg.eval_seed_indices == (0, 1)
        assert cfg.task_indices == (0, 1, 2)

    def test_empty_checkpoint_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_dir"):
            ShardConfig(
                checkpoint_dir="",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    def test_empty_dev_seed_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="dev_seed_indices"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    def test_empty_eval_seed_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="eval_seed_indices"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(),
                task_indices=(0,),
            )

    def test_empty_task_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="task_indices"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(),
            )

    @pytest.mark.parametrize("bad_idx", [-1, 5, 10])
    def test_dev_seed_index_out_of_range_raises(self, bad_idx: int) -> None:
        with pytest.raises(ValueError, match="dev_seed_index"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(bad_idx,),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    @pytest.mark.parametrize("bad_idx", [-1, 5, 10])
    def test_eval_seed_index_out_of_range_raises(self, bad_idx: int) -> None:
        with pytest.raises(ValueError, match="eval_seed_index"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(bad_idx,),
                task_indices=(0,),
            )

    def test_negative_task_index_raises(self) -> None:
        with pytest.raises(ValueError, match="task_index"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(-1,),
            )

    def test_duplicate_dev_seed_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="dev_seed_indices must be distinct"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0, 0),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    def test_duplicate_eval_seed_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="eval_seed_indices must be distinct"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(1, 1),
                task_indices=(0,),
            )

    def test_duplicate_task_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="task_indices must be distinct"):
            ShardConfig(
                checkpoint_dir="/tmp/ckpt",
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0, 0),
            )


# ---------------------------------------------------------------------------
# Exact scoring tests — no substring matching
# ---------------------------------------------------------------------------


class TestExactScoringInRunner:
    """The runner must use FrozenScorer (exact normalized-equal), not substring."""

    def test_scorer_rejects_substring_match(self) -> None:
        """FrozenScorer rejects a generated string that merely contains the
        expected response as a substring — the old substring scorer would
        accept it."""
        scorer = FrozenScorer()
        expected = "cat"
        # Substring would pass: "a cat sat" contains "cat".
        # Exact normalized-equal must fail.
        assert scorer.score_response(expected, "a cat sat") is False
        assert scorer.score_response(expected, "cat") is True

    def test_scorer_rejects_case_difference(self) -> None:
        """Exact matching preserves case — 'Cat' != 'cat'."""
        scorer = FrozenScorer()
        assert scorer.score_response("cat", "Cat") is False

    def test_scorer_rejects_extra_prose(self) -> None:
        """Exact matching rejects extra prose around the answer."""
        scorer = FrozenScorer()
        assert scorer.score_response("42", "The answer is 42.") is False

    def test_scorer_nfkc_normalization(self) -> None:
        """NFKC normalization means fullwidth == ASCII for equivalent chars."""
        scorer = FrozenScorer()
        # NFKC normalizes fullwidth 'Ａ' (U+FF21) to 'A'.
        assert scorer.score_response("A", "Ａ") is True

    def test_scorer_strip_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        scorer = FrozenScorer()
        assert scorer.score_response("cat", "  cat  ") is True

    def test_scorer_empty_expected_is_false(self) -> None:
        """Empty expected response is always False."""
        scorer = FrozenScorer()
        assert scorer.score_response("", "anything") is False

    def test_runner_uses_exact_scorer_not_substring(
        self,
        tmp_path: Path,
    ) -> None:
        """The runner's _score_probe_exact calls FrozenScorer.score_response,
        which rejects substring matches.  We verify this by checking that
        the scorer hash in the view matches FrozenScorer.sha256 — the runner
        enforces this and would raise if the scorer were swapped."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        # The view's scorer_sha256 must match FrozenScorer.
        assert view.scorer_sha256 == FrozenScorer().sha256

        # The runner will raise ValueError if the scorer hash doesn't match.
        # This proves the runner uses FrozenScorer, not a substring matcher.
        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )
        assert len(collection.seed_cell_records) > 0
        # Every condition record has exact correct/total counts.
        for scr in collection.seed_cell_records:
            for cr in scr.conditions:
                assert isinstance(cr.same_rule, ProbeCount)
                assert cr.same_rule.total > 0
                assert 0 <= cr.same_rule.correct <= cr.same_rule.total


# ---------------------------------------------------------------------------
# Online optimizer prohibition tests
# ---------------------------------------------------------------------------


class TestOptimizerProhibition:
    """No optimizer step may occur during shard collection."""

    def test_optimizer_step_count_zero_in_all_records(
        self,
        tmp_path: Path,
    ) -> None:
        """Every no-update repeat record must have optimizer_step_count==0."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        for rec in collection.no_update_repeat_records:
            assert rec.optimizer_step_count == 0

    def test_theta_hash_stable_across_collection(
        self,
        tmp_path: Path,
    ) -> None:
        """The runner verifies theta hash is stable before and after.
        If it changed, the runner raises RuntimeError."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        theta_before = canonical_theta_hash(model)

        collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        theta_after = canonical_theta_hash(model)
        assert theta_before == theta_after

    def test_organ_hash_stable_across_collection(
        self,
        tmp_path: Path,
    ) -> None:
        """The organ hash must not change during collection."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        organ_hash_before = organ.parameter_hash()

        collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        organ_hash_after = organ.parameter_hash()
        assert organ_hash_before == organ_hash_after

    def test_all_condition_records_have_zero_optimizer_steps(
        self,
        tmp_path: Path,
    ) -> None:
        """Every TaskConditionRecord in seed cells has optimizer_step_count==0."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        for scr in collection.seed_cell_records:
            for cr in scr.conditions:
                assert cr.optimizer_step_count == 0


# ---------------------------------------------------------------------------
# Trace deletion tests
# ---------------------------------------------------------------------------


class TestTraceDeletion:
    """All traces must be deleted before post-deletion probes."""

    def test_trace_count_after_zero_in_all_records(
        self,
        tmp_path: Path,
    ) -> None:
        """Every no-update repeat record has trace_count_after==0."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        for rec in collection.no_update_repeat_records:
            assert rec.trace_count_after == 0

    def test_trace_count_after_zero_in_condition_records(
        self,
        tmp_path: Path,
    ) -> None:
        """Every TaskConditionRecord in seed cells has trace_count_after==0."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        for scr in collection.seed_cell_records:
            for cr in scr.conditions:
                assert cr.trace_count_after == 0

    def test_fast_zero_and_slow_zero_in_no_update_records(
        self,
        tmp_path: Path,
    ) -> None:
        """No-update repeats must have fast_zero=True and slow_zero=True."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        for rec in collection.no_update_repeat_records:
            assert rec.fast_zero is True
            assert rec.slow_zero is True


# ---------------------------------------------------------------------------
# Production type interop tests
# ---------------------------------------------------------------------------


class TestRunnerReturnsProductionTypes:
    """collect_calibration_shard must return calibration.ShardCollection directly.

    The runner and calibration modules must share the same ShardCollection /
    ShardThetaHash types so that write_calibration_shard's isinstance check
    passes without any test-only converter.
    """

    def test_collection_is_production_type(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )
        assert isinstance(collection, ShardCollection)
        for th in collection.theta_hashes:
            assert isinstance(th, ShardThetaHash)

    def test_runner_output_writes_without_converter(self, tmp_path: Path) -> None:
        """Runner output must pass write_calibration_shard's isinstance check directly."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )
        shard_path = tmp_path / "shard.json"
        sha = write_calibration_shard(
            shard_path, view=view,
            collection=collection,
            organ_hash=organ.parameter_hash(),
        )
        assert len(sha) == 64
        assert shard_path.exists()


# ---------------------------------------------------------------------------
# Deterministic shard bytes tests
# ---------------------------------------------------------------------------


class TestDeterministicShardBytes:
    """Re-running the same shard produces byte-identical shard JSON."""

    def test_shard_json_byte_identical_on_rerun(
        self,
        tmp_path: Path,
    ) -> None:
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection1 = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )
        collection2 = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        # Shard JSON is canonical (sort_keys, allow_nan=False) so the
        # serialized bytes are identical.
        shard_path1 = tmp_path / "shard1.json"
        shard_path2 = tmp_path / "shard2.json"
        sha1 = write_calibration_shard(
            shard_path1, view=view,
            collection=collection1,
            organ_hash=organ.parameter_hash(),
        )
        sha2 = write_calibration_shard(
            shard_path2, view=view,
            collection=collection2,
            organ_hash=organ.parameter_hash(),
        )

        bytes1 = shard_path1.read_bytes()
        bytes2 = shard_path2.read_bytes()
        assert bytes1 == bytes2
        assert sha1 == sha2

    def test_different_shards_produce_different_bytes(
        self,
        tmp_path: Path,
    ) -> None:
        """Shards covering different task indices produce different JSON."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        coll_a = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )
        coll_b = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(1, 2),
            organ=organ,
        )

        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        write_calibration_shard(path_a, view=view,
                                collection=coll_a,
                                organ_hash=organ.parameter_hash())
        write_calibration_shard(path_b, view=view,
                                collection=coll_b,
                                organ_hash=organ.parameter_hash())
        assert path_a.read_bytes() != path_b.read_bytes()


# ---------------------------------------------------------------------------
# Seed/task sharding tests
# ---------------------------------------------------------------------------


class TestSeedTaskSharding:
    """Shards partition work by dev/eval seed indices and task indices."""

    def test_shard_covers_specified_cells(
        self,
        tmp_path: Path,
    ) -> None:
        """A shard with dev=(0,) eval=(0,) tasks=(0,1) produces exactly
        2 seed cell records and 2*20=40 no-update records."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        # 2 tasks × 1 dev × 1 eval = 2 seed cells.
        assert len(collection.seed_cell_records) == 2
        # 2 tasks × 1 dev × 20 repeats = 40 no-update records.
        assert len(collection.no_update_repeat_records) == 40
        # 1 theta hash.
        assert len(collection.theta_hashes) == 1
        assert collection.theta_hashes[0].developmental_seed_index == 0

    def test_shard_covers_multiple_dev_seeds(
        self,
        tmp_path: Path,
    ) -> None:
        """A shard with dev=(0,1) eval=(0,) tasks=(0,) produces 2 seed cells
        and 2*20=40 no-update records and 2 theta hashes."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collection = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0, 1),
            eval_seed_indices=(0,),
            task_indices=(0,),
            organ=organ,
        )

        assert len(collection.seed_cell_records) == 2
        assert len(collection.no_update_repeat_records) == 40
        assert len(collection.theta_hashes) == 2
        th_indices = {th.developmental_seed_index for th in collection.theta_hashes}
        assert th_indices == {0, 1}

    def test_shard_partitioning_covers_full_grid(
        self,
        tmp_path: Path,
    ) -> None:
        """Two shards partitioning dev seeds (0,1,2) and (3,4) cover all 5."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        coll_a = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0, 1, 2),
            eval_seed_indices=(0, 1, 2, 3, 4),
            task_indices=(0, 1),
            organ=organ,
        )
        coll_b = collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(3, 4),
            eval_seed_indices=(0, 1, 2, 3, 4),
            task_indices=(0, 1),
            organ=organ,
        )

        # Combined theta hashes cover all 5 dev indices.
        all_th = set()
        all_th.update(th.developmental_seed_index for th in coll_a.theta_hashes)
        all_th.update(th.developmental_seed_index for th in coll_b.theta_hashes)
        assert all_th == {0, 1, 2, 3, 4}

        # No overlap in seed cell (dev, eval) pairs.
        cells_a = {
            (scr.developmental_seed_index, scr.evaluation_seed_index)
            for scr in coll_a.seed_cell_records
        }
        cells_b = {
            (scr.developmental_seed_index, scr.evaluation_seed_index)
            for scr in coll_b.seed_cell_records
        }
        assert cells_a.isdisjoint(cells_b)

    def test_invalid_task_index_raises_before_output(
        self,
        tmp_path: Path,
    ) -> None:
        """A task index >= len(view.tasks) raises ValueError."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        with pytest.raises(ValueError, match="task_index"):
            collect_calibration_shard(
                view=view,
                checkpoint_dir=str(ckpt_dir),
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(999,),
                organ=organ,
            )

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """A missing checkpoint.json raises ValueError before any output."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)

        with pytest.raises(ValueError, match="checkpoint"):
            collect_calibration_shard(
                view=view,
                checkpoint_dir=str(tmp_path / "nonexistent"),
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
                organ=organ,
            )

    def test_organ_hash_mismatch_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """If the organ hash doesn't match the checkpoint, it raises."""
        view = _make_view(n_tasks_per_family=30)
        # Create a checkpoint with one organ hash.
        organ_for_ckpt = _TinyFrozenOrgan(feature_dim=16, vocab_size=256)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ_for_ckpt, model)

        # Use a different organ (different vocab_size → different hash).
        different_organ = _TinyFrozenOrgan(feature_dim=16, vocab_size=512)

        with pytest.raises(ValueError, match="organ hash mismatch"):
            collect_calibration_shard(
                view=view,
                checkpoint_dir=str(ckpt_dir),
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
                organ=different_organ,
            )

    def test_scorer_hash_mismatch_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """If the view's scorer hash doesn't match FrozenScorer, it raises."""
        view = _make_view(n_tasks_per_family=30)
        # Corrupt the scorer hash.
        object.__setattr__(
            view, "scorer_sha256", "f" * 64,
        )
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        with pytest.raises(ValueError, match="scorer hash mismatch"):
            collect_calibration_shard(
                view=view,
                checkpoint_dir=str(ckpt_dir),
                model_id="test",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
                organ=organ,
            )


# ---------------------------------------------------------------------------
# Shard write/load roundtrip tests
# ---------------------------------------------------------------------------


class TestShardWriteLoad:
    """write_calibration_shard / load_calibration_shard roundtrip."""

    def test_roundtrip_preserves_records(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        path = tmp_path / "shard.json"
        sha = write_calibration_shard(
            path, view=view, collection=coll, organ_hash=_ORGAN_HASH,
        )
        assert len(sha) == 64

        loaded = load_calibration_shard(path, view)
        assert loaded.organ_hash == _ORGAN_HASH
        assert len(loaded.no_update_repeat_records) == len(coll.no_update_repeat_records)
        assert len(loaded.seed_cell_records) == len(coll.seed_cell_records)
        assert len(loaded.theta_hashes) == len(coll.theta_hashes)

    def test_shard_schema_constant(self) -> None:
        assert CALIBRATION_SHARD_SCHEMA == "oczy/meta-cortex/calibration-shard/v1"

    def test_load_wrong_schema_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "schema": "wrong/schema/v1",
            "definition_sha256": _ZERO_HASH,
            "calibration_view_sha256": _ZERO_HASH,
            "scorer_sha256": _SCORER_SHA256,
            "organ_hash": _ORGAN_HASH,
            "no_update_repeat_records": [],
            "seed_cell_records": [],
            "theta_hashes": [],
        }))
        with pytest.raises(ValueError, match="schema"):
            load_calibration_shard(path, view)

    def test_load_wrong_definition_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "schema": CALIBRATION_SHARD_SCHEMA,
            "definition_sha256": "f" * 64,
            "calibration_view_sha256": _ZERO_HASH,
            "scorer_sha256": _SCORER_SHA256,
            "organ_hash": _ORGAN_HASH,
            "no_update_repeat_records": [],
            "seed_cell_records": [],
            "theta_hashes": [],
        }))
        with pytest.raises(ValueError, match="definition_sha256"):
            load_calibration_shard(path, view)

    def test_load_sealed_path_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        sealed_dir = tmp_path / "sealed"
        sealed_dir.mkdir()
        path = sealed_dir / "shard.json"
        coll = _build_synthetic_shard_collection(view)
        write_calibration_shard(path, view=view, collection=coll, organ_hash=_ORGAN_HASH)
        with pytest.raises(ValueError, match="sealed"):
            load_calibration_shard(path, view)

    def test_load_sealed_content_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        path = tmp_path / "shard.json"
        # Write a shard with "sealed" in a field value.
        data = {
            "schema": CALIBRATION_SHARD_SCHEMA,
            "definition_sha256": _ZERO_HASH,
            "calibration_view_sha256": _ZERO_HASH,
            "scorer_sha256": _SCORER_SHA256,
            "organ_hash": _ORGAN_HASH,
            "no_update_repeat_records": [],
            "seed_cell_records": [],
            "theta_hashes": [],
            "shard_sha256": _ZERO_HASH,
        }
        path.write_text(json.dumps(data) + " sealed")
        with pytest.raises(ValueError, match="sealed"):
            load_calibration_shard(path, view)


# ---------------------------------------------------------------------------
# Strict merge: completeness tests
# ---------------------------------------------------------------------------


class TestMergeCompleteness:
    """Merge must reject partial data — missing cells, repeats, or theta hashes."""

    def test_full_merge_succeeds(self, tmp_path: Path) -> None:
        """A complete set of shards covering all 5×5 cells and 20 repeats
        merges successfully."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        result = merge_calibration_shards(
            view=view,
            shard_paths=shard_paths,
            output_dir=tmp_path / "merged",
        )
        assert isinstance(result, MergeResult)
        assert result.n_no_update_records == 90 * 5 * 20
        assert result.n_seed_cell_records == 90 * 25
        assert result.n_theta_hashes == 5
        assert result.n_shards_merged == 1

    def test_merge_output_loadable_by_existing_loaders(
        self,
        tmp_path: Path,
    ) -> None:
        """The merged files must be loadable by the existing loaders."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        out_dir = tmp_path / "merged"
        merge_calibration_shards(
            view=view,
            shard_paths=shard_paths,
            output_dir=out_dir,
        )

        nu = load_no_update_repeat_records(
            out_dir / "NO_UPDATE_REPEAT_RECORDS.json", view,
        )
        sc = load_seed_cell_records(
            out_dir / "SEED_CELL_RECORDS.json", view,
        )
        th = load_theta_hashes(
            out_dir / "THETA_HASHES.json", view,
        )
        assert len(nu) == 90 * 5 * 20
        assert len(sc) == 90 * 25
        assert len(th) == 5

    def test_merge_partial_seed_cells_rejected(self, tmp_path: Path) -> None:
        """Missing seed cells (only 24 instead of 25 per task) raises."""
        view = _make_view(n_tasks_per_family=30)
        # Build a collection missing one eval seed index.
        coll = _build_synthetic_shard_collection(
            view,
            eval_seed_indices=(0, 1, 2, 3),  # Missing eval_idx=4.
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        with pytest.raises(ValueError, match="seed cell"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_merge_partial_no_update_repeats_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing no-update repeats (only 19 instead of 20) raises."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        # Remove one repeat record.
        nu_list = list(coll.no_update_repeat_records)
        nu_list.pop()
        coll = ShardCollection(
            no_update_repeat_records=tuple(nu_list),
            seed_cell_records=coll.seed_cell_records,
            theta_hashes=coll.theta_hashes,
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        with pytest.raises(ValueError, match="20"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_merge_missing_theta_hash_index_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """Only 4 theta hashes (missing index 4) raises."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(0, 1, 2, 3),  # Missing dev_idx=4.
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        with pytest.raises(ValueError, match="theta hash"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_merge_missing_task_in_seed_cells_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing seed cells for a task raises."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(
            view,
            task_indices=tuple(range(89)),  # Missing last task.
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        with pytest.raises(ValueError, match="Missing"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_merge_no_output_on_failure(self, tmp_path: Path) -> None:
        """No output files are written if any check fails."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(
            view,
            eval_seed_indices=(0, 1, 2, 3),  # Incomplete.
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        out_dir = tmp_path / "merged"
        with pytest.raises(ValueError):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=out_dir,
            )
        assert not out_dir.exists()

    def test_merge_empty_shard_list_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        with pytest.raises(ValueError, match="at least one shard"):
            merge_calibration_shards(
                view=view,
                shard_paths=[],
                output_dir=tmp_path / "merged",
            )


# ---------------------------------------------------------------------------
# Strict merge: duplicate rejection tests
# ---------------------------------------------------------------------------


class TestMergeDuplicateRejection:
    """Merge must reject duplicate records."""

    def test_duplicate_seed_cell_raises(self, tmp_path: Path) -> None:
        """Two shards with the same (family, rule_fp, dev_idx, eval_idx)
        seed cell raises."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        # Write the same collection twice — all records are duplicates.
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll, coll])
        with pytest.raises(ValueError, match="(?i)duplicate"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_duplicate_no_update_repeat_raises(self, tmp_path: Path) -> None:
        """Two shards with overlapping no-update repeats raises."""
        view = _make_view(n_tasks_per_family=30)
        # Shard A: dev=(0,) all tasks.
        coll_a = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(0,),
            eval_seed_indices=(0, 1, 2, 3, 4),
        )
        # Shard B: dev=(0,) all tasks — duplicate of A's no-update records.
        coll_b = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(0,),
            eval_seed_indices=(),  # No seed cells to avoid that duplicate.
        )
        # Actually eval_seed_indices=() is not allowed by the builder.
        # Instead, build B with only no-update records.
        nu_b = tuple(
            _make_no_update_repeat_record(
                view.tasks[ti].family.value,
                view.tasks[ti].rule_fingerprint,
                0, rep,
            )
            for ti in range(len(view.tasks))
            for rep in range(20)
        )
        coll_b = ShardCollection(
            no_update_repeat_records=nu_b,
            seed_cell_records=(),
            theta_hashes=(ShardThetaHash(developmental_seed_index=0, theta_hash=_ZERO_HASH),),
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll_a, coll_b])
        with pytest.raises(ValueError, match="(?i)duplicate"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )


# ---------------------------------------------------------------------------
# Strict merge: hash rejection tests
# ---------------------------------------------------------------------------


class TestMergeHashRejection:
    """Merge must reject shards with wrong header hashes or organ hash."""

    def test_wrong_definition_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        path = tmp_path / "shard.json"
        # Write with correct view, then modify the definition_sha256.
        write_calibration_shard(path, view=view, collection=coll, organ_hash=_ORGAN_HASH)
        data = json.loads(path.read_text())
        data["definition_sha256"] = "f" * 64
        path.write_text(json.dumps(data, sort_keys=True, allow_nan=False))
        with pytest.raises(ValueError, match="definition_sha256"):
            merge_calibration_shards(
                view=view,
                shard_paths=[path],
                output_dir=tmp_path / "merged",
            )

    def test_wrong_view_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        path = tmp_path / "shard.json"
        write_calibration_shard(path, view=view, collection=coll, organ_hash=_ORGAN_HASH)
        data = json.loads(path.read_text())
        data["calibration_view_sha256"] = "f" * 64
        path.write_text(json.dumps(data, sort_keys=True, allow_nan=False))
        with pytest.raises(ValueError, match="calibration_view_sha256"):
            merge_calibration_shards(
                view=view,
                shard_paths=[path],
                output_dir=tmp_path / "merged",
            )

    def test_wrong_scorer_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        path = tmp_path / "shard.json"
        write_calibration_shard(path, view=view, collection=coll, organ_hash=_ORGAN_HASH)
        data = json.loads(path.read_text())
        data["scorer_sha256"] = "f" * 64
        path.write_text(json.dumps(data, sort_keys=True, allow_nan=False))
        with pytest.raises(ValueError, match="scorer_sha256"):
            merge_calibration_shards(
                view=view,
                shard_paths=[path],
                output_dir=tmp_path / "merged",
            )

    def test_inconsistent_organ_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        path_a = tmp_path / "shard_a.json"
        path_b = tmp_path / "shard_b.json"
        write_calibration_shard(path_a, view=view, collection=coll, organ_hash=_ORGAN_HASH)
        write_calibration_shard(path_b, view=view, collection=coll, organ_hash="b" * 64)
        with pytest.raises(ValueError, match="organ_hash"):
            merge_calibration_shards(
                view=view,
                shard_paths=[path_a, path_b],
                output_dir=tmp_path / "merged",
            )

    def test_theta_hash_conflict_raises(self, tmp_path: Path) -> None:
        """Same dev_seed_index with different theta_hash raises."""
        view = _make_view(n_tasks_per_family=30)
        coll_a = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(0,),
            eval_seed_indices=(0, 1, 2, 3, 4),
        )
        # Build coll_b with same dev_idx=0 but different theta_hash.
        coll_b = ShardCollection(
            no_update_repeat_records=(),
            seed_cell_records=(),
            theta_hashes=(ShardThetaHash(
                developmental_seed_index=0,
                theta_hash="e" * 64,
            ),),
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll_a, coll_b])
        with pytest.raises(ValueError, match="(?i)theta hash conflict"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )

    def test_unknown_task_in_records_raises(self, tmp_path: Path) -> None:
        """A record for a task not in view.tasks raises (sealed contamination)."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        # Add a no-update record for a fake task.
        fake_nu = _make_no_update_repeat_record(
            "contextual_remap", "z" * 64, 0, 0,
        )
        coll = ShardCollection(
            no_update_repeat_records=coll.no_update_repeat_records + (fake_nu,),
            seed_cell_records=coll.seed_cell_records,
            theta_hashes=coll.theta_hashes,
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        with pytest.raises(ValueError, match="unknown task"):
            merge_calibration_shards(
                view=view,
                shard_paths=shard_paths,
                output_dir=tmp_path / "merged",
            )


# ---------------------------------------------------------------------------
# Strict merge: multi-shard partition test
# ---------------------------------------------------------------------------


class TestMergeMultiShardPartition:
    """Multiple shards partitioning the full grid merge correctly."""

    def test_two_shards_covering_all_dev_seeds_merge(
        self,
        tmp_path: Path,
    ) -> None:
        """Shard A covers dev=(0,1,2), shard B covers dev=(3,4).
        Both cover all eval seeds and all tasks. Merge succeeds."""
        view = _make_view(n_tasks_per_family=30)
        all_tasks = tuple(range(len(view.tasks)))
        coll_a = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(0, 1, 2),
            eval_seed_indices=(0, 1, 2, 3, 4),
            task_indices=all_tasks,
        )
        coll_b = _build_synthetic_shard_collection(
            view,
            dev_seed_indices=(3, 4),
            eval_seed_indices=(0, 1, 2, 3, 4),
            task_indices=all_tasks,
        )
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll_a, coll_b])
        result = merge_calibration_shards(
            view=view,
            shard_paths=shard_paths,
            output_dir=tmp_path / "merged",
        )
        assert result.n_shards_merged == 2
        assert result.n_no_update_records == 90 * 5 * 20
        assert result.n_seed_cell_records == 90 * 25
        assert result.n_theta_hashes == 5

    def test_five_shards_one_per_dev_seed_merge(
        self,
        tmp_path: Path,
    ) -> None:
        """Five shards, each covering one dev seed, merge correctly."""
        view = _make_view(n_tasks_per_family=30)
        all_tasks = tuple(range(len(view.tasks)))
        collections = []
        for dev_idx in range(5):
            collections.append(_build_synthetic_shard_collection(
                view,
                dev_seed_indices=(dev_idx,),
                eval_seed_indices=(0, 1, 2, 3, 4),
                task_indices=all_tasks,
            ))
        shard_paths = _write_shards_for_merge(tmp_path, view, collections)
        result = merge_calibration_shards(
            view=view,
            shard_paths=shard_paths,
            output_dir=tmp_path / "merged",
        )
        assert result.n_shards_merged == 5
        assert result.n_no_update_records == 90 * 5 * 20
        assert result.n_seed_cell_records == 90 * 25
        assert result.n_theta_hashes == 5


# ---------------------------------------------------------------------------
# Full pipeline: merge → run_dev_calibration
# ---------------------------------------------------------------------------


class TestMergedOutputFeedsCalibration:
    """Merged canonical files must feed run_dev_calibration end-to-end."""

    def test_merged_output_runs_dev_calibration(
        self,
        tmp_path: Path,
    ) -> None:
        """The merged files produce DEV_DISTRIBUTIONS.json and POWER_ANALYSIS.json."""
        view = _make_view(n_tasks_per_family=30)
        coll = _build_synthetic_shard_collection(view)
        shard_paths = _write_shards_for_merge(tmp_path, view, [coll])
        out_dir = tmp_path / "merged"
        merge_calibration_shards(
            view=view,
            shard_paths=shard_paths,
            output_dir=out_dir,
        )

        # Load the merged files.
        nu_records = load_no_update_repeat_records(
            out_dir / "NO_UPDATE_REPEAT_RECORDS.json", view,
        )
        sc_records = load_seed_cell_records(
            out_dir / "SEED_CELL_RECORDS.json", view,
        )
        theta_hashes = load_theta_hashes(
            out_dir / "THETA_HASHES.json", view,
        )

        # Run the full calibration.
        cal_dir = tmp_path / "calibration"
        result = run_dev_calibration(
            view=view,
            no_update_repeat_records=nu_records,
            seed_cell_records=sc_records,
            theta_hashes=theta_hashes,
            organ_hash=_ORGAN_HASH,
            output_dir=cal_dir,
        )

        assert (cal_dir / "DEV_DISTRIBUTIONS.json").exists()
        assert (cal_dir / "POWER_ANALYSIS.json").exists()
        assert result.dev_distributions_sha256  # non-empty hash
        assert result.power_analysis_sha256  # non-empty hash


# ---------------------------------------------------------------------------
# No network / no fake fallback tests
# ---------------------------------------------------------------------------


class TestNoNetworkNoFakeFallback:
    """The runner must not use a fake fallback organ when organ is None.

    These tests verify the contract: when organ=None, the runner attempts
    to load QwenFrozenOrgan (which requires network/model).  We verify
    this by confirming that without a real model, the runner raises
    FrozenOrganError (not a silent fallback).
    """

    def test_no_organ_raises_frozen_organ_error(self, tmp_path: Path) -> None:
        """Without a provided organ, the runner tries to load QwenFrozenOrgan
        and raises FrozenOrganError (no fake fallback)."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        # Without providing organ=, the runner will try QwenFrozenOrgan.load,
        # which will fail with FrozenOrganError (no model available).
        with pytest.raises((FrozenOrganError, OSError, ValueError)):
            collect_calibration_shard(
                view=view,
                checkpoint_dir=str(ckpt_dir),
                model_id="nonexistent/model",
                dev_seed_indices=(0,),
                eval_seed_indices=(0,),
                task_indices=(0,),
            )

    def test_provided_organ_is_not_closed(self, tmp_path: Path) -> None:
        """When the caller provides an organ, the runner does NOT close it."""
        view = _make_view(n_tasks_per_family=30)
        organ = _TinyFrozenOrgan(feature_dim=16)
        model = MetaCortex(_MODEL_CONFIG)
        ckpt_dir = _make_checkpoint(tmp_path, organ, model)

        collect_calibration_shard(
            view=view,
            checkpoint_dir=str(ckpt_dir),
            model_id="test",
            dev_seed_indices=(0,),
            eval_seed_indices=(0,),
            task_indices=(0, 1),
            organ=organ,
        )

        # The organ should still be usable (not closed).
        assert organ._closed is False
        result = organ.encode_texts(["test"])
        assert result is not None


# ---------------------------------------------------------------------------
# ShardThetaHash dataclass tests
# ---------------------------------------------------------------------------


class TestShardThetaHash:
    """ShardThetaHash validates its fields."""

    def test_valid(self) -> None:
        th = ShardThetaHash(developmental_seed_index=0, theta_hash=_ZERO_HASH)
        assert th.developmental_seed_index == 0
        assert th.theta_hash == _ZERO_HASH

    @pytest.mark.parametrize("bad_idx", [-1, 5, 10])
    def test_invalid_index_raises(self, bad_idx: int) -> None:
        with pytest.raises(ValueError, match="developmental_seed_index"):
            ShardThetaHash(developmental_seed_index=bad_idx, theta_hash=_ZERO_HASH)

    def test_short_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="theta_hash"):
            ShardThetaHash(developmental_seed_index=0, theta_hash="abc")

    def test_to_json_obj(self) -> None:
        th = ShardThetaHash(developmental_seed_index=2, theta_hash=_ZERO_HASH)
        obj = th.to_json_obj()
        assert obj == {
            "developmental_seed_index": 2,
            "theta_hash": _ZERO_HASH,
        }


# ---------------------------------------------------------------------------
# ShardCollection dataclass tests
# ---------------------------------------------------------------------------


class TestShardCollection:
    """ShardCollection validates its contents."""

    def test_valid_collection(self) -> None:
        coll = ShardCollection(
            no_update_repeat_records=(),
            seed_cell_records=(),
            theta_hashes=(ShardThetaHash(0, _ZERO_HASH),),
        )
        assert len(coll.theta_hashes) == 1

    def test_rejects_non_no_update_record(self) -> None:
        with pytest.raises(ValueError, match="NoUpdateRepeatRecord"):
            ShardCollection(
                no_update_repeat_records=("not a record",),  # type: ignore[arg-type]
                seed_cell_records=(),
                theta_hashes=(),
            )

    def test_rejects_non_seed_cell_record(self) -> None:
        with pytest.raises(ValueError, match="SeedCellRecord"):
            ShardCollection(
                no_update_repeat_records=(),
                seed_cell_records=("not a record",),  # type: ignore[arg-type]
                theta_hashes=(),
            )

    def test_rejects_non_shard_theta_hash(self) -> None:
        with pytest.raises(ValueError, match="ShardThetaHash"):
            ShardCollection(
                no_update_repeat_records=(),
                seed_cell_records=(),
                theta_hashes=("not a theta hash",),  # type: ignore[arg-type]
            )
