"""High-signal contract tests for Research/20 frozen language organ boundary.

These tests verify the behavioral contracts of the frozen language organ
using a test-only tiny differentiable organ — no network or model download.

Contracts verified:

  - CE gradients reach ``soft_bank`` but never organ parameters
  - Teacher-forced logits have shape ``[T, V]`` matching target token count
  - ``generate`` takes no target/correction argument and is deterministic greedy
  - ``encode_texts`` returns detached ``[N, D]`` mean-pooled float32 features
  - Organ hash changes when any parameter byte changes
  - Organ hash is unchanged through cortex write/consolidate cycles
  - ``assert_frozen`` passes for the tiny organ and fails for unfrozen params
  - ``QwenFrozenOrgan`` is marked ``requires_model`` (not run by default)
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from oczy.experiments.meta_cortex.contracts import (
    DEFAULT_FEATURE_DIM,
    DialogueMessage,
    ModelConfig,
)
from oczy.experiments.meta_cortex.model import MetaCortex
from oczy.experiments.meta_cortex.organ import (
    FrozenLanguageOrgan,
    FrozenOrganError,
    QwenFrozenOrgan,
)

_MODEL_TESTS_ENABLED = os.environ.get("OCZY_RUN_MODEL_TESTS", "") in ("1", "true")

# ---------------------------------------------------------------------------
# Test-only tiny frozen organ
# ---------------------------------------------------------------------------


class _TinyFrozenOrgan:
    """Test-only differentiable tiny frozen organ.

    Uses a frozen random embedding + linear head with ``requires_grad=False``.
    Gradients flow through ``soft_bank`` input but never to organ parameters.
    No network or model download required.
    """

    def __init__(self, feature_dim: int = 16, vocab_size: int = 128) -> None:
        self.feature_dim = feature_dim
        self._vocab_size = vocab_size
        self._closed = False

        # Frozen parameters (plain tensors, not nn.Parameter).
        torch.manual_seed(42)
        self._embedding = torch.randn(vocab_size, feature_dim, requires_grad=False)
        self._output_proj = torch.randn(feature_dim, vocab_size, requires_grad=False)
        self._output_bias = torch.zeros(vocab_size, requires_grad=False)

        self._initial_hash = self.parameter_hash()

    # -- Tokenizer --------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def _render(self, messages: list[DialogueMessage]) -> str:
        return "".join(f"{m.role}: {m.content}\n" for m in messages)

    # -- Protocol: encode_texts ------------------------------------------

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
            embeds = self._embedding[ids]  # [S, D]
            features.append(embeds.mean(dim=0).to(dtype=torch.float32))
        return torch.stack(features, dim=0).detach()

    # -- Protocol: teacher_forced_logits ---------------------------------

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

        bank = soft_bank[0]  # [L, D] — carries gradients from cortex
        bank_steer = bank.mean(dim=0)  # [D] — bank steering, carries gradients
        answer_embeds = self._embedding[answer_ids]  # [A, D] frozen

        # Bank genuinely steers answer-position logits via additive influence.
        steered = answer_embeds + bank_steer.unsqueeze(0)  # [A, D]
        logits = steered @ self._output_proj + self._output_bias  # [A, V]
        return logits

    # -- Protocol: teacher_forced_loss -----------------------------------

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
        return F.cross_entropy(logits, targets)

    # -- Protocol: specificity_kl ----------------------------------------

    def specificity_kl(
        self,
        messages: list[DialogueMessage],
        target: str,
        soft_bank: torch.Tensor,
        reference_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bank_logits = self.teacher_forced_logits(messages, target, soft_bank)
        bank_log_probs = F.log_softmax(bank_logits, dim=-1)

        with torch.no_grad():
            if reference_bank is not None:
                ref_logits = self.teacher_forced_logits(messages, target, reference_bank)
            else:
                # Organ-only: no bank steering.
                target_text = " " + target.lstrip()
                answer_ids = self._tokenize(target_text)
                answer_embeds = self._embedding[answer_ids]
                ref_logits = answer_embeds @ self._output_proj + self._output_bias
            ref_probs = F.softmax(ref_logits, dim=-1)

        bank_probs = bank_log_probs.exp()
        ref_log_probs = ref_probs.clamp(min=1e-12).log()
        kl = (bank_probs * (bank_log_probs - ref_log_probs)).sum(dim=-1)
        return kl.mean()

    # -- Protocol: generate ----------------------------------------------

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
        if soft_bank.dim() != 3 or soft_bank.shape[0] != 1:
            raise FrozenOrganError("soft_bank must be [1, L, D]")
        if soft_bank.shape[2] != self.feature_dim:
            raise FrozenOrganError("soft_bank feature dim mismatch")

        prompt_text = self._render(messages)
        prompt_ids = self._tokenize(prompt_text)
        bank = soft_bank[0]  # [L, D]
        bank_steer = bank.mean(dim=0)  # [D] — bank steering
        prompt_embeds = self._embedding[prompt_ids]  # [S, D]

        # First token: last prompt embedding + bank steering.
        first_embed = prompt_embeds[-1:] + bank_steer.unsqueeze(0)
        logits = first_embed @ self._output_proj + self._output_bias
        next_token = int(logits[-1].argmax().item())

        generated: list[str] = []
        for _ in range(max_new_tokens):
            generated.append(chr(min(next_token, 127)))
            next_embed = self._embedding[next_token].unsqueeze(0)
            steered = next_embed + bank_steer.unsqueeze(0)
            logits = steered @ self._output_proj + self._output_bias
            next_token = int(logits.argmax().item())

        return "".join(generated)

    # -- Protocol: parameter_hash ----------------------------------------

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

    # -- Protocol: assert_frozen -----------------------------------------

    def assert_frozen(self) -> None:
        for name, tensor in [
            ("embedding", self._embedding),
            ("output_proj", self._output_proj),
            ("output_bias", self._output_bias),
        ]:
            if tensor.requires_grad:
                raise FrozenOrganError(f"{name} has requires_grad=True")

    # -- Protocol: close -------------------------------------------------

    def close(self) -> None:
        self._closed = True

    # -- Audit helpers ---------------------------------------------------

    def initial_hash(self) -> str:
        return self._initial_hash

    def organ_parameter_ids(self) -> set[int]:
        return {id(self._embedding), id(self._output_proj), id(self._output_bias)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def organ() -> _TinyFrozenOrgan:
    return _TinyFrozenOrgan(feature_dim=16, vocab_size=128)


@pytest.fixture
def model() -> MetaCortex:
    return MetaCortex(ModelConfig(feature_dim=16, d_cortex=64, bank_width=2))


def _make_messages() -> list[DialogueMessage]:
    return [
        DialogueMessage(role="user", content="What is the code?"),
    ]


def _make_bank(L: int = 2, D: int = 16, requires_grad: bool = True) -> torch.Tensor:
    bank = torch.randn(1, L, D, dtype=torch.float32)
    if requires_grad:
        bank.requires_grad_(True)
    return bank


# ---------------------------------------------------------------------------
# Gradient flow: CE gradients reach bank, not organ params
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_ce_gradient_reaches_bank(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        loss = organ.teacher_forced_loss(messages, "secret", bank)
        loss.backward()
        assert bank.grad is not None
        assert torch.isfinite(bank.grad).all()
        assert bank.grad.abs().sum().item() > 0

    def test_organ_params_have_no_grad(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        loss = organ.teacher_forced_loss(messages, "secret", bank)
        loss.backward()
        # Organ params are plain tensors with requires_grad=False.
        assert not organ._embedding.requires_grad
        assert not organ._output_proj.requires_grad
        assert not organ._output_bias.requires_grad

    def test_kl_gradient_reaches_bank(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        kl = organ.specificity_kl(messages, "secret", bank)
        kl.backward()
        assert bank.grad is not None
        assert torch.isfinite(bank.grad).all()

    def test_kl_gradient_not_to_organ_params(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        kl = organ.specificity_kl(messages, "secret", bank)
        kl.backward()
        assert not organ._embedding.requires_grad
        assert not organ._output_proj.requires_grad
        assert not organ._output_bias.requires_grad

    def test_gradient_through_cortex_to_model_params(
        self, organ: _TinyFrozenOrgan, model: MetaCortex
    ) -> None:
        """End-to-end: loss → soft_bank → couple → read → write → model params."""
        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        batch_vals = torch.randn(1, 4, 16, dtype=torch.float32)
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        state = model.write(state, EventFeatureBatch(values=batch_vals)).state
        state = model.consolidate(state).state

        query = torch.randn(1, 16, dtype=torch.float32)
        readout = model.read(state, query)
        soft_bank = model.couple(readout)

        messages = _make_messages()
        loss = organ.teacher_forced_loss(messages, "secret", soft_bank)
        loss.backward()

        # Verify gradients reached cortex model parameters.
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                if param.grad.abs().sum().item() > 0:
                    has_grad = True
        assert has_grad, "No cortex parameter received a gradient"


# ---------------------------------------------------------------------------
# Teacher-forced logits shape and token-position slicing
# ---------------------------------------------------------------------------


class TestTeacherForcedLogits:
    def test_single_token_target(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        logits = organ.teacher_forced_logits(messages, "A", bank)
        assert logits.shape[1] == 128  # vocab_size
        assert logits.shape[0] >= 1  # at least 1 target token

    def test_multi_token_target(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        logits = organ.teacher_forced_logits(messages, "hello world", bank)
        # " hello world" → multiple tokens
        assert logits.shape[0] > 1
        assert logits.shape[1] == 128

    def test_logits_carry_gradient(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        logits = organ.teacher_forced_logits(messages, "secret", bank)
        assert logits.requires_grad

    def test_distinct_banks_produce_distinct_logits(
        self, organ: _TinyFrozenOrgan
    ) -> None:
        """Distinct controlled banks deterministically produce distinct logits."""
        bank_a = torch.zeros(1, 2, 16)
        bank_b = torch.full((1, 2, 16), 5.0)
        messages = _make_messages()
        logits_a = organ.teacher_forced_logits(messages, "secret", bank_a)
        logits_b = organ.teacher_forced_logits(messages, "secret", bank_b)
        assert not torch.allclose(logits_a, logits_b)

    def test_empty_target_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        messages = _make_messages()
        with pytest.raises(FrozenOrganError, match="empty target"):
            organ.teacher_forced_logits(messages, "", bank)

    def test_empty_messages_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank()
        with pytest.raises(FrozenOrganError, match="empty messages"):
            organ.teacher_forced_logits([], "target", bank)

    def test_wrong_bank_dim_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = torch.randn(2, 2, 16)  # batch=2, not 1
        messages = _make_messages()
        with pytest.raises(FrozenOrganError, match="soft_bank"):
            organ.teacher_forced_logits(messages, "target", bank)

    def test_wrong_feature_dim_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = torch.randn(1, 2, 32)  # wrong D
        messages = _make_messages()
        with pytest.raises(FrozenOrganError, match="feature dim"):
            organ.teacher_forced_logits(messages, "target", bank)

    def test_nonfinite_bank_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = torch.randn(1, 2, 16)
        bank[0, 0, 0] = float("inf")
        messages = _make_messages()
        with pytest.raises(FrozenOrganError, match="non-finite"):
            organ.teacher_forced_logits(messages, "target", bank)


# ---------------------------------------------------------------------------
# Generate: no target/correction arg, deterministic greedy
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_has_no_target_arg(self) -> None:
        sig = inspect.signature(_TinyFrozenOrgan.generate)
        params = set(sig.parameters.keys())
        assert "target" not in params
        assert "correction" not in params
        assert "task_id" not in params
        assert "messages" in params
        assert "soft_bank" in params
        assert "max_new_tokens" in params

    def test_generate_deterministic(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank(requires_grad=False)
        messages = _make_messages()
        out1 = organ.generate(messages, bank, max_new_tokens=8)
        out2 = organ.generate(messages, bank, max_new_tokens=8)
        assert out1 == out2

    def test_generate_returns_str(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank(requires_grad=False)
        messages = _make_messages()
        result = organ.generate(messages, bank, max_new_tokens=4)
        assert isinstance(result, str)

    def test_generate_max_new_tokens_zero_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank(requires_grad=False)
        messages = _make_messages()
        with pytest.raises(FrozenOrganError, match="max_new_tokens"):
            organ.generate(messages, bank, max_new_tokens=0)

    def test_generate_empty_messages_rejected(self, organ: _TinyFrozenOrgan) -> None:
        bank = _make_bank(requires_grad=False)
        with pytest.raises(FrozenOrganError, match="empty messages"):
            organ.generate([], bank, max_new_tokens=4)

    def test_generate_different_bank_different_output(
        self, organ: _TinyFrozenOrgan
    ) -> None:
        """Distinct controlled banks deterministically produce different output."""
        bank_a = torch.zeros(1, 2, 16)
        bank_b = torch.full((1, 2, 16), 100.0)
        messages = _make_messages()
        out_a = organ.generate(messages, bank_a, max_new_tokens=8)
        out_b = organ.generate(messages, bank_b, max_new_tokens=8)
        assert out_a != out_b


# ---------------------------------------------------------------------------
# encode_texts: mean-pooled, detached, correct width
# ---------------------------------------------------------------------------


class TestEncodeTexts:
    def test_shape_n_by_d(self, organ: _TinyFrozenOrgan) -> None:
        texts = ["hello", "world", "foo"]
        features = organ.encode_texts(texts)
        assert features.shape == (3, 16)

    def test_detached(self, organ: _TinyFrozenOrgan) -> None:
        features = organ.encode_texts(["hello"])
        assert not features.requires_grad

    def test_float32(self, organ: _TinyFrozenOrgan) -> None:
        features = organ.encode_texts(["hello"])
        assert features.dtype == torch.float32

    def test_empty_text_rejected(self, organ: _TinyFrozenOrgan) -> None:
        with pytest.raises(FrozenOrganError, match="empty"):
            organ.encode_texts([])

    def test_non_string_rejected(self, organ: _TinyFrozenOrgan) -> None:
        with pytest.raises((FrozenOrganError, TypeError)):
            organ.encode_texts([123])  # type: ignore[list-item]

    def test_mean_pooled(self, organ: _TinyFrozenOrgan) -> None:
        """Verify output is actually mean-pooled embeddings."""
        text = "abc"
        ids = organ._tokenize(text)
        expected = organ._embedding[ids].mean(dim=0).to(dtype=torch.float32)
        result = organ.encode_texts([text])
        assert torch.allclose(result[0], expected)

    def test_deterministic(self, organ: _TinyFrozenOrgan) -> None:
        texts = ["hello", "world"]
        f1 = organ.encode_texts(texts)
        f2 = organ.encode_texts(texts)
        assert torch.equal(f1, f2)


# ---------------------------------------------------------------------------
# Organ hash: changes on byte change, unchanged through cortex update
# ---------------------------------------------------------------------------


class TestOrganHash:
    def test_hash_is_64_char_hex(self, organ: _TinyFrozenOrgan) -> None:
        h = organ.parameter_hash()
        assert len(h) == 64
        int(h, 16)

    def test_hash_changes_on_byte_change(self) -> None:
        organ_a = _TinyFrozenOrgan(feature_dim=16, vocab_size=128)
        organ_b = _TinyFrozenOrgan(feature_dim=16, vocab_size=128)
        # Modify a parameter byte.
        organ_b._embedding[0, 0] += 1.0
        assert organ_a.parameter_hash() != organ_b.parameter_hash()

    def test_hash_changes_on_different_vocab_size(self) -> None:
        organ_a = _TinyFrozenOrgan(feature_dim=16, vocab_size=128)
        organ_b = _TinyFrozenOrgan(feature_dim=16, vocab_size=64)
        assert organ_a.parameter_hash() != organ_b.parameter_hash()

    def test_hash_unchanged_through_cortex_update(
        self, organ: _TinyFrozenOrgan, model: MetaCortex
    ) -> None:
        hash_before = organ.parameter_hash()
        # Run cortex operations.
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        batch = EventFeatureBatch(values=torch.randn(1, 4, 16, dtype=torch.float32))
        state = model.write(state, batch).state
        state = model.consolidate(state).state
        # Run a forward pass with soft_bank.
        query = torch.randn(1, 16, dtype=torch.float32)
        readout = model.read(state, query)
        soft_bank = model.couple(readout)
        _ = organ.teacher_forced_loss(_make_messages(), "target", soft_bank)
        hash_after = organ.parameter_hash()
        assert hash_before == hash_after

    def test_initial_hash_matches_current(self, organ: _TinyFrozenOrgan) -> None:
        assert organ.initial_hash() == organ.parameter_hash()


# ---------------------------------------------------------------------------
# assert_frozen
# ---------------------------------------------------------------------------


class TestAssertFrozen:
    def test_tiny_organ_passes(self, organ: _TinyFrozenOrgan) -> None:
        organ.assert_frozen()  # should not raise

    def test_unfrozen_param_fails(self) -> None:
        organ = _TinyFrozenOrgan()
        organ._embedding.requires_grad_(True)
        with pytest.raises(FrozenOrganError, match="requires_grad"):
            organ.assert_frozen()

    def test_assert_frozen_after_use(
        self, organ: _TinyFrozenOrgan, model: MetaCortex
    ) -> None:
        """assert_frozen still passes after training-style operations."""
        from oczy.experiments.meta_cortex.model import EventFeatureBatch

        state = model.initial_state(1, device="cpu", dtype=torch.float32)
        batch = EventFeatureBatch(values=torch.randn(1, 4, 16, dtype=torch.float32))
        state = model.write(state, batch).state
        state = model.consolidate(state).state
        query = torch.randn(1, 16, dtype=torch.float32)
        readout = model.read(state, query)
        soft_bank = model.couple(readout)
        loss = organ.teacher_forced_loss(_make_messages(), "target", soft_bank)
        loss.backward()
        organ.assert_frozen()  # should still pass


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_blocks_encode(self, organ: _TinyFrozenOrgan) -> None:
        organ.close()
        with pytest.raises(FrozenOrganError, match="closed"):
            organ.encode_texts(["test"])

    def test_close_blocks_teacher_forced(self, organ: _TinyFrozenOrgan) -> None:
        organ.close()
        bank = _make_bank()
        with pytest.raises(FrozenOrganError, match="closed"):
            organ.teacher_forced_loss(_make_messages(), "target", bank)

    def test_close_blocks_generate(self, organ: _TinyFrozenOrgan) -> None:
        organ.close()
        bank = _make_bank(requires_grad=False)
        with pytest.raises(FrozenOrganError, match="closed"):
            organ.generate(_make_messages(), bank, max_new_tokens=4)

    def test_close_idempotent(self, organ: _TinyFrozenOrgan) -> None:
        organ.close()
        organ.close()  # should not raise


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_tiny_organ_satisfies_protocol(self) -> None:
        organ = _TinyFrozenOrgan()
        # The Protocol is runtime_checkable.
        assert isinstance(organ, FrozenLanguageOrgan)

    def test_protocol_has_required_methods(self) -> None:
        required = {
            "encode_texts",
            "teacher_forced_logits",
            "teacher_forced_loss",
            "specificity_kl",
            "generate",
            "parameter_hash",
            "assert_frozen",
            "close",
        }
        for method in required:
            assert hasattr(FrozenLanguageOrgan, method), f"Protocol missing {method}"


# ---------------------------------------------------------------------------
# QwenFrozenOrgan (requires real model — not run by default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL_TESTS_ENABLED,
    reason="Requires real Qwen model download; set OCZY_RUN_MODEL_TESTS=1 to enable",
)
@pytest.mark.requires_model
class TestQwenFrozenOrgan:
    """Tests that require the real Qwen model to be downloaded.

    These are marked ``requires_model`` and skipped by default.
    """

    def test_load_and_hash(self) -> None:
        organ = QwenFrozenOrgan.load(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            feature_dim=DEFAULT_FEATURE_DIM,
        )
        try:
            h = organ.parameter_hash()
            assert len(h) == 64
            organ.assert_frozen()
        finally:
            organ.close()

    def test_encode_texts_shape(self) -> None:
        organ = QwenFrozenOrgan.load(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            feature_dim=DEFAULT_FEATURE_DIM,
        )
        try:
            features = organ.encode_texts(["hello", "world"])
            assert features.shape == (2, DEFAULT_FEATURE_DIM)
            assert features.dtype == torch.float32
            assert not features.requires_grad
        finally:
            organ.close()



# ---------------------------------------------------------------------------
# Offline-aware load-target resolution (no real model / no network)
# ---------------------------------------------------------------------------
#
# These tests defend the fail-closed remote behavior and local-path precedence
# of ``QwenFrozenOrgan.load`` by monkeypatching ``HFDriver.load`` so that no
# real model is ever downloaded.  The contract:
#
#   - ``OCZY_MODEL_DIR`` (non-empty dir) takes precedence over
#     ``OCZY_HF_MODEL_DIR`` (non-empty dir), which takes precedence over the
#     hub ``model_id`` — but ONLY outside offline/remote mode.
#   - Under ``OCZY_REMOTE_CPU_ONLY=1`` / ``HF_HUB_OFFLINE=1`` /
#     ``TRANSFORMERS_OFFLINE=1``, the hub-id fallback is forbidden: if no
#     local dir resolves, ``FrozenOrganError`` is raised BEFORE
#     ``HFDriver.load`` is ever called.
#   - The organ's recorded identity (``organ.model_id``) is always the
#     ORIGINAL requested ``model_id``; only ``HFDriver.load`` receives the
#     resolved local path.
#
# This mirrors the already-fixed R19 convention (s19_language_organ.py) and
# fixes the Kaggle DEV smoke failure where a hub id was passed to
# ``from_pretrained`` with ``local_files_only=True``.
#
#
# --- Fake HFDriver for monkeypatching ---------------------------------
#

class _FakeHFDriver:
    """Minimal HFDriver stand-in recording the model_id passed to ``load``.

    No real model is instantiated.  The returned instance carries just
    enough attributes for ``QwenFrozenOrgan.__init__`` to succeed so we can
    assert on ``organ.model_id`` afterwards.
    """

    last_loaded_id: str | None = None

    @classmethod
    def load(cls, *, model_id: str = "", **_kwargs: object) -> _FakeHFDriver:
        cls.last_loaded_id = model_id
        return cls()

    def __init__(self) -> None:
        self.model_id = _FakeHFDriver.last_loaded_id or ""
        self.n_embd = DEFAULT_FEATURE_DIM
        self.n_vocab = 128
        self._model = _FakeModel()
        self._tokenizer = _FakeTokenizer()


class _FakeModel:
    """Bare-minimum model stub for organ construction and hashing.

    ``named_parameters()`` yields nothing (no params to freeze/hash),
    ``eval()`` flips ``training=False``, and ``config`` is absent so
    ``parameter_hash`` uses an empty config dict.
    """

    training: bool = True

    def eval(self) -> None:
        self.training = False

    def parameters(self):
        return iter(())

    def named_parameters(self):
        return iter(())


class _FakeTokenizer:
    """Bare-minimum tokenizer stub for organ construction."""

    def encode(self, _text: str, add_special_tokens: bool = True) -> list[int]:
        return [1, 2, 3]


class TestOfflineLoadTargetResolution:
    """Tests for ``QwenFrozenOrgan.load`` offline resolution, monkeypatching
    ``HFDriver.load`` so no real model/network load occurs.
    """

    _HUB_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strip every offline/local env var before each test so they start
        from a clean baseline; individual tests set what they need."""
        for var in (
            "OCZY_MODEL_DIR",
            "OCZY_HF_MODEL_DIR",
            "OCZY_REMOTE_CPU_ONLY",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture(autouse=True)
    def _patch_hf_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatch ``HFDriver.load`` in the organ module so no real
        model is ever loaded."""
        import oczy.experiments.meta_cortex.organ as organ_mod

        _FakeHFDriver.last_loaded_id = None
        monkeypatch.setattr(organ_mod.HFDriver, "load", _FakeHFDriver.load)

    # -- environment precedence ------------------------------------------

    def test_model_dir_takes_precedence_over_hf_model_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """OCZY_MODEL_DIR wins over OCZY_HF_MODEL_DIR when both are set and
        non-empty.  HFDriver.load receives the OCZY_MODEL_DIR path."""

        model_dir = tmp_path / "model_dir"
        hf_model_dir = tmp_path / "hf_model_dir"
        model_dir.mkdir()
        (model_dir / "marker").write_text("x")
        hf_model_dir.mkdir()
        (hf_model_dir / "marker").write_text("x")
        monkeypatch.setenv("OCZY_MODEL_DIR", str(model_dir))
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", str(hf_model_dir))

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == str(model_dir)
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    def test_hf_model_dir_used_when_model_dir_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When OCZY_MODEL_DIR is unset, OCZY_HF_MODEL_DIR is used if it is
        a non-empty directory."""
        hf_model_dir = tmp_path / "hf_model"
        hf_model_dir.mkdir()
        (hf_model_dir / "marker").write_text("x")
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", str(hf_model_dir))

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == str(hf_model_dir)
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    # -- invalid / missing local path -----------------------------------

    def test_empty_local_dir_rejected_falls_through_to_hub(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """An empty OCZY_MODEL_DIR directory is rejected (non-empty check);
        in non-offline mode the resolver falls through to the hub id."""

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setenv("OCZY_MODEL_DIR", str(empty_dir))

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == self._HUB_ID
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    def test_nonexistent_local_dir_rejected_falls_through_to_hub(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A nonexistent OCZY_MODEL_DIR is rejected; in non-offline mode the
        resolver falls through to the hub id."""
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == self._HUB_ID
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    # -- explicit offline refusal (fail-closed) -------------------------

    def test_offline_no_local_dir_raises_before_hfdriver(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under HF_HUB_OFFLINE=1 with no valid local dir, load must raise
        FrozenOrganError BEFORE HFDriver.load is ever called."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")
        monkeypatch.setenv("OCZY_HF_MODEL_DIR", "/nonexistent/path/xyz")

        with pytest.raises(FrozenOrganError, match="offline_model_unavailable"):
            QwenFrozenOrgan.load(model_id=self._HUB_ID)
        assert _FakeHFDriver.last_loaded_id is None, (
            "HFDriver.load must not be called when offline and no local dir"
        )

    def test_remote_cpu_only_no_env_vars_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under OCZY_REMOTE_CPU_ONLY=1 with no env vars set at all, load
        must raise FrozenOrganError before HFDriver.load."""
        monkeypatch.setenv("OCZY_REMOTE_CPU_ONLY", "1")

        with pytest.raises(FrozenOrganError, match="offline_model_unavailable"):
            QwenFrozenOrgan.load(model_id=self._HUB_ID)
        assert _FakeHFDriver.last_loaded_id is None

    def test_transformers_offline_empty_dir_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Under TRANSFORMERS_OFFLINE=1 with only an empty local dir, load
        must raise FrozenOrganError (empty dirs are not valid mounts)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        monkeypatch.setenv("OCZY_MODEL_DIR", str(empty_dir))

        with pytest.raises(FrozenOrganError, match="offline_model_unavailable"):
            QwenFrozenOrgan.load(model_id=self._HUB_ID)
        assert _FakeHFDriver.last_loaded_id is None

    # -- online model-id fallback ----------------------------------------

    def test_online_no_env_vars_falls_back_to_hub_id(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In non-offline mode with no local env dirs set, the resolver
        falls back to the original hub model_id, and the organ identity
        matches."""
        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == self._HUB_ID
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    def test_online_invalid_local_dir_falls_back_to_hub_id(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In non-offline mode with an invalid OCZY_MODEL_DIR, the resolver
        falls back to the hub id (network resolution still allowed)."""
        monkeypatch.setenv("OCZY_MODEL_DIR", "/nonexistent/path/abc")

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == self._HUB_ID
            assert organ.model_id == self._HUB_ID
        finally:
            organ.close()

    # -- organ identity preservation under local resolution -------------

    def test_organ_identity_is_requested_id_not_local_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When a local dir resolves, the organ's recorded identity
        (``model_id``) must be the ORIGINAL requested model_id, not the
        local path — while HFDriver.load received the local path."""

        local_dir = tmp_path / "models" / "Qwen2.5-0.5B-Instruct"
        local_dir.mkdir(parents=True)
        (local_dir / "marker").write_text("x")
        monkeypatch.setenv("OCZY_MODEL_DIR", str(local_dir))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        organ = QwenFrozenOrgan.load(model_id=self._HUB_ID)
        try:
            assert _FakeHFDriver.last_loaded_id == str(local_dir)
            assert organ.model_id == self._HUB_ID
            assert organ.model_id != str(local_dir)
        finally:
            organ.close()
