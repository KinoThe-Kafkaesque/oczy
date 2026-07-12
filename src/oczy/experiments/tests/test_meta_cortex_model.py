"""High-signal contract tests for Research/20 differentiable cortex model.

These tests verify the behavioral contracts of ``MetaCortex``:

  - Initial F/S exactly zero with ``[B,64,64]``; state methods do not mutate inputs
  - ``write`` exactly matches ``lambda*F + eta*outer(v,k)`` and leaves S unchanged
  - ``consolidate`` exactly matches ``(1-g)S + gF`` and returns bitwise-zero F
  - Read/couple shapes are ``[B,64]`` and ``[B,L,D]`` for one versus five events
  - Parameter breakdown sums to ``129D + 64L + 91_588``, including 207,364 and 93,780
  - A loss on bank output backpropagates finite gradients to writer, consolidator,
    reader, feature projection, slot embeddings, output projection, and gain
  - Repeated updates remain finite; invalid shape/dtype/device/nonfinite inputs raise

No network or model download is required.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F

from oczy.experiments.meta_cortex.contracts import CORTEX_DIM, ContractError, ModelConfig
from oczy.experiments.meta_cortex.model import (
    ConsolidationResult,
    EventFeatureBatch,
    MetaCortex,
    WriteResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEV = torch.device("cpu")
_DTYPE = torch.float32


def _make_model(D: int = 16, L: int = 2) -> MetaCortex:
    return MetaCortex(ModelConfig(feature_dim=D, d_cortex=64, bank_width=L))


def _make_event_batch(B: int, D: int) -> EventFeatureBatch:
    torch.manual_seed(0)
    return EventFeatureBatch(
        values=torch.randn(B, 4, D, dtype=_DTYPE, device=_DEV)
    )


def _make_query(B: int, D: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, D, dtype=_DTYPE, device=_DEV)


def _theta_hash(model: MetaCortex) -> str:
    parts = []
    for name, param in sorted(model.state_dict().items()):
        data = param.detach().cpu().contiguous().numpy().tobytes()
        parts.append(f"{name}:{hashlib.sha256(data).hexdigest()}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_initial_state_zero(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        assert torch.count_nonzero(state.fast).item() == 0
        assert torch.count_nonzero(state.slow).item() == 0

    def test_initial_state_shape(self) -> None:
        model = _make_model()
        for B in (1, 3, 8):
            state = model.initial_state(B, device=_DEV, dtype=_DTYPE)
            assert state.fast.shape == (B, CORTEX_DIM, CORTEX_DIM)
            assert state.slow.shape == (B, CORTEX_DIM, CORTEX_DIM)

    def test_initial_state_dtype(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        assert state.fast.dtype == _DTYPE
        assert state.slow.dtype == _DTYPE

    def test_initial_state_rejects_non_float32(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="float32"):
            model.initial_state(1, device=_DEV, dtype=torch.float64)

    def test_initial_state_rejects_batch_zero(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError):
            model.initial_state(0, device=_DEV, dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Write equation: F_next = lambda*F + eta*outer(v,k), S unchanged
# ---------------------------------------------------------------------------


class TestWriteEquation:
    def test_write_returns_new_state(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        result = model.write(state, batch)
        assert result.state is not state
        assert isinstance(result, WriteResult)

    def test_write_does_not_mutate_input_state(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        original_fast = state.fast.clone()
        original_slow = state.slow.clone()
        batch = _make_event_batch(1, 16)
        model.write(state, batch)
        assert torch.equal(state.fast, original_fast)
        assert torch.equal(state.slow, original_slow)

    def test_write_s_unchanged(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        result = model.write(state, batch)
        assert torch.equal(result.state.slow, state.slow)

    def test_write_f_matches_equation_from_zero(self) -> None:
        """From F=0, F_next = eta * outer(v, k) exactly."""
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)

        # We need to replicate the equation manually.
        C = 64
        values = batch.values
        B = 1
        proj = model.feature_projection(values)  # [1,4,64]
        roles = model.role_embeddings[:4]
        Z = torch.tanh(proj + roles.unsqueeze(0))
        Z_flat = Z.reshape(B, 4 * C)
        e = torch.tanh(model.event_fusion(Z_flat))
        e_norm = e / (e.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        f_ctx = torch.bmm(state.fast, e_norm.unsqueeze(-1)).squeeze(-1)
        h_in = torch.cat([e, f_ctx], dim=-1)
        h = F.silu(model.writer_hidden(h_in))
        k = torch.tanh(model.writer_key(h))
        k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        v = torch.tanh(model.writer_value(h))
        eta = torch.sigmoid(model.writer_eta(h)).view(1, 1, 1)
        lam = torch.sigmoid(model.writer_decay(h)).view(1, 1, 1)
        outer = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        expected_F = lam * state.fast + eta * outer

        result = model.write(state, batch)
        assert torch.allclose(result.state.fast, expected_F, atol=1e-5)

    def test_write_f_matches_equation_with_nonzero_F(self) -> None:
        """With nonzero F, F_next = lambda*F + eta*outer(v,k)."""
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        # First write to get nonzero F.
        batch1 = _make_event_batch(1, 16)
        result1 = model.write(state, batch1)
        state1 = result1.state

        # Second write from nonzero F.
        batch2 = _make_event_batch(1, 16)
        result2 = model.write(state1, batch2)

        # Manual computation.
        C = 64
        values = batch2.values
        B = 1
        proj = model.feature_projection(values)
        roles = model.role_embeddings[:4]
        Z = torch.tanh(proj + roles.unsqueeze(0))
        Z_flat = Z.reshape(B, 4 * C)
        e = torch.tanh(model.event_fusion(Z_flat))
        e_norm = e / (e.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        f_ctx = torch.bmm(state1.fast, e_norm.unsqueeze(-1)).squeeze(-1)
        h_in = torch.cat([e, f_ctx], dim=-1)
        h = F.silu(model.writer_hidden(h_in))
        k = torch.tanh(model.writer_key(h))
        k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        v = torch.tanh(model.writer_value(h))
        eta = torch.sigmoid(model.writer_eta(h)).view(1, 1, 1)
        lam = torch.sigmoid(model.writer_decay(h)).view(1, 1, 1)
        outer = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        expected_F = lam * state1.fast + eta * outer

        assert torch.allclose(result2.state.fast, expected_F, atol=1e-5)

    def test_write_key_value_shapes(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        result = model.write(state, batch)
        assert result.key.shape == (1, 64)
        assert result.value.shape == (1, 64)
        assert result.eta.shape == (1, 1)
        assert result.decay.shape == (1, 1)

    def test_write_eta_lambda_in_zero_one(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        result = model.write(state, batch)
        assert result.eta.min().item() >= 0.0
        assert result.eta.max().item() <= 1.0
        assert result.decay.min().item() >= 0.0
        assert result.decay.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Consolidation: S_next = (1-g)S + gF, F_next = 0
# ---------------------------------------------------------------------------


class TestConsolidation:
    def test_consolidate_returns_zero_F(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        write_result = model.write(state, batch)
        cons_result = model.consolidate(write_result.state)
        assert torch.count_nonzero(cons_result.state.fast).item() == 0

    def test_consolidate_s_matches_equation(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        write_result = model.write(state, batch)
        state_after_write = write_result.state

        # Manual consolidation.
        p = model.consolidation_probe
        p = p / (p.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        f_probe = torch.matmul(state_after_write.fast, p)
        s_probe = torch.matmul(state_after_write.slow, p)
        h_g_in = torch.cat([f_probe, s_probe], dim=-1)
        h_g = F.silu(model.consolidation_hidden(h_g_in))
        g = torch.sigmoid(model.consolidation_gate(h_g)).view(1, 1, 1)
        expected_S = (1.0 - g) * state_after_write.slow + g * state_after_write.fast

        cons_result = model.consolidate(state_after_write)
        assert torch.allclose(cons_result.state.slow, expected_S, atol=1e-5)

    def test_consolidate_gate_shape(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        write_result = model.write(state, batch)
        cons_result = model.consolidate(write_result.state)
        assert isinstance(cons_result, ConsolidationResult)
        assert cons_result.gate.shape == (1, 1)

    def test_consolidate_gate_in_zero_one(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        write_result = model.write(state, batch)
        cons_result = model.consolidate(write_result.state)
        assert cons_result.gate.min().item() >= 0.0
        assert cons_result.gate.max().item() <= 1.0

    def test_consolidate_does_not_mutate_input(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        write_result = model.write(state, batch)
        original_fast = write_result.state.fast.clone()
        original_slow = write_result.state.slow.clone()
        model.consolidate(write_result.state)
        assert torch.equal(write_result.state.fast, original_fast)
        assert torch.equal(write_result.state.slow, original_slow)

    def test_consolidate_from_zero_state(self) -> None:
        """Consolidating a zero state should keep S=0 and F=0."""
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        cons_result = model.consolidate(state)
        assert torch.count_nonzero(cons_result.state.fast).item() == 0
        assert torch.count_nonzero(cons_result.state.slow).item() == 0


# ---------------------------------------------------------------------------
# Read and couple shapes
# ---------------------------------------------------------------------------


class TestReadCouple:
    def test_read_shape(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        query = _make_query(1, 16)
        readout = model.read(state, query)
        assert readout.shape == (1, 64)

    def test_couple_shape(self) -> None:
        D, L = 16, 2
        model = _make_model(D=D, L=L)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        query = _make_query(1, D)
        readout = model.read(state, query)
        bank = model.couple(readout)
        assert bank.shape == (1, L, D)

    def test_couple_shape_independent_of_events(self) -> None:
        """L never changes regardless of how many writes happened."""
        D, L = 16, 3
        model = _make_model(D=D, L=L)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)

        # 1 event.
        batch = _make_event_batch(1, D)
        state = model.write(state, batch).state
        query = _make_query(1, D)
        bank1 = model.couple(model.read(state, query))
        assert bank1.shape == (1, L, D)

        # 4 more events (total 5).
        for _ in range(4):
            batch = _make_event_batch(1, D)
            state = model.write(state, batch).state
        bank5 = model.couple(model.read(state, query))
        assert bank5.shape == (1, L, D)

    def test_read_batch_3(self) -> None:
        D = 16
        model = _make_model(D=D)
        state = model.initial_state(3, device=_DEV, dtype=_DTYPE)
        query = _make_query(3, D)
        readout = model.read(state, query)
        assert readout.shape == (3, 64)

    def test_couple_batch_3(self) -> None:
        D, L = 16, 2
        model = _make_model(D=D, L=L)
        readout = torch.randn(3, 64, dtype=_DTYPE)
        bank = model.couple(readout)
        assert bank.shape == (3, L, D)


# ---------------------------------------------------------------------------
# Zero/swap state
# ---------------------------------------------------------------------------


class TestStateMethods:
    def test_zero_state_returns_zeros(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state = model.write(state, batch).state
        zeroed = model.zero_state(state)
        assert torch.count_nonzero(zeroed.fast).item() == 0
        assert torch.count_nonzero(zeroed.slow).item() == 0

    def test_zero_state_preserves_shape(self) -> None:
        model = _make_model()
        state = model.initial_state(3, device=_DEV, dtype=_DTYPE)
        zeroed = model.zero_state(state)
        assert zeroed.fast.shape == state.fast.shape
        assert zeroed.slow.shape == state.slow.shape

    def test_zero_state_does_not_mutate_input(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state = model.write(state, batch).state
        original_fast = state.fast.clone()
        model.zero_state(state)
        assert torch.equal(state.fast, original_fast)

    def test_swap_state_copies_donor(self) -> None:
        model = _make_model()
        state_a = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        state_b = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state_a = model.write(state_a, batch).state

        swapped = model.swap_state(state_a, state_b)
        assert torch.equal(swapped.fast, state_b.fast)
        assert torch.equal(swapped.slow, state_b.slow)

    def test_swap_state_does_not_mutate_inputs(self) -> None:
        model = _make_model()
        state_a = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        state_b = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state_a = model.write(state_a, batch).state
        original_a_fast = state_a.fast.clone()
        original_b_fast = state_b.fast.clone()
        model.swap_state(state_a, state_b)
        assert torch.equal(state_a.fast, original_a_fast)
        assert torch.equal(state_b.fast, original_b_fast)

    def test_swap_state_rejects_shape_mismatch(self) -> None:
        model = _make_model()
        state_a = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        state_b = model.initial_state(3, device=_DEV, dtype=_DTYPE)
        with pytest.raises(ValueError, match="shape"):
            model.swap_state(state_a, state_b)

    def test_zero_swap_do_not_change_theta(self) -> None:
        model = _make_model()
        hash_before = _theta_hash(model)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state = model.write(state, batch).state
        model.zero_state(state)
        model.swap_state(state, model.initial_state(1, device=_DEV, dtype=_DTYPE))
        hash_after = _theta_hash(model)
        assert hash_before == hash_after


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


class TestParameterAccounting:
    def test_parameter_count_d16_l2(self) -> None:
        model = _make_model(D=16, L=2)
        # 129*16 + 64*2 + 91588 = 2064 + 128 + 91588 = 93780
        assert model.parameter_count() == 93780

    def test_parameter_count_d896_l3(self) -> None:
        model = _make_model(D=896, L=3)
        # 129*896 + 64*3 + 91588 = 115584 + 192 + 91588 = 207364
        assert model.parameter_count() == 207364

    def test_parameter_breakdown_sums_to_total(self) -> None:
        model = _make_model(D=16, L=2)
        breakdown = model.parameter_breakdown()
        assert sum(breakdown.values()) == model.parameter_count()

    def test_parameter_breakdown_has_all_groups(self) -> None:
        model = _make_model(D=16, L=2)
        breakdown = model.parameter_breakdown()
        expected_groups = {
            "feature_projection",
            "role_embeddings",
            "event_fusion",
            "writer_hidden",
            "writer_key",
            "writer_value",
            "writer_eta",
            "writer_decay",
            "consolidation_probe",
            "consolidation_hidden",
            "consolidation_gate",
            "reader_hidden",
            "reader_out",
            "slot_embeddings",
            "layer_norm",
            "coupler_output",
            "coupler_log_gain",
        }
        assert set(breakdown.keys()) == expected_groups

    def test_parameter_count_matches_torch(self) -> None:
        model = _make_model(D=16, L=2)
        torch_count = sum(p.numel() for p in model.parameters())
        assert model.parameter_count() == torch_count

    def test_additional_bank_token_adds_64_params(self) -> None:
        m_l2 = _make_model(D=16, L=2)
        m_l3 = _make_model(D=16, L=3)
        assert m_l3.parameter_count() - m_l2.parameter_count() == 64


# ---------------------------------------------------------------------------
# Logical persistent bytes
# ---------------------------------------------------------------------------


class TestLogicalPersistentBytes:
    def test_before_consolidation_32768(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state = model.write(state, batch).state
        # F is nonzero, so both F+S = 32768.
        assert model.logical_persistent_bytes(state) == 32768

    def test_after_consolidation_16384(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, 16)
        state = model.write(state, batch).state
        state = model.consolidate(state).state
        assert model.logical_persistent_bytes(state) == 16384

    def test_zero_state_16384(self) -> None:
        model = _make_model()
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        # F is zero, so only S counts = 16384.
        assert model.logical_persistent_bytes(state) == 16384


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_loss_backprops_to_all_groups(self) -> None:
        """A loss on bank output backpropagates to every developmental group."""
        D, L = 16, 2
        model = _make_model(D=D, L=L)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)

        # Multi-event unroll.
        for _ in range(3):
            batch = _make_event_batch(1, D)
            state = model.write(state, batch).state
        state = model.consolidate(state).state

        query = _make_query(1, D)
        readout = model.read(state, query)
        bank = model.couple(readout)
        loss = bank.pow(2).mean()
        loss.backward()

        grad_groups = {
            "feature_projection": model.feature_projection.weight,
            "role_embeddings": model.role_embeddings,
            "event_fusion": model.event_fusion.weight,
            "writer_hidden": model.writer_hidden.weight,
            "writer_key": model.writer_key.weight,
            "writer_value": model.writer_value.weight,
            "writer_eta": model.writer_eta.weight,
            "writer_decay": model.writer_decay.weight,
            "consolidation_probe": model.consolidation_probe,
            "consolidation_hidden": model.consolidation_hidden.weight,
            "consolidation_gate": model.consolidation_gate.weight,
            "reader_hidden": model.reader_hidden.weight,
            "reader_out": model.reader_out.weight,
            "slot_embeddings": model.slot_embeddings,
            "layer_norm": model.layer_norm.weight,
            "coupler_output": model.coupler_output.weight,
            "coupler_log_gain": model.coupler_log_gain,
        }
        for name, param in grad_groups.items():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
            assert param.grad.abs().sum().item() > 0, f"{name} has zero gradient"

    def test_gradients_finite_through_consolidation(self) -> None:
        D = 16
        model = _make_model(D=D)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = _make_event_batch(1, D)
        state = model.write(state, batch).state
        state = model.consolidate(state).state
        query = _make_query(1, D)
        readout = model.read(state, query)
        loss = readout.pow(2).mean()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()


# ---------------------------------------------------------------------------
# Repeated writes stay finite
# ---------------------------------------------------------------------------


class TestRepeatedWrites:
    def test_100_writes_finite(self) -> None:
        D = 16
        model = _make_model(D=D)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        for _ in range(100):
            batch = _make_event_batch(1, D)
            state = model.write(state, batch).state
            assert torch.isfinite(state.fast).all()
            assert torch.isfinite(state.slow).all()

    def test_100_writes_state_norm_bounded(self) -> None:
        D = 16
        model = _make_model(D=D)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        for _ in range(100):
            batch = _make_event_batch(1, D)
            state = model.write(state, batch).state
        # Sigmoid-bounded eta/lambda + tanh-bounded v + normalized k
        # means F should not explode.
        assert state.fast.abs().max().item() < 100.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_write_rejects_wrong_shape(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        bad_batch = EventFeatureBatch(
            values=torch.randn(1, 3, 16, dtype=_DTYPE)  # wrong role count
        )
        with pytest.raises(ValueError):
            model.write(state, bad_batch)

    def test_write_rejects_wrong_feature_dim(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        bad_batch = EventFeatureBatch(
            values=torch.randn(1, 4, 32, dtype=_DTYPE)  # wrong D
        )
        with pytest.raises(ValueError):
            model.write(state, bad_batch)

    def test_write_rejects_wrong_dtype(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        bad_batch = EventFeatureBatch(
            values=torch.randn(1, 4, 16, dtype=torch.float64)
        )
        with pytest.raises(ValueError):
            model.write(state, bad_batch)

    def test_write_rejects_nonfinite(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        bad_values = torch.randn(1, 4, 16, dtype=_DTYPE)
        bad_values[0, 0, 0] = float("inf")
        bad_batch = EventFeatureBatch(values=bad_values)
        with pytest.raises(ValueError):
            model.write(state, bad_batch)

    def test_write_rejects_batch_mismatch(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        batch = EventFeatureBatch(
            values=torch.randn(2, 4, 16, dtype=_DTYPE)  # B=2 vs state B=1
        )
        with pytest.raises(ValueError):
            model.write(state, batch)

    def test_read_rejects_wrong_shape(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        with pytest.raises(ValueError):
            model.read(state, torch.randn(1, 32, dtype=_DTYPE))

    def test_read_rejects_batch_mismatch(self) -> None:
        model = _make_model(D=16)
        state = model.initial_state(1, device=_DEV, dtype=_DTYPE)
        with pytest.raises(ValueError):
            model.read(state, torch.randn(2, 16, dtype=_DTYPE))

    def test_couple_rejects_wrong_dim(self) -> None:
        model = _make_model(D=16)
        with pytest.raises(ValueError):
            model.couple(torch.randn(1, 32, dtype=_DTYPE))

    def test_couple_rejects_nonfinite(self) -> None:
        model = _make_model(D=16)
        bad = torch.randn(1, 64, dtype=_DTYPE)
        bad[0, 0] = float("nan")
        with pytest.raises(ValueError):
            model.couple(bad)


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    def test_d_cortex_must_be_64(self) -> None:
        with pytest.raises(Exception, match="64"):
            ModelConfig(feature_dim=16, d_cortex=32, bank_width=2)

    def test_negative_feature_dim_rejected(self) -> None:
        with pytest.raises(ContractError):
            ModelConfig(feature_dim=-1, d_cortex=64, bank_width=2)

    def test_zero_bank_width_rejected(self) -> None:
        with pytest.raises(ContractError):
            ModelConfig(feature_dim=16, d_cortex=64, bank_width=0)
