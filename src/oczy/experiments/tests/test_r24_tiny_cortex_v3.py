"""Contract tests for the R24-v3 tiny differentiable cortex."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oczy.experiments.r24_tiny_decoder.tiny_cortex_v3 import (
    EVENT_WRITES,
    FEATURE_DIM,
    PARAMETER_BUDGET,
    PERSISTENT_BYTES_PER_TASK,
    READOUT_DIM,
    STATE_DIM,
    CortexState,
    EventFeatureBatch,
    TinyCortexConfig,
    TinyCortexV3,
)


def _randn(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


def _nonzero_state(batch_size: int = 2) -> CortexState:
    return CortexState(
        fast=_randn(batch_size, STATE_DIM, STATE_DIM, seed=1),
        slow=_randn(batch_size, STATE_DIM, STATE_DIM, seed=2),
    )


def test_one_write_matches_r20_equation_and_does_not_mutate_inputs() -> None:
    model = TinyCortexV3(init_seed=7)
    state = _nonzero_state()
    event = _randn(2, FEATURE_DIM, seed=3)
    fast_before = state.fast.clone()
    slow_before = state.slow.clone()
    event_before = event.clone()

    projected = torch.tanh(model.feature_projection(event) + model.role_embeddings[0])
    encoded = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
    fast_context = torch.bmm(state.fast, encoded.unsqueeze(-1)).squeeze(-1)
    hidden = F.silu(model.writer_hidden(torch.cat((encoded, fast_context), dim=-1)))
    key = torch.tanh(model.writer_key(hidden))
    key = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
    value = torch.tanh(model.writer_value(hidden))
    eta = torch.sigmoid(model.writer_eta(hidden))
    decay = torch.sigmoid(model.writer_decay(hidden))
    expected = decay.view(2, 1, 1) * state.fast + eta.view(2, 1, 1) * torch.bmm(
        value.unsqueeze(-1), key.unsqueeze(1)
    )

    result = model.write(state, EventFeatureBatch(event))

    assert result.state.fast.shape == (2, STATE_DIM, STATE_DIM)
    assert result.key.shape == result.value.shape == (2, STATE_DIM)
    assert result.eta.shape == result.decay.shape == (2, 1)
    assert torch.allclose(result.state.fast, expected)
    assert torch.equal(result.state.slow, state.slow)
    assert result.state.slow.data_ptr() != state.slow.data_ptr()
    assert torch.equal(state.fast, fast_before)
    assert torch.equal(state.slow, slow_before)
    assert torch.equal(event, event_before)


def test_exact_three_write_unroll_matches_three_single_writes() -> None:
    model = TinyCortexV3(init_seed=11)
    state = model.initial_state(3)
    events = _randn(3, EVENT_WRITES, FEATURE_DIM, seed=12)
    fast_before = state.fast.clone()
    events_before = events.clone()

    manual = state
    single_results = []
    for event_index in range(EVENT_WRITES):
        single = model.write(manual, events[:, event_index, :])
        single_results.append(single)
        manual = single.state
    unrolled = model.write_three(state, EventFeatureBatch(events))

    assert torch.equal(unrolled.state.fast, manual.fast)
    assert torch.equal(unrolled.state.slow, manual.slow)
    assert torch.equal(
        unrolled.keys,
        torch.stack([result.key for result in single_results], dim=1),
    )
    assert unrolled.keys.shape == unrolled.values.shape == (3, EVENT_WRITES, STATE_DIM)
    assert unrolled.etas.shape == unrolled.decays.shape == (3, EVENT_WRITES, 1)
    assert torch.equal(model.write_events(state, events).state.fast, manual.fast)
    assert torch.equal(state.fast, fast_before)
    assert torch.equal(events, events_before)

    with pytest.raises(ValueError, match="shape"):
        model.write_three(state, events[:, :2, :])


def test_consolidation_equation_clears_fast_bit_exactly_without_mutation() -> None:
    model = TinyCortexV3(init_seed=13)
    state = _nonzero_state()
    fast_before = state.fast.clone()
    slow_before = state.slow.clone()

    probe = model.consolidation_probe
    probe = probe / (probe.norm(dim=-1, keepdim=True) + 1e-8)
    fast_probe = torch.matmul(state.fast, probe)
    slow_probe = torch.matmul(state.slow, probe)
    hidden = F.silu(
        model.consolidation_hidden(torch.cat((fast_probe, slow_probe), dim=-1))
    )
    gate = torch.sigmoid(model.consolidation_gate(hidden))
    expected_slow = (1.0 - gate.view(2, 1, 1)) * state.slow + gate.view(
        2, 1, 1
    ) * state.fast

    result = model.consolidate(state)

    assert result.gate.shape == (2, 1)
    assert torch.all((result.gate >= 0.0) & (result.gate <= 1.0))
    assert torch.allclose(result.state.slow, expected_slow)
    assert torch.equal(result.state.fast, torch.zeros_like(result.state.fast))
    assert result.state.fast.data_ptr() != state.fast.data_ptr()
    assert result.state.slow.data_ptr() != state.slow.data_ptr()
    assert torch.equal(state.fast, fast_before)
    assert torch.equal(state.slow, slow_before)
    assert model.logical_persistent_bytes(result.state) == PERSISTENT_BYTES_PER_TASK
    assert model.post_consolidation_bytes_per_task(result.state) == 1024


def test_query_reader_and_projection_match_equations_and_shapes() -> None:
    model = TinyCortexV3(init_seed=17)
    state = _nonzero_state(batch_size=4)
    query_features = _randn(4, FEATURE_DIM, seed=18)

    query = torch.tanh(
        model.feature_projection(query_features) + model.role_embeddings[1]
    )
    query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
    fast_read = torch.bmm(state.fast, query.unsqueeze(-1)).squeeze(-1)
    slow_read = torch.bmm(state.slow, query.unsqueeze(-1)).squeeze(-1)
    hidden = F.silu(
        model.reader_hidden(torch.cat((query, fast_read, slow_read), dim=-1))
    )
    expected_state_readout = torch.tanh(model.reader_out(hidden))
    expected_r = torch.tanh(model.readout_projection(expected_state_readout))

    state_readout = model.read_state(state, query_features)
    r = model.read(state, query_features)

    assert state_readout.shape == (4, STATE_DIM)
    assert r.shape == (4, READOUT_DIM)
    assert torch.allclose(state_readout, expected_state_readout)
    assert torch.allclose(model.project(state_readout), expected_r)
    assert torch.allclose(r, expected_r)
    assert torch.equal(model(state, query_features), r)
    assert torch.isfinite(r).all()


def test_parameter_count_is_exact_and_within_registered_budget() -> None:
    model = TinyCortexV3()
    assert model.parameter_count() == 6995
    assert sum(model.parameter_breakdown().values()) == 6995
    assert sum(parameter.numel() for parameter in model.parameters()) == 6995
    assert model.parameter_count() <= PARAMETER_BUDGET == 10_240


def test_all_parameter_groups_receive_finite_nonzero_gradients() -> None:
    model = TinyCortexV3(init_seed=19)
    state = model.initial_state(2)
    events = _randn(2, EVENT_WRITES, FEATURE_DIM, seed=20)
    queries = _randn(2, FEATURE_DIM, seed=21)
    target = _randn(2, READOUT_DIM, seed=22)

    state = model.write_three(state, events).state
    state = model.consolidate(state).state
    readout = model.read(state, queries)
    loss = (readout - target).square().mean()
    loss.backward()

    assert set(model.parameter_breakdown()) == {
        name.split(".")[0] for name, _ in model.named_parameters()
    }
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_seeded_initialization_hash_is_local_and_deterministic() -> None:
    torch.manual_seed(1_234_567)
    global_state_before = torch.random.get_rng_state().clone()
    first = TinyCortexV3(init_seed=7)
    global_state_after = torch.random.get_rng_state()
    second = TinyCortexV3(init_seed=7)
    different = TinyCortexV3(init_seed=8)

    assert torch.equal(global_state_before, global_state_after)
    assert first.parameter_hash() == second.parameter_hash()
    assert first.parameter_hash() != different.parameter_hash()
    assert first.parameter_hash() == (
        "933c84fbb38e6af78a8dcb49cd6932732e889ecc94e8b9e97bc32afad70dc4d5"
    )
    assert len(first.parameter_hash()) == 64


def test_zero_and_swap_clone_state_without_changing_inputs_or_parameters() -> None:
    model = TinyCortexV3(init_seed=23)
    theta_before = model.parameter_hash()
    recipient = _nonzero_state()
    donor = CortexState(
        fast=_randn(2, STATE_DIM, STATE_DIM, seed=24),
        slow=_randn(2, STATE_DIM, STATE_DIM, seed=25),
    )
    recipient_before = (recipient.fast.clone(), recipient.slow.clone())
    donor_before = (donor.fast.clone(), donor.slow.clone())

    zeroed = model.zero_state(recipient)
    swapped = model.swap_state(recipient, donor)

    assert torch.count_nonzero(zeroed.fast).item() == 0
    assert torch.count_nonzero(zeroed.slow).item() == 0
    assert torch.equal(swapped.fast, donor.fast)
    assert torch.equal(swapped.slow, donor.slow)
    assert swapped.fast.data_ptr() != donor.fast.data_ptr()
    assert swapped.slow.data_ptr() != donor.slow.data_ptr()
    assert torch.equal(recipient.fast, recipient_before[0])
    assert torch.equal(recipient.slow, recipient_before[1])
    assert torch.equal(donor.fast, donor_before[0])
    assert torch.equal(donor.slow, donor_before[1])
    assert model.parameter_hash() == theta_before

    with pytest.raises(ValueError, match="shape"):
        model.swap_state(recipient, model.initial_state(1))


def test_shape_dtype_and_finiteness_validation() -> None:
    model = TinyCortexV3()
    state = model.initial_state(2)

    with pytest.raises(ValueError, match="shape"):
        model.write(state, torch.randn(2, FEATURE_DIM - 1))
    with pytest.raises(ValueError, match="float32"):
        model.write(state, torch.randn(2, FEATURE_DIM, dtype=torch.float64))
    nonfinite_event = torch.zeros(2, FEATURE_DIM)
    nonfinite_event[0, 0] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        model.write(state, nonfinite_event)
    nonfinite_state = CortexState(
        fast=torch.full((2, STATE_DIM, STATE_DIM), torch.nan),
        slow=torch.zeros(2, STATE_DIM, STATE_DIM),
    )
    with pytest.raises(ValueError, match="non-finite"):
        model.consolidate(nonfinite_state)
    with pytest.raises(ValueError, match="shape"):
        model.read(state, torch.randn(1, FEATURE_DIM))
    with pytest.raises(ValueError, match="shape"):
        model.project(torch.randn(2, STATE_DIM + 1))
    with pytest.raises(ValueError, match="float32"):
        model.initial_state(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="feature_dim"):
        TinyCortexConfig(feature_dim=32)


def test_repeated_updates_remain_finite_and_preconsolidation_uses_two_matrices() -> None:
    model = TinyCortexV3(init_seed=29)
    state = model.initial_state(1)
    for event_index in range(100):
        state = model.write(
            state,
            _randn(1, FEATURE_DIM, seed=30 + event_index),
        ).state
    assert torch.isfinite(state.fast).all()
    assert torch.isfinite(state.slow).all()
    assert model.logical_persistent_bytes(state) == 2 * PERSISTENT_BYTES_PER_TASK
    with pytest.raises(ValueError, match="bit-zero"):
        model.post_consolidation_bytes_per_task(state)
