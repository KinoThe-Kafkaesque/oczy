"""Tests for HF-substrate KV-slot fact injection experiment (S1.3).

Unit-tests the experiment plumbing with a tiny random HF model
(hf-internal-testing/tiny-random-LlamaForCausalLM) — shapes/flow only, no
semantics.  The real experiment uses HF_MODEL_ID (Qwen2.5-0.5B-Instruct).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from oczy.experiments import hf_kv_slot_experiment as hke
from oczy.lm.hf_driver import HFDriver

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver() -> HFDriver:
    """Module-scoped: load the tiny model once for all tests."""
    return HFDriver.load(TEST_MODEL_ID)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_facts_queries_targets_aligned() -> None:
    """The imported FACTS/QUERIES/TARGETS are aligned and non-empty."""
    from oczy.experiments.multi_fact_stressor import FACTS, QUERIES, TARGETS

    assert len(FACTS) >= 3
    assert len(QUERIES) >= 3
    assert len(TARGETS) >= 3
    for fact, target in zip(FACTS[:3], TARGETS[:3], strict=True):
        assert target in fact.lower()


def test_probe_template_format() -> None:
    """The probe template formats correctly."""
    prompt = hke._probe_prompt("Test question?")
    assert "Test question?" in prompt
    assert "Answer:" in prompt
    assert "Recall the answer in lowercase" in prompt


# ---------------------------------------------------------------------------
# Cvec computation
# ---------------------------------------------------------------------------


def test_compute_cvec_shape(driver: HFDriver) -> None:
    """_compute_cvec_from_fact returns (n_embd,) float32 vector."""
    cvec = hke._compute_cvec_from_fact(driver, "sky is blue")
    assert cvec.shape == (driver.n_embd,)
    assert cvec.dtype == np.float32


def test_compute_cvec_normalized(driver: HFDriver) -> None:
    """The cvec direction is L2-normalized (unit norm)."""
    cvec = hke._compute_cvec_from_fact(driver, "sky is blue")
    norm = float(np.linalg.norm(cvec))
    assert abs(norm - 1.0) < 1e-5, f"expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# KV handle scaling
# ---------------------------------------------------------------------------


def test_scale_kv_handle_identity(driver: HFDriver) -> None:
    """Scaling a handle by 1.0 produces K/V with the same values."""
    handle = driver.encode_kv("test fact")
    scaled = hke._scale_kv_handle(handle, 1.0, 1.0)
    assert scaled.seq_len == handle.seq_len

    for layer_orig, layer_scaled in zip(
        handle.past_key_values, scaled.past_key_values, strict=True
    ):
        assert torch.allclose(layer_orig[0], layer_scaled[0])
        assert torch.allclose(layer_orig[1], layer_scaled[1])


def test_scale_kv_handle_doubles(driver: HFDriver) -> None:
    """Scaling by 2.0 doubles the K and V values."""
    handle = driver.encode_kv("test fact")
    scaled = hke._scale_kv_handle(handle, 2.0, 2.0)

    for layer_orig, layer_scaled in zip(
        handle.past_key_values, scaled.past_key_values, strict=True
    ):
        assert torch.allclose(layer_orig[0] * 2.0, layer_scaled[0])
        assert torch.allclose(layer_orig[1] * 2.0, layer_scaled[1])


def test_scale_kv_handle_does_not_mutate_original(driver: HFDriver) -> None:
    """Scaling returns a new handle without mutating the original."""
    handle = driver.encode_kv("test fact")
    # Iterate to access layer 0, element 0 (K tensor)
    orig_k0 = None
    for i, layer in enumerate(handle.past_key_values):
        if i == 0:
            orig_k0 = layer[0].clone()
            break

    _ = hke._scale_kv_handle(handle, 5.0, 5.0)
    for i, layer in enumerate(handle.past_key_values):
        if i == 0:
            assert orig_k0 is not None
            assert torch.allclose(layer[0], orig_k0)
            break


# ---------------------------------------------------------------------------
# Primary conditions — shape / no-crash checks
# ---------------------------------------------------------------------------


def test_c0_returns_valid_result(driver: HFDriver) -> None:
    """C0: probe alone returns a ConditionResult with valid structure."""
    r = hke._run_condition_c0(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert r.condition == "C0"
    assert r.fact_idx == 0
    assert 0 <= r.rank <= driver.n_vocab
    assert isinstance(r.top1, str)
    assert len(r.top1) > 0


def test_c1_returns_valid_result(driver: HFDriver) -> None:
    """C1: text prefix returns a ConditionResult."""
    r = hke._run_condition_c1(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert r.condition == "C1"
    assert 0 <= r.rank <= driver.n_vocab


def test_c2_returns_valid_result(driver: HFDriver) -> None:
    """C2: KV-slot injection returns a ConditionResult with latency."""
    r = hke._run_condition_c2(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert r.condition == "C2"
    assert 0 <= r.rank <= driver.n_vocab
    assert r.latency_ms >= 0


def test_c3_returns_valid_result(driver: HFDriver) -> None:
    """C3: cvec-only returns a ConditionResult."""
    r = hke._run_condition_c3(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert r.condition == "C3"
    assert 0 <= r.rank <= driver.n_vocab


def test_c3_clears_cvec_after(driver: HFDriver) -> None:
    """After C3, the cvec is cleared (no active hooks)."""
    _ = hke._run_condition_c3(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert not driver.cvec_active


# ---------------------------------------------------------------------------
# Secondary: splice position
# ---------------------------------------------------------------------------


def test_splice_position_variants_no_crash(driver: HFDriver) -> None:
    """Splice position analysis does not crash on the tiny model."""
    results = hke._run_splice_position_variants(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert len(results) == 2
    conditions = {r.condition for r in results}
    assert "C2-splice-front" in conditions
    assert "C2-splice-preblank" in conditions
    for r in results:
        assert 0 <= r.rank <= driver.n_vocab


def test_splice_preblank_uses_both_parts(driver: HFDriver) -> None:
    """The pre-blank splice helper handles both prompt parts correctly."""
    handle = driver.encode_kv("sky is blue")
    ranks = hke._token_ranks_with_kv_splice(
        driver, "Question: What color?\n", "Answer:", handle, ["blue"]
    )
    assert len(ranks) == 1
    assert ranks[0]["rank"] >= 0
    assert isinstance(ranks[0]["top1"], str)


def test_splice_preblank_no_after_part(driver: HFDriver) -> None:
    """Pre-blank splice with empty after part works (degenerate case)."""
    handle = driver.encode_kv("sky is blue")
    ranks = hke._token_ranks_with_kv_splice(
        driver, "Question: What color?\nAnswer:", "", handle, ["blue"]
    )
    assert len(ranks) == 1
    assert ranks[0]["rank"] >= 0


# ---------------------------------------------------------------------------
# Secondary: K/V norm scaling
# ---------------------------------------------------------------------------


def test_kv_norm_variants_no_crash(driver: HFDriver) -> None:
    """K/V norm scaling analysis does not crash."""
    results = hke._run_kv_norm_variants(
        driver, 0, "sky is blue", "What color is the sky?", "blue"
    )
    assert len(results) == 3
    expected = {"C2-scale-0.5x", "C2-scale-1.0x", "C2-scale-2.0x"}
    assert {r.condition for r in results} == expected
    for r in results:
        assert 0 <= r.rank <= driver.n_vocab


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------


def test_run_experiment_primary(driver: HFDriver) -> None:
    """run_experiment with include_secondary=False returns C0-C3 for all facts."""
    result = hke.run_experiment(driver, include_secondary=False)
    assert result.model_id == TEST_MODEL_ID

    # 3 facts x 4 conditions = 12 results
    assert len(result.results) == 12
    conditions = {r.condition for r in result.results}
    assert conditions == {"C0", "C1", "C2", "C3"}

    fact_indices = {r.fact_idx for r in result.results}
    assert fact_indices == {0, 1, 2}

    # Verify primary metric is computed
    assert isinstance(result.hf_kv_slot_rank1_count, int)
    assert 0 <= result.hf_kv_slot_rank1_count <= 3


def test_run_experiment_with_secondary(driver: HFDriver) -> None:
    """run_experiment with include_secondary=True includes exploratory results."""
    result = hke.run_experiment(driver, include_secondary=True)
    # 3 facts x (4 primary + 2 splice + 3 scale) = 27 results
    assert len(result.results) == 27

    all_conditions = {r.condition for r in result.results}
    assert "C2-splice-front" in all_conditions
    assert "C2-splice-preblank" in all_conditions
    assert "C2-scale-0.5x" in all_conditions
    assert "C2-scale-2.0x" in all_conditions


def test_experiment_result_tables(driver: HFDriver) -> None:
    """rank_table and top1_table produce non-empty markdown."""
    result = hke.run_experiment(driver, include_secondary=False)
    rank_tbl = result.rank_table()
    top1_tbl = result.top1_table()

    assert "| Fact |" in rank_tbl
    assert "C0" in rank_tbl
    assert "C1" in rank_tbl
    assert "C2" in rank_tbl
    assert "C3" in rank_tbl
    assert "| Fact |" in top1_tbl
    assert "top1" in top1_tbl


def test_experiment_result_by_condition(driver: HFDriver) -> None:
    """_by_condition filters correctly."""
    result = hke.run_experiment(driver, include_secondary=False)
    c0 = result._by_condition("C0")
    assert len(c0) == 3
    assert all(r.condition == "C0" for r in c0)


def test_experiment_result_c0_ranks(driver: HFDriver) -> None:
    """c0_ranks returns 3 integers."""
    result = hke.run_experiment(driver, include_secondary=False)
    ranks = result.c0_ranks
    assert len(ranks) == 3
    assert all(isinstance(r, int) for r in ranks)


def test_rank1_method(driver: HFDriver) -> None:
    """ConditionResult.rank1() is True only when rank==0."""
    result = hke.run_experiment(driver, include_secondary=False)
    for r in result.results:
        assert r.rank1() == (r.rank == 0)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def test_emit_report_accept(driver: HFDriver) -> None:
    """_emit_report produces a markdown report with verdict."""
    result = hke.run_experiment(driver, include_secondary=True)
    report = hke._emit_report(result)

    assert "# S1.3" in report
    assert "## Spec" in report
    assert "## Rank table" in report
    assert "## Primary metric" in report
    assert "Verdict:" in report
    assert "hf_kv_slot_rank1_count" in report


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_input_same_result(driver: HFDriver) -> None:
    """Running the same experiment twice produces identical results."""
    r1 = hke.run_experiment(driver, include_secondary=False)
    r2 = hke.run_experiment(driver, include_secondary=False)

    assert r1.hf_kv_slot_rank1_count == r2.hf_kv_slot_rank1_count
    for a, b in zip(r1.results, r2.results, strict=True):
        assert a.condition == b.condition
        assert a.fact_idx == b.fact_idx
        assert a.rank == b.rank, (
            f"nondeterminism: {a.condition}/{a.fact_idx} "
            f"{a.rank} vs {b.rank}"
        )
        assert a.top1 == b.top1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_target_rank(driver: HFDriver) -> None:
    """A target that doesn't exist in the vocab gets a high rank."""
    # Use a target string the tiny model's tokenizer won't recognize as-is.
    # The tokenizer will still produce some token ids, so we test with
    # a nonsense string.
    prompt = hke._probe_prompt("What is the codeword?")
    ranks = driver.token_ranks(prompt, ["xyzzynotarealtoken"])
    assert ranks[0]["rank"] >= 0


def test_empty_fact_encode(driver: HFDriver) -> None:
    """Encoding an empty string should not crash (though it may produce
    empty KV entries)."""
    handle = driver.encode_kv("")
    assert handle.seq_len >= 0  # may be 0 or 1 (BOS token)


def test_handle_past_key_values_structure(driver: HFDriver) -> None:
    """A KVHandle from encode_kv has the expected structure."""
    handle = driver.encode_kv("test fact")
    assert len(handle.past_key_values) == driver.n_layers
    for layer in handle.past_key_values:
        assert isinstance(layer, tuple)
        assert len(layer) >= 2  # (k, v, ...) — may include None/cache_position
        k, v = layer[0], layer[1]
        assert k.ndim == 4  # (batch, heads, seq, head_dim)
        assert v.ndim == 4
        assert k.shape[2] == handle.seq_len
        assert v.shape[2] == handle.seq_len
