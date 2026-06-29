"""Tests for OrganismAgent cortex-LM answer delegation flag."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from oczy.experiments.organism import OrganismAgent


class _MockCortexAgent:
    """Minimal CortexAgent stand-in that only implements answer()."""

    def answer(
        self,
        request: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        metabolize: bool = False,
    ) -> dict[str, Any]:
        return {"answer": "cortex reply"}


def test_organism_delegates_to_cortex_agent() -> None:
    mock_cortex = _MockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock_cortex}
    )
    assert organism.answer("x") == "cortex reply"


def test_organism_legacy_answer_path_still_runs() -> None:
    organism = OrganismAgent({})
    organism.plastic_cortex.answer = lambda request: "legacy plastic reply"
    assert organism.answer("x") == "legacy plastic reply"


def test_organism_missing_cortex_agent_fallback_warns() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_cortex_lm_answer": True})
    assert organism.cortex_agent is None


# ---------------------------------------------------------------------------
# Cortex perceive/observe correction path + context-addressed slot store
# ---------------------------------------------------------------------------


class _MockDriver:
    """Deterministic text-sensitive embedding surface for scope keying."""

    def __init__(self, n_embd: int = 8) -> None:
        self.n_embd = n_embd

    def peek_embedding(self, text: str, last_token_only: bool = False) -> np.ndarray:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:4], "big"
        )
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=self.n_embd).astype(np.float32)


class _MockCortex:
    """Holds a warm_state vector the organism can read/write."""

    def __init__(self, d_cortex: int = 4) -> None:
        self.warm_state = np.zeros(d_cortex, dtype=np.float32)


class _ScopeMockCortexAgent:
    """CortexAgent stand-in exposing driver/cortex/perceive/observe/answer.

    ``perceive`` produces a distinct warm_state per call so two corrections
    yield different stored slots; ``answer`` snapshots the warm_state the
    organism applied before articulating so tests can assert on it.
    """

    def __init__(self) -> None:
        self.driver = _MockDriver()
        self.cortex = _MockCortex()
        self.perceive_calls: list[dict[str, Any]] = []
        self.observe_calls: int = 0
        self.answer_calls: list[dict[str, Any]] = []
        self._last_hidden: np.ndarray | None = None
        self._warm_seq: int = 0

    def perceive(self, utterance: str, correction_signal: float | None = None):
        self.perceive_calls.append(
            {"utterance": utterance, "correction_signal": correction_signal}
        )
        self._warm_seq += 1
        warm = np.full(
            self.cortex.warm_state.shape, float(self._warm_seq), dtype=np.float32
        )
        self.cortex.warm_state = warm
        self._last_hidden = self.driver.peek_embedding(utterance)
        return warm.copy()

    def observe(self) -> np.ndarray:
        self.observe_calls += 1
        return self.cortex.warm_state.copy()

    def answer(
        self,
        request: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        metabolize: bool = False,
        ) -> dict[str, Any]:
        self.answer_calls.append(
            {"request": request, "warm": self.cortex.warm_state.copy()}
        )
        return {"answer": f"cortex reply {request}"}


def test_correction_routes_through_cortex_perceive() -> None:
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    assert mock.perceive_calls, "perceive must be called on correction"
    call = mock.perceive_calls[0]
    assert call["utterance"] == "No, here 'file' means a disk file."
    assert call["correction_signal"] == 1.0


def test_correction_stores_warm_state_in_scope_slot() -> None:
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    # A slot must have been written, keyed by the request embedding.
    assert len(organism._scope_slot_keys) == 1
    assert len(organism._scope_slot_warm) == 1
    # The stored warm_state is the one perceive returned (seq=1).
    np.testing.assert_allclose(
        organism._scope_slot_warm[0], np.ones(4, dtype=np.float32)
    )


def test_answer_applies_stored_scope_slot_before_articulate() -> None:
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    # perceive ran once during learning; answer must NOT re-perceive but must
    # restore the stored slot warm_state before the mock answer() records it.
    organism.answer("open the file")
    assert len(mock.answer_calls) == 1
    np.testing.assert_allclose(
        mock.answer_calls[0]["warm"], np.ones(4, dtype=np.float32)
    )


def test_answer_zeros_warm_state_on_scope_miss() -> None:
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    # Teach one context, then ask a completely unrelated context.
    organism.learn("open the file", "No, here 'file' means a disk file.")
    # Seed a non-zero warm_state so we can detect the zeroing.
    mock.cortex.warm_state = np.full(4, 9.0, dtype=np.float32)
    organism.answer("tell me a joke")
    np.testing.assert_allclose(
        mock.answer_calls[0]["warm"], np.zeros(4, dtype=np.float32)
    )


def test_dual_sense_scope_slots_coexist() -> None:
    """Correcting one sense must not overwrite another context's slot."""
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    organism.learn("open the cell", "No, here 'cell' means a spreadsheet cell.")
    # Two distinct slots keyed by the two request embeddings.
    assert len(organism._scope_slot_keys) == 2
    # Re-asking each context restores its own warm_state (seq 1 and 2).
    organism.answer("open the file")
    np.testing.assert_allclose(
        mock.answer_calls[-1]["warm"], np.ones(4, dtype=np.float32)
    )
    organism.answer("open the cell")
    np.testing.assert_allclose(
        mock.answer_calls[-1]["warm"], np.full(4, 2.0, dtype=np.float32)
    )


def test_scope_key_returns_none_without_driver() -> None:
    organism = OrganismAgent({})
    assert organism._scope_key("anything") is None


def test_cortex_correction_path_runs_when_agent_present() -> None:
    """A cortex_agent attached always populates the scope-slot store,
    without requiring use_cortex_lm_answer.  The label is bound via the
    driver embedding, so no perceive() call is needed unless LM articulation
    is enabled.
    """
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent({"cortex_agent": mock})
    assert not organism.use_cortex_lm_answer
    organism.learn("open the file", "No, here 'file' means a disk file.")
    assert mock.perceive_calls == []
    assert organism._scope_slot_keys
    assert organism._scope_slot_label
    assert organism._scope_slot_label[0] == "a disk file"


def test_cortex_lm_answer_flag_perceives_correction() -> None:
    """When use_cortex_lm_answer=True, perceiving the correction captures
    warm_state for later articulation."""
    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    assert len(mock.perceive_calls) == 1
    assert organism._scope_slot_keys
    assert organism._scope_slot_warm[0] is not None
    assert organism._scope_slot_label[0] == "a disk file"


def test_organism_pickle_roundtrip_preserves_scope_slots() -> None:
    import io
    import pickle

    mock = _ScopeMockCortexAgent()
    organism = OrganismAgent(
        {"use_cortex_lm_answer": True, "cortex_agent": mock}
    )
    organism.learn("open the file", "No, here 'file' means a disk file.")
    buf = io.BytesIO()
    pickle.dump(organism, buf)
    buf.seek(0)
    restored = pickle.load(buf)
    assert len(restored._scope_slot_keys) == 1
    np.testing.assert_allclose(
        restored._scope_slot_warm[0], np.ones(4, dtype=np.float32)
    )
