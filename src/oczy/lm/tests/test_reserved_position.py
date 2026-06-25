"""Unit tests for the ReservedPosition steering surface.

These tests mock the underlying ``Llama`` instance so they can run without a
GGUF model or GPU.  They verify the prefix-injection logic, clearing,
backward-compatible wrappers, and status introspection.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from oczy.lm.cvec_driver import CVecDriverConfig
from oczy.lm.cvec_driver import LlamaCVecDriver
from oczy.lm.cvec_driver import ReservedPosition
from oczy.lm.cvec_driver import llama_cpp


def _make_driver() -> LlamaCVecDriver:
    """Return a driver wired to a fake LLM."""
    llm = MagicMock()
    llm.n_embd.return_value = 16
    llm.n_vocab.return_value = 100
    llm._ctx.ctx = 1  # raw llama_context_p pointer expected by probe code
    llm._model.model = 0  # raw llama_model_p pointer expected by probe code

    # _probe_n_layers touches raw ctypes helpers; swap them out for the
    # duration of construction and restore immediately afterwards.
    orig_get_model = llama_cpp.llama_get_model
    orig_n_layer = llama_cpp.llama_n_layer

    def _fake_get_model(_ctx: int) -> int:  # noqa: ANN001
        return 0

    def _fake_n_layer(_model: int) -> int:  # noqa: ANN001
        return 4

    llama_cpp.llama_get_model = _fake_get_model
    llama_cpp.llama_n_layer = _fake_n_layer
    try:
        return LlamaCVecDriver(llm, CVecDriverConfig())
    finally:
        llama_cpp.llama_get_model = orig_get_model
        llama_cpp.llama_n_layer = orig_n_layer


def _last_prompt(llm: MagicMock) -> str:
    """Return the prompt string passed to ``create_completion``."""
    call = llm.create_completion.call_args
    assert call is not None
    args, _kwargs = call
    return args[0]


def test_reserved_position_set_and_generate_prefixes() -> None:
    driver = _make_driver()
    prompt = "What is the answer?"
    prefix = "The answer is vertical. "
    driver._llm.create_completion.return_value = {"choices": [{"text": "ok"}]}

    driver.set_reserved_position(
        ReservedPosition(text=prefix, source="test", exact_uptake_score=1.0)
    )

    assert driver.reserved_position_active is True
    assert driver.reserved_position is not None
    assert driver.reserved_position.source == "test"
    assert driver.reserved_position.exact_uptake_score == 1.0

    driver.generate(prompt)
    assert _last_prompt(driver._llm) == prefix + prompt


def test_generate_avoids_duplicating_prefix() -> None:
    driver = _make_driver()
    prefix = "Do not repeat. "
    driver._llm.create_completion.return_value = {"choices": [{"text": "ok"}]}

    driver.set_reserved_position(ReservedPosition(text=prefix))
    driver.generate(prefix + "tail")

    assert _last_prompt(driver._llm) == prefix + "tail"


def test_clear_reserved_position_removes_prefix() -> None:
    driver = _make_driver()
    driver._llm.create_completion.return_value = {"choices": [{"text": "ok"}]}

    driver.set_reserved_position(ReservedPosition(text="X"))
    driver.clear_reserved_position()

    assert driver.reserved_position is None
    assert driver.reserved_position_active is False

    driver.generate("raw")
    assert _last_prompt(driver._llm) == "raw"


def test_articulation_prefix_wrapper_still_works() -> None:
    driver = _make_driver()
    driver._llm.create_completion.return_value = {"choices": [{"text": "ok"}]}

    # Backward-compatible literal-text API.
    driver.set_articulation_prefix("legacy ")
    assert driver.reserved_position_active is True
    assert driver.articulation_prefix == "legacy "

    driver.generate("q")
    assert _last_prompt(driver._llm) == "legacy q"

    driver.clear_articulation_prefix()
    assert driver.reserved_position is None
    assert driver.reserved_position_active is False


def test_status_exposes_source_and_preview_and_serializes() -> None:
    driver = _make_driver()
    long_text = "A" * 80
    driver.set_reserved_position(
        ReservedPosition(text=long_text, source="probe", domain_uptake_score=0.5)
    )

    status = driver.status()
    assert status["reserved_position_active"] is True
    assert status["reserved_position_source"] == "probe"
    preview = status["reserved_position_text_preview"]
    assert preview is not None
    assert preview.startswith("A" * 60)
    assert preview.endswith("...")
    assert "articulation_prefix" not in status
    # Smoke check: status must round-trip through JSON.
    json.dumps(status)

    driver.clear_reserved_position()
    status2 = driver.status()
    assert status2["reserved_position_active"] is False
    assert status2["reserved_position_source"] is None
    assert status2["reserved_position_text_preview"] is None
