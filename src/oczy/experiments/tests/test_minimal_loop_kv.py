"""Contract tests for MinimalOrganismKV (S2.2 KV content channel).

Exercises the observable contracts of the minimal organism against a tiny
random LlamaForCausalLM loaded once per module: prefix vs kv content
channels, prompt-token parity/difference, KV-handle caching semantics, and
the C0 vanilla-driver scoring path.
"""

from __future__ import annotations

from typing import Any

import pytest

from oczy.experiments.minimal_loop_kv import (
    MinimalOrganismKV,
    _PROBE_TEMPLATE,
    _score_probe,
)
from oczy.experiments.organism_curriculum.dataset import Episode, Probe
from oczy.lm.hf_driver import HFDriver, KVHandle

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver() -> HFDriver:
    """Module-scoped: load the tiny model once for all tests."""
    return HFDriver.load(TEST_MODEL_ID)


def _episode() -> dict[str, Any]:
    """A single correction episode that yields non-empty content text."""
    return {
        "initial_request": "What is X?",
        "default_response": "I don't know",
        "correction_utterance": "No, 'X' means alpha.",
        "corrected_label": "alpha",
        "corrected_response": "X means alpha",
        "domain": "test",
    }


def _plain_prompt_tokens(drv: HFDriver, query: str) -> int:
    """Token count of the bare probe prompt with no prefix applied."""
    prompt = _PROBE_TEMPLATE.format(query)
    return int(drv._tokenize(prompt).shape[1])  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1 & 2 — content channel selects whether an articulation prefix is set
# ---------------------------------------------------------------------------


def test_kv_channel_never_sets_prefix(driver: HFDriver) -> None:
    """Consolidating a kv-channel organism must not set an articulation prefix."""
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert driver.articulation_prefix is None
    finally:
        driver.clear_articulation_prefix()


def test_prefix_channel_sets_prefix(driver: HFDriver) -> None:
    """Consolidating a prefix-channel organism must set a non-empty prefix."""
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="prefix", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        prefix = driver.articulation_prefix
        assert prefix is not None
        assert prefix != ""
    finally:
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 3 & 4 — prompt-token audit: C2 parity vs C0, C1 strictly larger than C0
# ---------------------------------------------------------------------------


def test_prompt_token_parity_kv_vs_vanilla(driver: HFDriver) -> None:
    """C2 (kv) visible prompt tokens equal the plain prompt (no prefix)."""
    driver.clear_articulation_prefix()
    query = "What is X?"
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert organism.prompt_token_count(query) == _plain_prompt_tokens(
            driver, query
        )
    finally:
        driver.clear_articulation_prefix()


def test_prompt_token_difference_prefix_vs_vanilla(driver: HFDriver) -> None:
    """C1 (prefix) visible prompt tokens strictly exceed the plain prompt."""
    driver.clear_articulation_prefix()
    query = "What is X?"
    try:
        organism = MinimalOrganismKV(content_channel="prefix", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert organism.prompt_token_count(query) > _plain_prompt_tokens(
            driver, query
        )
    finally:
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 5 & 6 — KV handle is created once at consolidation, never re-encoded at answer
# ---------------------------------------------------------------------------


def _kv_handle_seq_len(handle: KVHandle) -> int:
    """Original encoded length of the handle, captured before any answer()
    extends the cache by appending generated tokens.

    ``seq_len`` is set once at ``encode_kv`` time and never reset, so it is a
    stable fingerprint of the original encoding — unlike the live cache size,
    which grows as ``generate_with_kv`` appends tokens.
    """
    return handle.seq_len


def test_kv_handle_created_once_per_consolidation(driver: HFDriver) -> None:
    """answer() must reuse the cached handle object rather than re-encoding.

    The handle's live cache grows as ``generate_with_kv`` appends generated
    tokens, so byte-size is NOT stable across answers; the contract is that the
    SAME handle object (and its original ``seq_len``) is reused — i.e.
    ``encode_kv`` is not called again.
    """
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert isinstance(organism.kv_handle, KVHandle)
        handle_before = organism.kv_handle
        seq_len_before = _kv_handle_seq_len(handle_before)
        assert seq_len_before > 0

        organism.answer("What is X?")
        assert organism.kv_handle is handle_before
        assert _kv_handle_seq_len(organism.kv_handle) == seq_len_before

        organism.answer("What is X again?")
        assert organism.kv_handle is handle_before
        assert _kv_handle_seq_len(organism.kv_handle) == seq_len_before
    finally:
        driver.clear_articulation_prefix()



def test_encode_kv_not_called_during_answer(driver: HFDriver) -> None:
    """After consolidation, answer() must not call driver.encode_kv."""
    driver.clear_articulation_prefix()
    original_encode_kv = driver.encode_kv
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()  # this call uses the real encode_kv

        def _exploding_encode_kv(_text: str) -> KVHandle:
            raise AssertionError("encode_kv must not be called during answer()")

        driver.encode_kv = _exploding_encode_kv  # type: ignore[method-assign]
        answer = organism.answer("What is X?")
        assert isinstance(answer, str)
    finally:
        driver.encode_kv = original_encode_kv  # type: ignore[method-assign]
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 7 — splice position: kv answer uses generate_with_kv with a pre-blank prompt
# ---------------------------------------------------------------------------


def test_kv_answer_uses_generate_with_kv_with_preblank_prompt(
    driver: HFDriver,
) -> None:
    """kv answer() routes through generate_with_kv (not generate) and the
    prompt passed in must not contain the consolidated content text."""
    driver.clear_articulation_prefix()
    original_gwkv = driver.generate_with_kv
    original_gen = driver.generate
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        content_text = organism.consolidate()
        assert content_text, "fixture episode must produce non-empty content"

        recorded: dict[str, Any] = {}

        def _recording_gwkv(prompt: str, handle: KVHandle, max_tokens: int = 32) -> str:
            recorded["gwkv_prompt"] = prompt
            return original_gwkv(prompt, handle, max_tokens=max_tokens)

        def _recording_gen(prompt: str, max_tokens: int = 32) -> str:
            recorded["gen_prompt"] = prompt
            return original_gen(prompt, max_tokens=max_tokens)

        driver.generate_with_kv = _recording_gwkv  # type: ignore[method-assign]
        driver.generate = _recording_gen  # type: ignore[method-assign]

        organism.answer("What is X?")

        assert "gwkv_prompt" in recorded, "answer() did not use generate_with_kv"
        assert "gen_prompt" not in recorded, "answer() used plain generate() for kv"
        assert content_text not in recorded["gwkv_prompt"], (
            "consolidated content leaked into the visible prompt"
        )
    finally:
        driver.generate_with_kv = original_gwkv  # type: ignore[method-assign]
        driver.generate = original_gen  # type: ignore[method-assign]
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 8 — consolidated_prefix_token_count is zero for the kv channel
# ---------------------------------------------------------------------------


def test_consolidated_prefix_token_count_zero_for_kv(driver: HFDriver) -> None:
    """kv channel never carries a prefix, so its prefix token count is 0."""
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert organism.consolidated_prefix_token_count() == 0
    finally:
        driver.clear_articulation_prefix()


def test_consolidated_prefix_token_count_nonzero_for_prefix(
    driver: HFDriver,
) -> None:
    """prefix channel carries a prefix, so its token count is positive
    (guards against the kv-zero result being vacuous)."""
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="prefix", driver=driver)
        organism.perceive(_episode())
        organism.consolidate()
        assert organism.consolidated_prefix_token_count() > 0
    finally:
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 9 — answer() never touches the hippocampus
# ---------------------------------------------------------------------------


class _ExplodingHippocampus:
    """Stand-in that raises on any attribute access (method call)."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"hippocampus.{name} must not be accessed during answer()"
        )


def test_answer_does_not_access_hippocampus(driver: HFDriver) -> None:
    """perceive() stores into the hippocampus, but answer() must not read it."""
    driver.clear_articulation_prefix()
    try:
        organism = MinimalOrganismKV(content_channel="kv", driver=driver)
        organism.perceive(_episode())  # exercises the real hippocampus
        organism.consolidate()

        organism.hippocampus = _ExplodingHippocampus()  # type: ignore[assignment]
        answer = organism.answer("What is X?")
        assert isinstance(answer, str)
    finally:
        driver.clear_articulation_prefix()


# ---------------------------------------------------------------------------
# 10 — C0 vanilla-driver scoring path (organism is None)
# ---------------------------------------------------------------------------


def _probe_and_episode() -> tuple[Probe, Episode]:
    probe = Probe(
        request="What is X?",
        expected="alpha",
        category="retention",
        match_mode="contains",
    )
    episode = Episode(
        id="test-1",
        initial_request="What is X?",
        default_response="I don't know",
        correction_utterance="No, 'X' means alpha.",
        corrected_label="alpha",
        corrected_response="X means alpha",
        domain="test",
        probes=(probe,),
    )
    return probe, episode


def test_score_probe_c0_vanilla_driver_returns_bool(driver: HFDriver) -> None:
    """_score_probe with organism=None scores via the vanilla driver and
    returns a boolean match verdict."""
    probe, episode = _probe_and_episode()
    result = _score_probe(None, probe, episode, vanilla_driver=driver)
    assert isinstance(result, bool)


def test_score_probe_c0_requires_vanilla_driver(driver: HFDriver) -> None:
    """The C0 path guards against a missing vanilla driver."""
    probe, episode = _probe_and_episode()
    with pytest.raises(AssertionError):
        _score_probe(None, probe, episode, vanilla_driver=None)
