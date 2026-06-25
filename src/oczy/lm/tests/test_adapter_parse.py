"""Unit tests for LanguageAdapter parse prompt and fallback logic.

Does not require llama_cpp; injects a mock LLM into ``LanguageAdapter._llm``.
"""

from __future__ import annotations

import json
from oczy.lm.adapter import LanguageAdapter



class _MockLLM:
    """Returns a fixed string on create_chat_completion."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def create_chat_completion(self, **kwargs) -> dict:
        self.last_messages = kwargs.get("messages")
        return {"choices": [{"message": {"content": self.reply}}]}


def _make_adapter(reply: str) -> LanguageAdapter:
    adapter = LanguageAdapter()
    adapter._llm = _MockLLM(reply)
    adapter._loaded = True
    return adapter


def test_plain_query_returns_accepted_empty_correction() -> None:
    reply = json.dumps(
        {
            "query": "What is the weather today?",
            "answer": "",
            "correction": "",
            "corrected_answer": "",
            "outcome": "accepted",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("What is the weather today?")
    assert ep["outcome"] == "accepted"
    assert ep["corrected_answer"] == ""
    assert ep["correction"] == ""


def test_correction_extracts_y_part() -> None:
    reply = json.dumps(
        {
            "query": "Schedule the batch.",
            "answer": "",
            "correction": "No, 'batch' here means ML training batch.",
            "corrected_answer": "ML training batch",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("Schedule the batch. No, 'batch' here means ML training batch.")
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "ML training batch"


def test_sanity_check_clears_spurious_corrected_answer() -> None:
    """LM sometimes answers the factual question and puts the answer in corrected_answer."""
    reply = json.dumps(
        {
            "query": "What is the capital of France?",
            "answer": "",
            "correction": "",
            "corrected_answer": "Paris",
            "outcome": "accepted",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("What is the capital of France?")
    assert ep["outcome"] == "accepted"
    assert ep["corrected_answer"] == ""


def test_sanity_check_downgrades_missing_corrected_answer() -> None:
    reply = json.dumps(
        {
            "query": "Update the profile.",
            "answer": "",
            "correction": "No, 'profile' means business vertical.",
            "corrected_answer": "",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("Update the profile. No, 'profile' means business vertical.")
    # corrected_answer is empty, so outcome is downgraded to accepted.
    assert ep["outcome"] == "accepted"


def test_malformed_json_falls_back_to_minimal_episode() -> None:
    adapter = _make_adapter("not json")
    ep = adapter.nl_to_episode("Do something.")
    assert ep["outcome"] == "accepted"
    assert ep["query"] == "Do something."



def test_short_correction_model() -> None:
    reply = json.dumps(
        {
            "query": "",
            "answer": "",
            "correction": "No, 'model' means ML model.",
            "corrected_answer": "ML model",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("No, 'model' means ML model.")
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "ML model"


def test_short_correction_run() -> None:
    reply = json.dumps(
        {
            "query": "",
            "answer": "",
            "correction": "'run' means pipeline run.",
            "corrected_answer": "pipeline run",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("'run' means pipeline run.")
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "pipeline run"


def test_clean_corrected_answer_strips_redefined_word() -> None:
    """If the LM returns the redefined word inside corrected_answer, strip it."""
    reply = json.dumps(
        {
            "query": "Update the profile.",
            "answer": "",
            "correction": "No, 'profile' means business vertical.",
            "corrected_answer": "profile means business vertical",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode(
        "Update the profile. No, 'profile' means business vertical."
    )
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "business vertical"


def test_upgrade_accepted_when_correction_wording_present() -> None:
    """An accepted parse that still carries a Y value should be upgraded."""
    reply = json.dumps(
        {
            "query": "Update the profile.",
            "answer": "",
            "correction": "",
            "corrected_answer": "business vertical",
            "outcome": "accepted",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode(
        "Update the profile. No, 'profile' means business vertical."
    )
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "business vertical"


def test_extract_meaning_fallback_for_empty_corrected_answer() -> None:
    """Standalone corrections missing corrected_answer are recovered from correction."""
    reply = json.dumps(
        {
            "query": "",
            "answer": "",
            "correction": "No, 'key' means API key.",
            "corrected_answer": "",
            "outcome": "corrected",
            "source": "user_utterance",
        }
    )
    adapter = _make_adapter(reply)
    ep = adapter.nl_to_episode("No, 'key' means API key.")
    assert ep["outcome"] == "corrected"
    assert ep["corrected_answer"] == "API key"

if __name__ == "__main__":
    test_plain_query_returns_accepted_empty_correction()
    test_correction_extracts_y_part()
    test_sanity_check_clears_spurious_corrected_answer()
    test_sanity_check_downgrades_missing_corrected_answer()
    test_malformed_json_falls_back_to_minimal_episode()
    test_short_correction_model()
    test_short_correction_run()
    test_clean_corrected_answer_strips_redefined_word()
    test_upgrade_accepted_when_correction_wording_present()
    test_extract_meaning_fallback_for_empty_corrected_answer()
    print("adapter parse tests passed")
