"""Trivial baseline agents that satisfy the benchmark interface."""

from __future__ import annotations

from .dataset import build_answer_registry


class AlwaysWrongAgent:
    """A baseline that never learns from correction.

    It returns the same constant answer for every request. This lets the
    benchmark verify that the scorer awards zero transfer/forgetting scores
    while keeping persistent memory at zero.
    """

    def __init__(self, constant_answer: str = "I don't know.") -> None:
        self.constant_answer = constant_answer

    def answer(self, _user_request: str) -> str:
        return self.constant_answer

    def correct(self, _correction: str, _expected: str) -> None:
        """Ignored: this agent never updates."""
        return None

    def persistent_memory(self) -> bytes:
        return b""


class OracleAgent:
    """A baseline that knows the corrected answer for every dataset request.

    This validates that a perfect agent receives a perfect score card.
    """

    def __init__(self) -> None:
        self._map = build_answer_registry()

    def answer(self, user_request: str) -> str:
        return self._map.get(user_request, "I don't know.")

    def correct(self, _correction: str, _expected: str) -> None:
        """Ignored: the oracle already knows every expected answer."""
        return None

    def persistent_memory(self) -> bytes:
        """Serialize the oracle lookup table as persistent memory."""
        return str(self._map).encode("utf-8")
