"""Scoring logic for the Correction-to-Competence Benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataset import Episode, Probe


def normalize(text: str) -> str:
    """Forgiving normalization for text comparison."""
    return " ".join(text.lower().split())


def matches(answer: str, expected: str) -> bool:
    """Check whether ``answer`` matches ``expected`` after normalization."""
    return normalize(answer) == normalize(expected)


@dataclass(frozen=True)
class ProbeResult:
    probe: "Probe"
    answer: str
    correct: bool


@dataclass(frozen=True)
class EpisodeResult:
    episode: "Episode"
    initial_answer: str
    post_correction_answer: str
    probe_results: tuple[ProbeResult, ...]

    def fixed_immediately(self) -> bool:
        return matches(self.post_correction_answer, self.episode.corrected_answer)


class Scorer:
    """Compute Correction-to-Competence metrics."""

    @staticmethod
    def correction_uptake_latency(results: tuple[EpisodeResult, ...]) -> float:
        """Fraction of episodes that still need another turn after one correction.

        0.0 means every correction was fixed immediately; 1.0 means none were.
        """
        if not results:
            return 0.0
        return 1.0 - sum(r.fixed_immediately() for r in results) / len(results)

    @staticmethod
    def _score_by_category(results: tuple[EpisodeResult, ...], category: str) -> float:
        total = 0
        correct = 0
        for result in results:
            for pr in result.probe_results:
                if pr.probe.category == category:
                    total += 1
                    if pr.correct:
                        correct += 1
        if total == 0:
            return 1.0
        return correct / total

    @classmethod
    def transfer_score(cls, results: tuple[EpisodeResult, ...]) -> float:
        """Correctness rate on transfer probes."""
        return cls._score_by_category(results, "transfer")

    @classmethod
    def scope_score(cls, results: tuple[EpisodeResult, ...]) -> float:
        """Correctness rate on scope (anti-overgeneralization) probes."""
        return cls._score_by_category(results, "scope")

    @classmethod
    def forgetting_score(cls, results: tuple[EpisodeResult, ...]) -> float:
        """Correctness rate on unrelated forgetting probes."""
        return cls._score_by_category(results, "forgetting")

    @staticmethod
    def _memory_bytes(agent: Any) -> int:
        """Return the number of persistent bytes reported by ``agent``.

        Agents can expose their persistent memory by implementing one of:

        * ``persistent_memory()`` -> bytes
        * ``persistent_memory`` -> bytes  # attribute
        * ``memory_size()`` -> int
        """
        mem = getattr(agent, "persistent_memory", None)
        if callable(mem):
            mem = mem()
        if isinstance(mem, bytes):
            return len(mem)
        if isinstance(mem, str):
            return len(mem.encode("utf-8"))
        size_fn = getattr(agent, "memory_size", None)
        if callable(size_fn):
            return size_fn()
        return 0

    @classmethod
    def memory_bytes_per_delta(
        cls,
        agent: Any,
        results: tuple[EpisodeResult, ...],
    ) -> float:
        """Persistent memory bytes divided by the number of distinct learned lessons.

        A "lesson" is counted when the agent actually changed its answer after
        one correction. Lower is better.
        """
        memory_bytes = cls._memory_bytes(agent)
        distinct_deltas = sum(1 for r in results if r.fixed_immediately())
        if distinct_deltas == 0:
            return 0.0 if memory_bytes == 0 else math.inf
        return memory_bytes / distinct_deltas

    @classmethod
    def score(
        cls,
        results: tuple[EpisodeResult, ...],
        agent: Any,
    ) -> dict[str, float]:
        """Return the full score card for a completed benchmark run."""
        return {
            "correction_uptake_latency": cls.correction_uptake_latency(results),
            "transfer_score": cls.transfer_score(results),
            "scope_score": cls.scope_score(results),
            "forgetting_score": cls.forgetting_score(results),
            "memory_bytes_per_delta": cls.memory_bytes_per_delta(agent, results),
        }
