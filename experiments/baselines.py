"""Baseline and ablation agents for the Oczy organism curriculum.

Each agent exposes:

- ``answer(request: str) -> str``
- ``correct(correction: str, expected_answer: str) -> None``
- ``consolidate() -> None``
- ``memory_bytes() -> int``

They are deliberately minimal and serve as controls against the full-stack
``OrganismAgent``.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from experiments.profiler import AgentProfiler

from plastic_cortex import PlasticCortex
from neural_hippocampus import NeuralHippocampus
from identity_hypernetwork import IdentityHypernetwork
class ZeroMemoryAgent:
    """Ignores all corrections and always emits the same default wrong answer."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._answer: str = (config or {}).get("answer", "default wrong answer")
        self.profiler = AgentProfiler([])

    def answer(self, _request: str) -> str:
        return self._answer

    def correct(self, _correction: str, _expected_answer: str) -> None:
        """Corrections are ignored by design."""
        return None

    def learn(self, _request: str, _correction: str) -> None:
        """Eval-suite compatible no-op learning hook."""
        return None

    def consolidate(self) -> None:
        return None

    def memory_bytes(self) -> int:
        return sys.getsizeof(self._answer)

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()


class ContextOnlyAgent:
    """Remembers raw correction strings and answers by substring matching."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._context: list[dict[str, str]] = []
        self._default: str = (config or {}).get("answer", "I don't know.")
        self.profiler = AgentProfiler([])

    def answer(self, request: str) -> str:
        lowered = request.lower()
        # Most recent matching correction wins.
        for entry in reversed(self._context):
            stored_request = entry.get("request", "").lower()
            correction = entry.get("correction", "").lower()
            if stored_request in lowered or lowered in stored_request:
                return entry.get("correction", self._default)
            # Also allow the correction text itself to be a trigger.
            if correction and correction in lowered:
                return entry.get("expected", self._default)
        return self._default

    def correct(self, correction: str, expected_answer: str) -> None:
        self._context.append(
            {"correction": correction, "expected": expected_answer, "request": ""}
        )

    def learn(self, request: str, correction: str) -> None:
        self._context.append(
            {
                "request": request,
                "correction": correction,
                "expected": self._extract_expected_from_correction(correction),
            }
        )

    def consolidate(self) -> None:
        """Raw-context agent has no slow consolidation step."""
        return None

    def memory_bytes(self) -> int:
        total = sys.getsizeof(self._context)
        for entry in self._context:
            total += sum(sys.getsizeof(v) for v in entry.values())
        return total

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        text = correction.lower()
        for marker in ("means ", "is ", "refers to ", "should be ", "use "):
            idx = text.find(marker)
            if idx != -1:
                candidate = correction[idx + len(marker) :].strip().strip(".'\"")
                if candidate:
                    return candidate
        return correction


class FastOnlyAgent:
    """Wraps only ``PlasticCortex``; fast weights get updated on each correction."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.cortex = PlasticCortex(config)
        self.profiler = AgentProfiler(["plastic_cortex"])

    def answer(self, request: str) -> str:
        with self.profiler.profile("plastic_cortex"):
            return self.cortex.answer(request)

    def correct(self, correction: str, expected_answer: str) -> None:
        with self.profiler.profile("plastic_cortex"):
            self.cortex.correct(correction, expected_answer)

    def learn(self, request: str, correction: str) -> None:
        expected = self._extract_expected_from_correction(correction)
        # Make sure the request has entered the recurrent context first.
        with self.profiler.profile("plastic_cortex"):
            self.cortex.answer(request)
            self.cortex.correct(correction, expected)

    def consolidate(self) -> None:
        """Fast weights are retained; no separate consolidation step."""
        return None

    def memory_bytes(self) -> int:
        return sys.getsizeof(self.cortex)

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        text = correction.lower()
        for marker in ("means ", "is ", "refers to ", "should be ", "use "):
            idx = text.find(marker)
            if idx != -1:
                candidate = correction[idx + len(marker) :].strip().strip(".'\"")
                if candidate:
                    return candidate
        return correction


class HippocampusOnlyAgent:
    """Stores high-surprise episodes in ``NeuralHippocampus`` and answers from
    the most relevant stored episode.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.memory = NeuralHippocampus(config.get("neural_hippocampus"))
        self._default: str = config.get("answer", "I don't know.")
        self._last_request: str | None = None
        self._last_answer: str | None = None
        self._surprise_threshold: float = float(
            config.get("surprise_threshold", 0.5)
        )
        self.profiler = AgentProfiler(["neural_hippocampus"])

    def answer(self, request: str) -> str:
        with self.profiler.profile("neural_hippocampus"):
            replays = self.memory.reinforce(query=request, k=1)
        if replays:
            corrected = replays[0].get("corrected_answer")
            if corrected:
                return corrected
        # If nothing is in memory, use the default placeholder.
        return self._default

    def correct(self, correction: str, expected_answer: str) -> None:
        request = self._last_request or ""
        answer = self._last_answer or ""
        prediction_error = 1.0  # Assume maximal surprise in this ablation.
        if prediction_error > self._surprise_threshold:
            with self.profiler.profile("neural_hippocampus"):
                self.memory.store(
                    query=request,
                    answer=answer,
                    correction=correction,
                    prediction_error=prediction_error,
                )
                # Also remember the corrected answer so answer() can return it later.
                episode = {
                    "query": request,
                    "answer": answer,
                    "correction": correction,
                    "prediction_error": prediction_error,
                    "corrected_answer": expected_answer,
                }
                self.memory.memory.write(episode)

    def learn(self, request: str, correction: str) -> None:
        self._last_request = request
        self._last_answer = self.answer(request)
        expected = self._extract_expected_from_correction(correction)
        self.correct(correction, expected)

    def consolidate(self) -> None:
        with self.profiler.profile("neural_hippocampus"):
            self.memory.consolidate()

    def memory_bytes(self) -> int:
        return int(self.memory.status().get("trace_bytes", 0))

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        text = correction.lower()
        for marker in ("means ", "is ", "refers to ", "should be ", "use "):
            idx = text.find(marker)
            if idx != -1:
                candidate = correction[idx + len(marker) :].strip().strip(".'\"")
                if candidate:
                    return candidate
        return correction


class IdentityOnlyAgent:
    """Learns via ``IdentityHypernetwork`` updates and answers by ranking known
    concepts using adapter deltas.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.identity = IdentityHypernetwork(
            **(config.get("identity_hypernetwork") or {})
        )
        self._default: str = config.get("answer", "I don't know.")
        self.profiler = AgentProfiler(["identity_hypernetwork"])

    def answer(self, request: str) -> str:
        with self.profiler.profile("identity_hypernetwork"):
            adapters = self.identity.generate_adapters()
        concept_scores = adapters.get("concept_scores", {})
        request_tokens = set(_tokenize(request))

        best_concept: str | None = None
        best_score = float("-inf")
        for concept, delta in concept_scores.items():
            score = float(delta)
            if concept in request_tokens:
                score += 1.0
            if score > best_score:
                best_score = score
                best_concept = concept
        return best_concept if best_concept is not None else self._default

    def correct(self, correction: str, expected_answer: str) -> None:
        with self.profiler.profile("identity_hypernetwork"):
            self.identity.update_identity(
                {
                    "source": "user_correction",
                    "correct_label": expected_answer,
                    "token": expected_answer,
                }
            )

    def learn(self, request: str, correction: str) -> None:
        expected = self._extract_expected_from_correction(correction)
        with self.profiler.profile("identity_hypernetwork"):
            self.identity.update_identity(
                {
                    "source": "user_correction",
                    "correct_label": expected,
                    "token": expected,
                }
            )

    def consolidate(self) -> None:
        """Identity is a slow latent; nothing to drop."""
        return None

    def memory_bytes(self) -> int:
        status = self.identity.status()
        return len(str(status).encode("utf-8"))

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        text = correction.lower()
        for marker in ("means ", "is ", "refers to ", "should be ", "use "):
            idx = text.find(marker)
            if idx != -1:
                candidate = correction[idx + len(marker) :].strip().strip(".'\"")
                if candidate:
                    return candidate
        return correction


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]
