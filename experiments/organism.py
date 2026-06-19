"""Full-stack Oczy organism agent.

Wires six workspace organs together into a single agent that can answer,
learn from corrections, consolidate raw traces, and report approximate memory
use.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from experiments.profiler import AgentProfiler

from plastic_cortex import PlasticCortex
from neural_hippocampus import NeuralHippocampus
from world_model_critic import WorldModelCritic
from identity_hypernetwork import IdentityHypernetwork
from skill_immune_cortex import SkillImmuneCortex
from experience_autoencoder import ExperienceAutoencoder


class OrganismAgent:
    """End-to-end plastic-world-model agent.

    The six sub-modules are wired together without changing their internal
    implementations; this class only orchestrates their public APIs.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})

        self.plastic_cortex = PlasticCortex(config.get("plastic_cortex"))
        self.neural_hippocampus = NeuralHippocampus(config.get("neural_hippocampus"))
        self.world_model_critic = WorldModelCritic(config.get("world_model_critic"))
        self.identity_hypernetwork = IdentityHypernetwork(
            **(config.get("identity_hypernetwork") or {})
        )
        self.skill_immune_cortex = SkillImmuneCortex(config.get("skill_immune_cortex"))
        self.experience_autoencoder = ExperienceAutoencoder(
            config.get("experience_autoencoder")
        )
        self.profiler = AgentProfiler(
            [
                "plastic_cortex",
                "neural_hippocampus",
                "world_model_critic",
                "identity_hypernetwork",
                "skill_immune_cortex",
                "experience_autoencoder",
            ]
        )

        self._last_request: str | None = None
        self._last_answer: str | None = None
        self._low_confidence_threshold = float(config.get("low_confidence_threshold", 0.6))
        self._high_correction_threshold = float(config.get("high_correction_threshold", 0.4))
        self._surprise_threshold = float(config.get("surprise_threshold", 0.5))
        self._unk = "I don't know."

    def answer(self, request: str) -> str:
        """Produce an answer using the full organ stack."""
        # 1. Immune check: see if any previous mistake detector fires for the raw
        # request.  We do not have a proposed answer yet, so pass an empty one.
        with self.profiler.profile("skill_immune_cortex"):
            immune_responses = self.skill_immune_cortex.check(request, "")
        if immune_responses:
            # Surface immune guidance in-line so downstream modules can see it.
            meta = "[immune] " + " ".join(immune_responses)
            request_with_meta = f"{meta} {request}"
        else:
            request_with_meta = request

        # 2. Fast-weight / recurrent answer.
        with self.profiler.profile("plastic_cortex"):
            fast_answer = self.plastic_cortex.answer(request_with_meta)

        # 3. World-model confidence check.
        with self.profiler.profile("world_model_critic"):
            critic_pred = self.world_model_critic.predict_acceptance(
                query=request, proposed_answer=fast_answer
            )
        accepted_prob = float(critic_pred.get("accepted_prob", 0.0))
        correction_likelihood = float(critic_pred.get("correction_likelihood", 0.0))
        low_confidence = (
            accepted_prob < self._low_confidence_threshold
            or correction_likelihood > self._high_correction_threshold
        )

        candidate_answers = list(self.plastic_cortex.labels)
        replay_hint: str | None = None
        if low_confidence:
            with self.profiler.profile("neural_hippocampus"):
                replays = self.neural_hippocampus.reinforce(query=request, k=3)
            if replays:
                # Try to recover the corrected sense stored in the replay.
                for episode in replays:
                    corrected = episode.get("corrected_answer", "")
                    if corrected:
                        replay_hint = corrected
                        candidate_answers.append(corrected)
                        break

        # 4. Apply identity-hypernetwork concept-score deltas to rank the
        # candidate labels.
        with self.profiler.profile("identity_hypernetwork"):
            adapters = self.identity_hypernetwork.generate_adapters()
        concept_scores = adapters.get("concept_scores", {})
        final_answer = self._rank_answer(
            request=request,
            candidates=candidate_answers,
            fast_answer=fast_answer,
            replay_hint=replay_hint,
            concept_scores=concept_scores,
        )

        self._last_request = request
        self._last_answer = final_answer
        return final_answer

    def _rank_answer(
        self,
        request: str,
        candidates: list[str],
        fast_answer: str,
        replay_hint: str | None,
        concept_scores: dict[str, float],
    ) -> str:
        """Pick the best candidate by combining request overlap, immune hints,
        and identity-adapter concept deltas."""
        request_tokens = set(self._tokenize(request))

        def _tokens(text: str) -> set[str]:
            return set(self._tokenize(text))

        def _score(label: str) -> float:
            label_tokens = _tokens(label)
            if not label_tokens:
                return 0.0

            # Start from strong preference for what the fast organ returned.
            score = 1.0 if label == fast_answer else 0.0
            if replay_hint and label == replay_hint:
                score += 0.5

            # Token-overlap with the request (simple matching signal).
            score += len(request_tokens & label_tokens) / max(len(label_tokens), len(request_tokens), 1)

            # Identity-adapter token boosts: every matching concept token adds
            # its delta to this label.
            for token in label_tokens:
                score += float(concept_scores.get(token, 0.0))

            return score

        best_label = fast_answer
        best_score = _score(best_label)
        for label in candidates:
            if label == fast_answer:
                continue
            s = _score(label)
            if s > best_score or (s == best_score and label == fast_answer):
                best_label = label
                best_score = s
        return best_label
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]

    def correct(self, correction: str, expected_answer: str) -> None:
        """Learn from an explicit correction.

        Also compatible with the eval-suite ``learn`` dispatch: that method
        passes ``(request, correction)`` with no explicit expected answer.  We
        surface a separate ``learn`` adapter for that path and treat the second
        argument as the correction text.
        """
        return self._learn_from_correction(
            request=self._last_request,
            correction=correction,
            expected_answer=expected_answer,
        )

    def learn(self, request: str, correction: str) -> None:
        """Eval-suite compatible learning hook (request + correction only)."""
        # Try to recover an explicit corrected label from the correction text.
        expected_answer = self._extract_expected_from_correction(correction)
        self._last_request = request
        # Need a prior answer to compute critic surprise.
        with self.profiler.profile("plastic_cortex"):
            prior_answer = self.plastic_cortex.answer(request)
        self._last_answer = prior_answer
        self._learn_from_correction(request, correction, expected_answer)

    def _learn_from_correction(
        self,
        request: str | None,
        correction: str,
        expected_answer: str,
    ) -> None:
        if self._last_answer:
            prior_answer = self._last_answer
        else:
            with self.profiler.profile("plastic_cortex"):
                prior_answer = self.plastic_cortex.answer(request)

        # a. Critic surprise / prediction error.
        with self.profiler.profile("world_model_critic"):
            pred = self.world_model_critic.predict_acceptance(
                query=request, proposed_answer=prior_answer
            )
        accepted_prob = float(pred.get("accepted_prob", 0.0))
        # The user corrected us, so the true outcome is "corrected" (accepted=0).
        prediction_error = accepted_prob

        # b. Update the fast-weight organ.
        with self.profiler.profile("plastic_cortex"):
            self.plastic_cortex.correct(correction, expected_answer)

        # Also update the world model with the observed correction.
        with self.profiler.profile("world_model_critic"):
            self.world_model_critic.record_outcome(
                query=request, proposed_answer=prior_answer, correction=correction
            )

        # c. If prediction error is high, store the episode in slow memory and
        # update long-term identity / immune structures.
        if prediction_error > self._surprise_threshold:
            with self.profiler.profile("neural_hippocampus"):
                self.neural_hippocampus.store(
                    query=request,
                    answer=prior_answer,
                    correction=correction,
                    prediction_error=prediction_error,
                )

            episode = {
                "situation": request,
                "model_answer": prior_answer,
                "correction": correction,
                "revised_answer": expected_answer,
                "outcome": "corrected",
                "source": "user_correction",
                "corrected_answer": expected_answer,
            }
            with self.profiler.profile("experience_autoencoder"):
                self.experience_autoencoder.encode(episode)

            with self.profiler.profile("identity_hypernetwork"):
                self.identity_hypernetwork.update_identity(
                    {
                        "source": "user_correction",
                        "correct_label": expected_answer,
                        "token": expected_answer,
                    }
                )

            with self.profiler.profile("skill_immune_cortex"):
                self.skill_immune_cortex.add_detector(
                    correction_text=correction,
                    mistake_class="corrected_sense",
                    response=expected_answer,
                )

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        """Very small heuristic to pull the corrected label out of free text."""
        text = correction.lower()
        for marker in ("means ", "is ", "refers to ", "should be ", "use "):
            idx = text.find(marker)
            if idx != -1:
                candidate = correction[idx + len(marker) :].strip().strip(".'\"")
                if candidate:
                    return candidate
        # Fall back to the whole correction text.
        return correction

    def consolidate(self) -> None:
        """Move hippocampal traces to slow updates and clear raw trace state.

        Identity / hypernetwork adapters are retained; they are the consolidated
        slow knowledge.
        """
        with self.profiler.profile("neural_hippocampus"):
            self.neural_hippocampus.consolidate()
        self._last_request = None
        self._last_answer = None

    def memory_bytes(self) -> int:
        """Approximate total serialized size across all organs."""
        total = 0
        for module in (
            self.plastic_cortex,
            self.neural_hippocampus,
            self.world_model_critic,
            self.identity_hypernetwork,
            self.skill_immune_cortex,
            self.experience_autoencoder,
        ):
            total += self._module_bytes(module)

        return total

    def profile_summary(self) -> dict[str, Any]:
        """Return per-component call counts, elapsed time, and peak memory."""
        return self.profiler.summary()

    @staticmethod
    def _module_bytes(module: Any) -> int:
        """Best-effort byte count for a module.

        Uses explicit ``status()`` byte fields when available, otherwise falls
        back to the size of a JSON serialization.
        """
        try:
            status = module.status()
        except Exception:
            status = None

        if isinstance(status, dict):
            if "trace_bytes" in status:
                return int(status["trace_bytes"])
            if "bytes" in status:
                return int(status["bytes"])
            # No explicit byte field: use serialized status as a proxy.
            return len(json.dumps(status, default=str).encode("utf-8"))

        # Critic has no status(); estimate from its attributes.
        try:
            payload = json.dumps(module.__dict__, default=str)
        except Exception:
            payload = ""
        return max(len(payload.encode("utf-8")), sys.getsizeof(module))
