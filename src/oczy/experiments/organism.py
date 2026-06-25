"""Full-stack Oczy organism agent.

Wires six workspace organs together into a single agent that can answer,
learn from corrections, consolidate raw traces, and report approximate memory
use.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oczy.experiments.cortex_agent import CortexAgent

from experience_autoencoder import ExperienceAutoencoder
from identity_hypernetwork import IdentityHypernetwork
from neural_hippocampus import NeuralHippocampus
from oczy.common import extract_expected_from_correction, tokenize
from oczy.experiments.profiler import AgentProfiler
from plastic_cortex import PlasticCortex
from skill_immune_cortex import SkillImmuneCortex
from world_model_critic import WorldModelCritic

# ``plastic_cortex.lm_cortex`` pulls in numba and a heavier compiled stack.
# Let ImportError pass through (the LM backend simply isn't available on
# stripped-down installs); any other error indicates a real problem and
# should surface, not be silenced.  (Previously this was ``except Exception``
# which masked AttributeError/typos/SyntaxError in the lm_cortex module.)
try:
    from plastic_cortex.lm_cortex import LMPlasticCortex
except ImportError:
    LMPlasticCortex = None

class OrganismAgent:
    """End-to-end plastic-world-model agent.

    The six sub-modules are wired together without changing their internal
    implementations; this class only orchestrates their public APIs.

    For the LM-backend variant (which uses an :class:`LMPlasticCortex`
    model rather than the small word-association :class:`PlasticCortex`),
    use :class:`LMBackendAgent` --- it is kept as a separate class so the
    fast-weight/critic/hippocampus replay pipeline cannot be silently
    disabled by a config flag.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        config = self.config

        if config.get("backend") == "lm":
            raise ValueError(
                "OrganismAgent does not accept backend='lm' directly; "
                "use LMBackendAgent so the LM-only pipeline is visible "
                "in the type rather than a runtime config flag."
            )
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
        self.use_cortex_lm_answer = bool(config.get("use_cortex_lm_answer", False))
        self.use_cortex_policy = bool(config.get("use_cortex_policy", False))
        self.use_value_baseline = bool(config.get("use_value_baseline", False))
        self.cortex_policy_weight = float(config.get("cortex_policy_weight", 1.0))
        self.cortex_agent: CortexAgent | None = config.get("cortex_agent")
        if self.use_cortex_lm_answer and self.cortex_agent is None:
            warnings.warn(
                "use_cortex_lm_answer=True but no cortex_agent provided; "
                "falling back to the legacy PlasticCortex answer path.",
                stacklevel=2,
            )
        if self.use_cortex_policy and self.cortex_agent is None:
            warnings.warn(
                "use_cortex_policy=True but no cortex_agent provided; "
                "falling back to the legacy ranking path.",
                stacklevel=2,
            )
        if self.use_value_baseline and self.cortex_agent is None:
            warnings.warn(
                "use_value_baseline=True but no cortex_agent provided; "
                "policy updates will use baseline=0.0.",
                stacklevel=2,
            )

    def answer(self, request: str) -> str:
        """Produce an answer using the full organ stack."""
        if self.use_cortex_lm_answer and self.cortex_agent is not None:
            return self.cortex_agent.answer(request)["answer"]

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

        # 4. Optional CortexAgent policy-head scoring of candidates.
        policy_scores: dict[str, float] | None = None
        if self.use_cortex_policy and self.cortex_agent is not None:
            try:
                if self.cortex_agent._last_utterance != request:
                    self.cortex_agent.perceive(request)
                raw_scores = self.cortex_agent.policy_score(candidate_answers)
                policy_scores = {
                    cand: float(raw_scores[i])
                    for i, cand in enumerate(candidate_answers)
                    if i < len(raw_scores)
                }
            except Exception:
                policy_scores = None

        # 5. Apply identity-hypernetwork concept-score deltas to rank the
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
            policy_scores=policy_scores,
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
        policy_scores: dict[str, float] | None = None,
    ) -> str:
        """Pick the best candidate by combining request overlap, immune hints,
        identity-adapter concept deltas, and an optional cortex policy head."""
        request_tokens = set(self._tokenize(request))

        def _tokens(text: str) -> set[str]:
            return set(self._tokenize(text))

        def _score(label: str) -> float:
            label_tokens = _tokens(label)
            policy_delta = (
                self.cortex_policy_weight * float(policy_scores.get(label, 0.0))
                if policy_scores is not None
                else 0.0
            )

            # A policy head can still express a preference for a label that
            # happens to have no usable tokens here.
            if not label_tokens:
                return policy_delta

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

            # Optional CortexAgent policy-head score.
            score += policy_delta

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
        # Thin wrapper over the shared glue-layer tokenizer.  Kept as a
        # method so existing call sites (``self._tokenize(...)``) keep
        # working; the shared :func:`oczy_common.tokenize` is the single
        # source of truth for stopword filtering and minimum token length.
        return tokenize(text)

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
        # If no explicit expected answer was given, extract one from the correction text.
        if not expected_answer:
            expected_answer = self._extract_expected_from_correction(correction)

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
                    corrected_answer=expected_answer,
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

            # d. (Optional) Train the CortexAgent policy head from the correction.
            if (
                self.use_cortex_policy
                and self.cortex_agent is not None
                and hasattr(self.cortex_agent, "policy_update")
            ):
                try:
                    labels = list(self.plastic_cortex.labels)
                    if expected_answer and expected_answer not in labels:
                        labels.append(expected_answer)
                    if prior_answer and prior_answer not in labels:
                        labels.insert(0, prior_answer)
                    chosen = prior_answer if prior_answer else labels[0]
                    chosen_idx = labels.index(chosen)
                    baseline = 0.0
                    if self.use_value_baseline:
                        value_hidden = (
                            getattr(self.cortex_agent, "_prev_hidden", None)
                            or getattr(self.cortex_agent, "_last_hidden", None)
                        )
                        if value_hidden is not None and hasattr(
                            self.cortex_agent.world_model_critic, "predict_value"
                        ):
                            baseline = self.cortex_agent.world_model_critic.predict_value(
                                query=request or "",
                                proposed_answer=prior_answer or "",
                                lm_hidden=value_hidden,
                            )
                    self.cortex_agent.policy_update(
                        labels,
                        chosen_idx=chosen_idx,
                        reward=-1.0,
                        baseline=baseline,
                    )
                    # Also reinforce the corrected expected action positively.
                    if expected_answer and expected_answer in labels:
                        expected_idx = labels.index(expected_answer)
                        self.cortex_agent.policy_update(
                            labels,
                            chosen_idx=expected_idx,
                            reward=1.0,
                            baseline=baseline,
                        )
                except Exception:
                    # Policy update is advisory; never break the correction path.
                    pass

    @staticmethod
    def _extract_expected_from_correction(correction: str) -> str:
        """Pull the corrected label out of a free-text correction.

        Delegates to :func:`oczy_common.extract_expected_from_correction`,
        which is the single source of truth shared with all baseline
        ablation agents in ``experiments/baselines.py``.  Previously each
        baseline carried its own byte-identical thin copy while this
        class carried a richer version, and the two silently drifted.
        """
        return extract_expected_from_correction(correction)

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

    def status(self) -> dict[str, Any]:
        """Return a serializable snapshot of the organism state."""
        return {
            "memory_bytes": self.memory_bytes(),
            "profile_summary": self.profile_summary(),
        }

    def reset_state(self) -> None:
        """Reset the plastic cortex session state and scratchpads."""
        self.plastic_cortex.reset_state()
        self._last_request = None
        self._last_answer = None

    def profile_summary(self) -> dict[str, Any]:
        """Return per-component call counts, elapsed time, and peak memory."""
        return self.profiler.summary()

    def save(self, path: Path | str) -> None:
        """Persist the full organism state to *path* using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> OrganismAgent:
        """Load a previously saved organism state."""
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------
    # The ``AgentProfiler`` holds live timing state that is meaningless
    # across a save/load round-trip (timers mid-flight, monotonic-clock
    # references).  Drop it on the way out and rebuild a fresh one on the
    # way back in so an unpickled organism starts with a clean profile.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("profiler", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
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

    @staticmethod
    def _module_bytes(module: Any) -> int:
        """Best-effort byte count for a module.

        Every organ now exposes a standardized ``status()["serialized_bytes"]``
        field (pickle.dumps of the organ at HIGHEST_PROTOCOL).  This is the
        canonical cross-organ byte contract; the previous mixed
        ``trace_bytes`` / ``bytes`` / JSON-length fallbacks are deprecated
        and only kept as last-resort fallbacks for organs that have not
        yet been updated.
        """
        try:
            status = module.status(include_size=True)
        except Exception:
            try:
                status = module.status()
            except Exception:
                status = None


        if isinstance(status, dict):
            # Canonical contract: every organ reports this.
            if "serialized_bytes" in status:
                return int(status["serialized_bytes"])
            # Legacy preferred fields, kept for backwards compat.
            if "trace_bytes" in status:
                return int(status["trace_bytes"])
            if "bytes" in status:
                return int(status["bytes"])
        # Final fallback: pickle the organ directly.  Matches the
        # canonical definition without going through status().
        try:
            return len(pickle.dumps(module, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 0


class LMBackendAgent:
    """LM-backend variant of the Oczy agent.

    Uses an :class:`LMPlasticCortex` (a real, numba-accelerated neural LM)
    as the answer-generating surface instead of :class:`OrganismAgent`'s
    small word-association :class:`PlasticCortex`.  Split out from
    :class:`OrganismAgent` so the LM-only fast path is visible in the
    type system rather than hidden behind a ``backend='lm'`` config flag
    that silently short-circuited the critic/hippocampus/identity pipeline
    in ``answer()`` while still running ``learn()`` against those organs.

    The full organ stack is constructed so that ``learn()`` and
    ``consolidate()`` still write to the hippocampus / critic / immune /
    identity / autoencoder organs --- those counts and slow updates are
    preserved --- but ``answer()`` only consults the LM (plus the immune
    cortex for inline guidance).  Wiring replay back into the LM answer
    path is tracked separately as the LM-replay feature; this class
    only makes the existing half-wired LM path honest about what it does.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if LMPlasticCortex is None:
            raise RuntimeError(
                "LM backend not available: plastic_cortex.lm_cortex failed to import"
            )
        self.config = dict(config or {})
        lm_cfg = self.config.get("lm", {})
        checkpoint = self.config.get(
            "lm_checkpoint", "plastic-cortex/checkpoints/lm/model.pkl"
        )
        if Path(checkpoint).exists():
            self.plastic_cortex = LMPlasticCortex.load(checkpoint)
        else:
            self.plastic_cortex = LMPlasticCortex(lm_cfg)

        # The slow-path organs are constructed even though ``answer()``
        # doesn't consult them yet, so that ``learn()`` writes survive
        # into the same structures an :class:`OrganismAgent` would read.
        self.neural_hippocampus = NeuralHippocampus(self.config.get("neural_hippocampus"))
        self.world_model_critic = WorldModelCritic(self.config.get("world_model_critic"))
        self.identity_hypernetwork = IdentityHypernetwork(
            **(self.config.get("identity_hypernetwork") or {})
        )
        self.skill_immune_cortex = SkillImmuneCortex(self.config.get("skill_immune_cortex"))
        self.experience_autoencoder = ExperienceAutoencoder(
            self.config.get("experience_autoencoder")
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
        self._surprise_threshold = float(self.config.get("surprise_threshold", 0.5))

    def answer(self, request: str) -> str:
        """LM-backend answer path: immune check, then LM generation."""
        with self.profiler.profile("skill_immune_cortex"):
            immune_responses = self.skill_immune_cortex.check(request, "")
        if immune_responses:
            meta = "[immune] " + " ".join(immune_responses)
            request_with_meta = f"{meta} {request}"
        else:
            request_with_meta = request
        with self.profiler.profile("plastic_cortex"):
            lm_answer = self.plastic_cortex.answer(
                request_with_meta, max_tokens=100, temperature=1.0
            )
        self._last_request = request
        self._last_answer = lm_answer
        return lm_answer

    def correct(self, correction: str, expected_answer: str) -> None:
        """Learn from an explicit correction (same shape as OrganismAgent)."""
        return self._learn_from_correction(
            request=self._last_request,
            correction=correction,
            expected_answer=expected_answer,
        )

    def learn(self, request: str, correction: str) -> None:
        """Eval-suite compatible learning hook."""
        expected_answer = extract_expected_from_correction(correction)
        self._last_request = request
        with self.profiler.profile("plastic_cortex"):
            prior_answer = self.plastic_cortex.answer(request, max_tokens=100, temperature=1.0)
        self._last_answer = prior_answer
        self._learn_from_correction(request, correction, expected_answer)

    def _learn_from_correction(
        self,
        request: str | None,
        correction: str,
        expected_answer: str,
    ) -> None:
        prior_answer = self._last_answer or ""
        with self.profiler.profile("world_model_critic"):
            pred = self.world_model_critic.predict_acceptance(
                query=request, proposed_answer=prior_answer
            )
        prediction_error = float(pred.get("accepted_prob", 0.0))
        if not expected_answer:
            expected_answer = extract_expected_from_correction(correction)

        with self.profiler.profile("world_model_critic"):
            self.world_model_critic.record_outcome(
                query=request, proposed_answer=prior_answer, correction=correction
            )

        if prediction_error > self._surprise_threshold:
            with self.profiler.profile("neural_hippocampus"):
                self.neural_hippocampus.store(
                    query=request,
                    answer=prior_answer,
                    correction=correction,
                    prediction_error=prediction_error,
                    corrected_answer=expected_answer,
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

    def consolidate(self) -> None:
        with self.profiler.profile("neural_hippocampus"):
            self.neural_hippocampus.consolidate()
        self._last_request = None
        self._last_answer = None

    def memory_bytes(self) -> int:
        return OrganismAgent._module_bytes(self.plastic_cortex) + sum(
            OrganismAgent._module_bytes(m)
            for m in (
                self.neural_hippocampus,
                self.world_model_critic,
                self.identity_hypernetwork,
                self.skill_immune_cortex,
                self.experience_autoencoder,
            )
        )

    def status(self) -> dict[str, Any]:
        return {
            "backend": "lm",
            "memory_bytes": self.memory_bytes(),
            "profile_summary": self.profiler.summary(),
        }

    def profile_summary(self) -> dict[str, Any]:
        return self.profiler.summary()
