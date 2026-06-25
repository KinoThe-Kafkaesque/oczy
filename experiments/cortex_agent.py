"""CortexAgent: the un-inverted organism.

Wires ``KVCortex`` (live mutation surface) and ``LlamaCVecDriver`` (frozen
LM articulator) together with the existing organ metabolism. The cortex
is the centre: it observes LM hidden states on perceive(), mutates its
warm_state, and emits per-layer cvecs that the driver injects into the
LM's forward pass on articulate(). The LM never learns; every mutation
lives in the cortex.

This is Goal 3 from ``GOALS.md``: build the CortexAgent driver glue
between cortex, driver, and organ metabolism.

Lifecycle per turn:

    perceive(utterance, correction_signal) -> hidden -> cortex.observe
    metabolize()  -> fan organ metabolism off cortex.state
                    (critic reads drift; hippocampus stores hidden;
                     immune registers; autoencoder trains proj_hidden)
    articulate()  -> cortex.emit_all_cvecs() -> driver.set_cvecs_per_layer
                                                 -> driver.generate()
                    (cortex steers the LM; LM is frozen)
    (harness) consolidate() -> cortex.consolidate(hippocampus replays)
                               + organ consolidate() fans out
"""

from __future__ import annotations

import dataclasses
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _name in (
    "plastic-cortex",
    "neural-hippocampus",
    "world-model-critic",
    "identity-hypernetwork",
    "skill-immune-cortex",
    "experience-autoencoder",
):
    _src = _REPO_ROOT / _name / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import numpy as np

from neural_hippocampus import NeuralHippocampus
from world_model_critic import WorldModelCritic
from experience_autoencoder import ExperienceAutoencoder
from identity_hypernetwork import IdentityHypernetwork
from skill_immune_cortex import SkillImmuneCortex
from experience_autoencoder.autoencoder import HEBBIAN_LR

from experiments.digestive_gate import DigestiveGate, DigestiveGateConfig


from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig
from oczy_lm import CVecDriverConfig, LlamaCVecDriver

from experiments.codebase_qa.knowledge_store import KnowledgeStore

# Heuristic correction-signal detector. The cortex's neuromodulator needs
# to know when to fire high plasticity; this is a stop-gap that will be
# replaced by the WorldModelCritic's drift-based signal once Goal 3 fully
# converts the critic to a tensor-input consumer.
_CORRECTION_MARKERS = (
    "no, ", "no:", "wrong, ", "wrong:", "correction:",
    "correct:", "expected:", "not what i meant",
    "i meant", "actually,", "rather than",
)


def _looks_like_correction(text: str) -> bool:
    """Cheap lexical detector for correction_signal driving."""
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _CORRECTION_MARKERS)


@dataclass
class CortexAgentConfig:
    """Sizes + LM loading settings for CortexAgent.

    d_cortex defaults to 64 (small enough that the cvec projector memory
    stays at ~2 MB for a 16-attention-layer / 2048-embd model) while
    still being expressive enough to host a few distinct intent basins.
    """

    cortex: KVCortexConfig | None = None
    driver: CVecDriverConfig | None = None

    # Cvec amplitude applied at articulate() time. Determined empirically
    # on LFM2.5-1.2B-Instruct Q4_K_M with steering_mode="raw_hidden":
    #   scale < 0.0005  -> no effect on greedy decoding
    #   scale 0.001     -> clean steering, output shifts toward correction
    #   scale >= 0.005  -> off-manifold, falls into token-repetition garbage
    # The amplifier is per-LM-residual-norm dependent: if the host LM or
    # quant changes, re-sweep the scale before trusting this default.

    # Drift threshold above which metabolize() considers the correction
    # strong enough to force the WorldModelCritic's correction path even
    # without a textual correction marker.
    correction_drift_threshold: float = 0.05

    # Optional scalar metabolic gate. When None, a backward-compatible
    # default is derived from correction_drift_threshold so existing
    # behavior is preserved.
    digestive_gate: DigestiveGateConfig | None = None

    # If True, turn() will call consolidate() automatically when the
    # digestive gate reports pressure above threshold.
    auto_consolidate: bool = True

    articulate_scale: float = 0.001


class CortexAgent:
    """Un-inverted organism: cortex mutates, LM articulates, organs metabolise.

    The agent OWNS one driver (one frozen LM, one cvec adapter slot set).
    Separate agents require separate drivers because ``llama_set_adapter_cvec``
    writes per-context cortex state.
    """

    def __init__(
        self,
        config: CortexAgentConfig | None = None,
        knowledge_store: KnowledgeStore | None = None,
        driver: LlamaCVecDriver | None = None,
    ) -> None:
        self.config = config or CortexAgentConfig()
        ccfg = self.config.cortex or KVCortexConfig()
        dcfg = self.config.driver or CVecDriverConfig()

        # Cortex must mirror driver shape: d_embd and n_layers come from
        # the LM, not from config-only defaults. We instantiate the
        # driver first to know its actual n_layers, then size the cortex.
        # CortexAgent is constructed with whatever KVCortexConfig the
        # caller passed -- but if d_embd or n_layers disagree with the
        # actual LM, the first articulate() will raise a shape mismatch.
        # To keep that contract honest we patch the cortex config here.
        # When a driver is supplied we reuse it to avoid duplicate LM loads.
        self.driver = driver if driver is not None else LlamaCVecDriver.load(dcfg)
        # Cortex must mirror driver shape: d_embd and n_layers come from
        # the LM, not from config-only defaults. We instantiate the
        # driver first to know its actual n_layers, then size the cortex.
        # CortexAgent is constructed with whatever KVCortexConfig the
        # caller passed -- but if d_embd or n_layers disagree with the
        # actual LM, the first articulate() will raise a shape mismatch.
        # To keep that contract honest we patch the cortex config here.
        # Mirror driver shape while preserving every caller-set field
        # (steering_mode especially -- dropping it silently reverts the
        # cortex to proj_random and the raw_hidden regime never engages).
        patched = dataclasses.replace(
            ccfg,
            d_embd=self.driver.n_embd,
            n_layers=self.driver.n_layers,
        )
        self.cortex = KVCortex(patched)

        # Existing organ metabolism. The cortex drives them; they don't
        # drive the cortex (no string-fed fast-weight replacement of the
        # cortex's intent).
        self.neural_hippocampus = NeuralHippocampus()
        self.world_model_critic = WorldModelCritic()
        self.identity_hypernetwork = IdentityHypernetwork()
        self.skill_immune_cortex = SkillImmuneCortex()
        self.experience_autoencoder = ExperienceAutoencoder()

        # Scalar metabolic gate sits downstream of all organs and decides
        # per-organ update weights plus consolidation pressure. When no
        # config is supplied, derive a backward-compatible default from
        # the agent's correction_drift_threshold.
        dg_cfg = self.config.digestive_gate or DigestiveGateConfig(
            novelty_threshold=self.config.correction_drift_threshold,
        )
        self.digestive_gate = DigestiveGate(config=dg_cfg)

        # Optional codebase knowledge store; recalled facts can be injected
        # into prompts during articulate() to ground the agent in repo facts.
        self.knowledge_store = knowledge_store

        self._last_utterance: str | None = None
        self._last_hidden: np.ndarray | None = None
        self._last_correction_signal: float = 0.0
        self._last_drift: float = 0.0

    # ------------------------------------------------------------------
    # Boot / cold path
    # ------------------------------------------------------------------
    def boot(self) -> None:
        """Cold boot: warm_state := cold_state.copy().

        Call once after construction (or after a long idle / topic change)
        so the cortexes starts a session from its persisted identity rather
        than the empty default that __init__ left it in.
        """
        self.cortex.reset_warm_from_cold()
        self.digestive_gate.reset()
        self._last_utterance = None
        self._last_hidden = None
        self._last_correction_signal = 0.0
        self._last_drift = 0.0

    def should_consolidate(self) -> bool:
        """Return True when the digestive gate says consolidation pressure
        has crossed the configured threshold.
        """
        return self.digestive_gate.should_consolidate()

    # ------------------------------------------------------------------
    # Warm path
    # ------------------------------------------------------------------
    def perceive(
        self,
        utterance: str,
        correction_signal: float | None = None,
    ) -> np.ndarray:
        """Feed an utterance through the LM's perception side into the cortex.

        Args:
            utterance: the user's NL input.
            correction_signal: optional explicit gate in [0, 1]. If None
                (default), _looks_like_correction() provides a binary
                proxy from lexical markers. Pass an explicit value when
                the caller has independent signals (e.g., from the
                WorldModelCritic's drift above threshold).

        Returns:
            The cortex's updated warm_state (ndarray, d_cortex,). Most
            callers ignore the return and call articulate() next.
        """
        if correction_signal is None:
            correction_signal = 1.0 if _looks_like_correction(utterance) else 0.0

        # Driver.peek_embedding returns the model's final-layer summary of
        # the prompt -- a (n_embd,) float32. This is Goal 2 staging; once
        # layer-L intermediate extraction is wired in, peek_layer(L)
        # replaces this call and the cortex sees deeper hidden signal.
        hidden = self.driver.peek_embedding(utterance, last_token_only=False)

        warm_before = self.cortex.warm_state.copy()
        warm_now = self.cortex.observe(hidden, correction_signal=correction_signal)
        drift = float(np.linalg.norm(warm_now - warm_before))

        self._last_utterance = utterance
        self._last_hidden = hidden
        self._last_correction_signal = correction_signal
        self._last_drift = drift

        return warm_now

    def metabolize(self, utterance: str | None = None) -> dict[str, Any]:
        """Fan organ metabolism off the cortex's current warm_state.

        This is the cortex-driven adaption path. The organs consume:

          * Cortex state norms (drift, warm/cold) -- as the surprise signal
            that the WorldModelCritic's `_last_correction_prob` proxy
            used to derive from string features. We attach the drift scalar
            to the critic's last-correction-prob so its predict_acceptance
            calls see a cortex-derived signal on subsequent turns.
          * The hidden vector the cortex absorbed -- stored in the
            hippocampus as a high-surprise episode keyed by the utterance.
            Replay becomes a tensor bank instead of a string-keyed store.
          * The autoencoder takes (utterance, hidden) as a learning step
            on the cortex's perception projector.

        Returns a status dict for inspection; the call is otherwise run
        for side effects on the cortex and organs.
        """
        text = utterance if utterance is not None else (self._last_utterance or "")
        hidden = self._last_hidden
        if hidden is None:
            # No perceive() has run yet -- nothing to metabolise.
            return {"metabolized": False, "reason": "no hidden cached"}

        correction_signal = self._last_correction_signal

        # The digestive gate expects bounded scalars. For backward
        # compatibility, treat any episode whose drift crossed the legacy
        # threshold as a correction-like event when gating identity/immune.
        gate_correction = float(
            max(correction_signal, self._last_drift > self.config.correction_drift_threshold)
        )
        scores = self.digestive_gate.ingest(
            drift=float(np.clip(self._last_drift, 0.0, 1.0)),
            correction_signal=gate_correction,
            novelty=1.0,
            identity_relevance=0.5,
            immune_conflict=0.0,
        )

        # Drive the critic with the cortex's drift scalar when the gate
        # allows it (default critic weight is 1.0, so this keeps the
        # previous always-on behavior unless a custom config lowers it).
        if scores["critic_weight"] > 0:
            self.world_model_critic._last_correction_prob = float(
                np.clip(self._last_drift, 0.0, 1.0)
            )
            self.world_model_critic.record_outcome(
                query=text,
                proposed_answer="",  # CortexAgent has no string answer to score.
                correction=text if correction_signal > 0.5 else None,
            )

        # High-drift episodes go to hippocampal storage as a replay bank
        # item keyed by the LM's hidden representation. The text field is
        # kept only for human debugging -- the cortex itself never reads
        # it back.
        if scores["hippocampus_weight"] > 0:
            self.neural_hippocampus.store(
                query=text,
                answer="",
                correction=text,
                prediction_error=self._last_drift,
                corrected_answer="",
            )

        # Identity accepts a token / correct_label; use the utterance's
        # first long alphanumeric token as the concept to update.
        if scores["identity_weight"] > 0:
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
            label = tokens[0] if tokens else "unknown"
            self.identity_hypernetwork.update_identity({
                "source": "user" if correction_signal < 0.5 else "user_correction",
                "correct_label": label,
                "token": label,
            })

        if scores["immune_weight"] > 0:
            self.skill_immune_cortex.add_detector(
                correction_text=text,
                mistake_class="cortex_drift",
                response="adjust_intent",
            )

        # Autoencoder gets every observation as a (passive) Hebbian-style
        # train step on the cortex's perception projector, scaled by the
        # gate's autoencoder weight so low-surprise steps learn less.
        autoencoder_lr = HEBBIAN_LR * float(scores["autoencoder_weight"])
        autoencoder_error = self.experience_autoencoder.train_step(
            {
                "situation": text,
                "model_answer": "",
                "correction": text if correction_signal > 0.5 else "",
                "revised_answer": "",
                "outcome": "corrected" if correction_signal > 0.5 else "accepted",
            },
            lr=autoencoder_lr,
        )

        return {
            "metabolized": True,
            "drift": self._last_drift,
            "correction_signal": correction_signal,
            "digestive_scores": scores,
            "consolidation_pressure": scores["consolidation_pressure"],
            "should_consolidate": self.digestive_gate.should_consolidate(),
            "hippocampus_wrote": scores["hippocampus_weight"] > 0,
            "autoencoder_error": autoencoder_error,
        }

    # ------------------------------------------------------------------
    # Knowledge store methods
    # ------------------------------------------------------------------
    def learn_fact(
        self,
        key: str,
        value: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a codebase fact to the attached knowledge store (no-op without one).

        Example::

            agent.learn_fact(
                "plastic-cortex vocab_size bug",
                "Clamp at vocab_size was removed so a 103-char tokenizer fits.",
            )
        """
        if self.knowledge_store is not None:
            self.knowledge_store.add_fact(key, value, metadata)


    # ------------------------------------------------------------------
    # Articulation (LM generation with cortex steering)
    # ------------------------------------------------------------------
    def articulate(
        self,
        prompt: str | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        apply_steering: bool = True,
        recall_query: str | None = None,
    ) -> str:
        """Generate text with the cortex's intent currently applied.

        Args:
            prompt: the prompt to feed the LM. If None, the last perceived
                utterance is used (so perceive() -> articulate() chains
                cleanly without re-stating the input).
            max_tokens: max generation length.
            temperature: LM sampling temperature. Defaults to 0.0
                (greedy) for deterministic post-correction behaviour.
            apply_steering: if True (default), apply the cortex's per-layer
                cvecs before generation and clear them after. If False,
                generate without cortex steering -- useful for baseline
                comparisons and the test suite.
            recall_query: optional query for the attached knowledge store.
                If provided (or defaulted from ``self._last_utterance`` when
                the store is present), retrieved facts are prepended to the
                prompt. No recall is performed when no store is attached.

        Returns:
            The LM's generated text. The cortex state is unchanged by this
            call (articulation is read-only w.r.t. warm_state).
        """
        if prompt is None:
            prompt = self._last_utterance or ""

        # Ground the prompt with retrieved repo facts when a knowledge
        # store is attached. Explicit recall_query takes precedence; otherwise
        # fall back to the last perceived utterance so perceive()->articulate()
        # chains naturally carry conversational context into recall.
        if self.knowledge_store is not None:
            query = recall_query if recall_query is not None else self._last_utterance
            if query is not None:
                prompt = self.knowledge_store.format_context(query) + prompt

        if not prompt:
            raise ValueError("articulate() needs a prompt or a prior perceive()")

        if apply_steering:
            cvecs = self.cortex.emit_all_cvecs()
            self.driver.set_cvecs_per_layer(cvecs, scale=self.config.articulate_scale)
            try:
                return self.driver.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature
                )
            finally:
                self.driver.clear_cvec()
        return self.driver.generate(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

    # Convenience: perceive -> metabolize -> optionally consolidate -> articulate.
    def turn(
        self,
        utterance: str,
        correction_signal: float | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        metabolize: bool = True,
    ) -> dict[str, Any]:
        """One full turn: absorb input, run metabolism, optionally consolidate, articulate reply."""
        warm = self.perceive(utterance, correction_signal=correction_signal)
        meta = self.metabolize(utterance) if metabolize else {"metabolized": False}

        consolidation = {"auto_consolidated": False}
        if (
            metabolize
            and self.config.auto_consolidate
            and self.should_consolidate()
        ):
            consolidation = {"auto_consolidated": True, **self.consolidate()}
            self.digestive_gate.reset()

        reply = self.articulate(
            prompt=utterance, max_tokens=max_tokens, temperature=temperature
        )
        return {
            "warm_norm": float(np.linalg.norm(warm)),
            "drift": self._last_drift,
            "correction_signal": self._last_correction_signal,
            "metabolized": meta.get("metabolized", False),
            "hippocampus_wrote": meta.get("hippocampus_wrote", False),
            "consolidated": consolidation["auto_consolidated"],
            "consolidation_summary": consolidation,
            "reply": reply,
        }

    # ------------------------------------------------------------------
    # Cold path (consolidation + persistence)
    # ------------------------------------------------------------------
    def consolidate(self) -> dict[str, Any]:
        """Move cortex warm into cold, plus organ consolidation fans.

        Replays are pulled from the hippocampus and passed as a list of
        d_embd vectors to cortex.consolidate(). Today the hippocampus
        returns episode dicts (string-keyed), so we synthesise replay
        tensors by re-embedding their `query` fields through the LM -- a
        placeholder until the hippocampus natively stores tensors.

        Returns a summary of what consolidation did.
        """
        # Hipppocampal consolidate produces slow-update summaries and
        # decays the raw traces it owns.
        summaries = self.neural_hippocampus.consolidate()

        # Build replay tensors from the consolidated queries. We pass these
        # to cortex.consolidate as the Second-speed update signal: the
        # cortex gets to absorb a tensor projection of what just got
        # persisted, even though the hippocampus itself still holds
        # strings today.
        replays: list[np.ndarray] = []
        for s in summaries:
            q = s.get("representative_query") or ""
            if not q:
                continue
            try:
                replays.append(self.driver.peek_embedding(q, last_token_only=False))
            except Exception:
                continue

        cortex_before = self.cortex.cold_state.copy()
        self.cortex.consolidate(replays=replays if replays else None)
        cortex_after = self.cortex.cold_state.copy()
        cold_drift = float(np.linalg.norm(cortex_after - cortex_before))

        return {
            "summary_count": len(summaries),
            "replay_count": len(replays),
            "cold_drift": cold_drift,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Persist the cortex's cold state plus all organ state.

        Warm state is intentionally NOT persisted: it is a session-level
        ephemeraliser. Cold state plus organ state is the agent's identity.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cortex_cold": self.cortex.cold_state.copy(),
            "cortex_proj_hidden": self.cortex.proj_hidden.copy(),
            "cortex_proj_c": self.cortex.proj_c.copy(),
            "cortex_config": self.cortex.config,
            "neural_hippocampus": self.neural_hippocampus,
            "world_model_critic": self.world_model_critic,
            "identity_hypernetwork": self.identity_hypernetwork,
            "skill_immune_cortex": self.skill_immune_cortex,
            "experience_autoencoder": self.experience_autoencoder,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str, config: CortexAgentConfig | None = None) -> "CortexAgent":
        """Reconstruct a CortexAgent from a saved state file.

        The cortex's cold_state, proj_hidden, and proj_c are restored
        from the saved payload; the driver and cortex wrapper are
        reconstructed from ``config`` (default: CortexAgentConfig()).
        """
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)

        agent = cls(config or CortexAgentConfig())
        # Restore learned cortex state. Overwrite whatever the freshly
        # initialised cortex had in cold_state and projectors.
        agent.cortex.cold_state = payload["cortex_cold"].astype(np.float32)
        agent.cortex.proj_hidden = payload["cortex_proj_hidden"].astype(np.float32)
        agent.cortex.proj_c = payload["cortex_proj_c"].astype(np.float32)
        agent.cortex.reset_warm_from_cold()

        agent.neural_hippocampus = payload["neural_hippocampus"]
        agent.world_model_critic = payload["world_model_critic"]
        agent.identity_hypernetwork = payload["identity_hypernetwork"]
        agent.skill_immune_cortex = payload["skill_immune_cortex"]
        agent.experience_autoencoder = payload["experience_autoencoder"]

        return agent

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "cortex": self.cortex.status(),
            "driver": self.driver.status(),
            "hippocampus": self.neural_hippocampus.status(),
            "critic": self.world_model_critic.status(),
            "identity": self.identity_hypernetwork.status(),
            "immune": self.skill_immune_cortex.status(),
            "autoencoder": self.experience_autoencoder.status(),
            "last_drift": self._last_drift,
            "last_correction_signal": self._last_correction_signal,
        }