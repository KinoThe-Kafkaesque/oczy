"""Transfer-generalization test for the CortexAgent policy head."""

from __future__ import annotations

import warnings

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.organism import OrganismAgent
from oczy.experiments.organism_curriculum.dataset import build_curriculum
from oczy.experiments.organism_curriculum.run_curriculum import _MockDriver, run_stage
from oczy.experiments.organism_curriculum.tests.test_shim_policy_delta import _match_key
from plastic_cortex.kv_cortex import KVCortexConfig


class _StableMockDriver(_MockDriver):
    """Deterministic, scaled, nearly-orthogonal hidden vectors.

    The bundled :class:`_MockDriver` is collision-prone in low dimensions; this
    subclass returns high-amplitude unit vectors seeded by the input text so the
    policy head learns a stable corrected-vs-wrong margin across test runs.
    """

    def __init__(self, n_embd: int = 16, n_layers: int = 2) -> None:
        super().__init__(n_embd=n_embd, n_layers=n_layers)
        self._cache: dict[str, np.ndarray] = {}

    def peek_embedding(
        self,
        text: str,
        last_token_only: bool = True,
    ) -> np.ndarray:
        del last_token_only
        if text not in self._cache:
            seed = 0
            for i, char in enumerate(text):
                seed = (seed + ord(char) * (i + 31)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.n_embd)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = (vec / norm) * 10.0
            self._cache[text] = vec.astype(np.float64)
        return self._cache[text].copy()


def _score_labels_for_request(
    cortex: CortexAgent,
    labels: list[str],
    request: str,
) -> dict[str, float]:
    """Compute policy scores for ``labels`` after perceiving ``request``."""
    # Perceiving the request updates the warm state / hidden used by policy_score.
    cortex.perceive(request)
    scores = cortex.policy_score(labels)
    return {labels[i]: float(scores[i]) for i in range(len(labels))}


def _label_score(
    text: str,
    scores: dict[str, float],
) -> float | None:
    key = _match_key(text, scores)
    return scores.get(key) if key else None


def test_policy_head_generalizes_to_transfer_probes() -> None:
    """After stage 0 corrections, policy head should favor corrected labels on stage 1 transfer probes."""
    stages = build_curriculum(stage_names=("stage_0_grounding", "stage_1_transfer"))
    assert len(stages) == 2, "expected stages 0 and 1"
    stage0, stage1 = stages

    config = {
        "use_cortex_policy": True,
        "use_value_baseline": True,
        "use_acceptance_policy_reward": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        agent = OrganismAgent(config)

    driver = _StableMockDriver()
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_policy_head=True,
        policy_learning_rate=0.001,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()
    # Zero-initialize policy weights for deterministic, monotonic learning.
    cortex._policy_W = np.zeros(cfg.cortex.d_cortex + driver.n_embd, dtype=np.float64)
    cortex._policy_b = 0.0
    agent.cortex_agent = cortex

    # Stage 0: acquire the corrected senses.
    run_stage(agent, stage0, adapter=None, instrument_policy=True)

    # Stage 1: present transfer probes (different wording, same concept).
    run_stage(agent, stage1, adapter=None, instrument_policy=True)

    labels = list(agent.plastic_cortex.labels)
    margins: list[float] = []

    for ep in stage1.episodes:
        correct_key = _match_key(ep.corrected_response, dict.fromkeys(labels, 0.0)) or _match_key(
            ep.corrected_label, dict.fromkeys(labels, 0.0)
        )
        wrong_key = _match_key(ep.default_response, dict.fromkeys(labels, 0.0))
        if not correct_key or not wrong_key or correct_key == wrong_key:
            continue
        for probe in ep.probes:
            if probe.category != "transfer":
                continue
            scores = _score_labels_for_request(cortex, labels, probe.request)
            correct_score = scores.get(correct_key)
            wrong_score = scores.get(wrong_key)
            if correct_score is None or wrong_score is None:
                continue
            margins.append(correct_score - wrong_score)

    assert margins, "no transfer-probe margins could be computed"
    avg_margin = sum(margins) / len(margins)
    assert avg_margin > 0.0, (
        f"expected positive transfer-probe policy margin, got {avg_margin}"
    )
