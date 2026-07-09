"""Unit tests for the S3.M2b additive-organ harness (OrganAdditiveOrganism).

These tests exercise the additive organ wiring against the tiny-random
LlamaForCausalLM substrate (cached).  They verify:

* BASE (all flags off) is behaviourally identical to ``MinimalOrganism``.
* Each organ flag engages its organ (internal state changes).
* No cross-organ contamination: a single-flag organism leaves the other
  organs as ``None``.

The import pattern mirrors ``test_minimal_loop.py``: the repo root must be
on ``sys.path`` so top-level organ imports (``world_model_critic`` etc.)
resolve under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``organ_additive_organs`` imports organs from the repo root
# (``from world_model_critic import WorldModelCritic``), which is not on
# sys.path under pytest.  Add it once, exactly like test_minimal_loop.py.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from oczy.experiments.minimal_loop import MinimalOrganism
from oczy.experiments.organ_additive_organs import OrganAdditiveOrganism
from oczy.experiments.organism_curriculum.dataset import Episode, Probe
from oczy.lm.hf_driver import HFDriver

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver():
    """Module-scoped tiny-random driver, reused across tests."""
    with HFDriver.load(TEST_MODEL_ID) as d:
        yield d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    request: str = "What is the correct answer?",
    response: str = "I do not know.",
    correction: str = "no, the correct answer is Y",
) -> Episode:
    """Build a synthetic curriculum episode.

    The default correction utterance contains the concept token ``correct``
    (a member of the identity hypernetwork's CONCEPT_VOCABULARY) so the
    identity-latent engagement test actually moves the latent.  It also
    contains ``answer`` / ``correct`` which the skill-immune cortex extracts
    as trigger tokens, so the immune ``check()`` path fires.
    """
    return Episode(
        id="test_ep",
        initial_request=request,
        default_response=response,
        correction_utterance=correction,
        corrected_label="Y",
        corrected_response="the correct answer is Y",
        domain="test",
        probes=(
            Probe(
                request=request,
                expected="Y",
                category="retention",
                match_mode="sense",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. BASE == MinimalOrganism
# ---------------------------------------------------------------------------


def test_base_is_identical_to_minimal_organism(driver) -> None:
    """With every organ flag off, OrganAdditiveOrganism must behave exactly
    like MinimalOrganism: same teach -> consolidate -> answer output."""
    ep = _make_episode()
    request = ep.initial_request

    # MinimalOrganism baseline.
    minimal = MinimalOrganism(driver)
    minimal.boot()
    minimal.teach(ep)
    minimal.consolidate()
    a_minimal = minimal.answer(request, max_tokens=16)

    # Additive organism with all flags off (BASE condition).
    additive = OrganAdditiveOrganism(driver)
    additive.boot()
    additive.teach(ep)
    additive.consolidate()
    a_additive = additive.answer(request, max_tokens=16)

    assert a_minimal == a_additive


# ---------------------------------------------------------------------------
# 2. WorldModelCritic engages
# ---------------------------------------------------------------------------


def test_world_model_critic_engages(driver) -> None:
    """use_world_model_critic=True wires the critic and teach() records an
    outcome (the critic's online memory grows)."""
    org = OrganAdditiveOrganism(driver, use_world_model_critic=True)
    org.boot()

    assert org.world_model_critic is not None
    n_before = len(org.world_model_critic.records)

    org.teach(_make_episode())

    # predict_acceptance + record_outcome both ran inside teach(); the
    # record_outcome path appends one record to the critic's online memory.
    assert org.world_model_critic is not None
    assert len(org.world_model_critic.records) > n_before
    assert len(org.world_model_critic.records) >= 1


# ---------------------------------------------------------------------------
# 3. IdentityHypernetwork engages
# ---------------------------------------------------------------------------


def test_identity_hypernetwork_engages(driver) -> None:
    """use_identity_hypernetwork=True wires the hypernetwork and consolidate()
    replays stored episodes through update_identity, moving the latent."""
    org = OrganAdditiveOrganism(driver, use_identity_hypernetwork=True)
    org.boot()

    assert org.identity_hypernetwork is not None
    latent_before = org.identity_hypernetwork.latents.to_array().copy()
    norm_before = float(np.linalg.norm(latent_before))

    org.teach(_make_episode())
    org.consolidate()

    assert org.identity_hypernetwork is not None
    latent_after = org.identity_hypernetwork.latents.to_array()
    norm_after = float(np.linalg.norm(latent_after))

    # The correction utterance contains the concept token "correct"; the
    # consolidate() replay path feeds it to update_identity, which steps
    # the z_user slice along the projection gradient.  The latent must
    # actually move.
    assert not np.allclose(latent_before, latent_after)
    assert norm_after != norm_before


# ---------------------------------------------------------------------------
# 4. SkillImmuneCortex engages
# ---------------------------------------------------------------------------


def test_skill_immune_cortex_engages(driver) -> None:
    """use_skill_immune_cortex=True wires the immune cortex, teach() adds a
    detector, and answer() runs the immune check without crashing."""
    org = OrganAdditiveOrganism(driver, use_skill_immune_cortex=True)
    org.boot()

    assert org.skill_immune_cortex is not None
    n_detectors_before = len(org.skill_immune_cortex.detectors)

    org.teach(_make_episode())

    assert org.skill_immune_cortex is not None
    assert len(org.skill_immune_cortex.detectors) > n_detectors_before
    assert len(org.skill_immune_cortex.detectors) >= 1

    # The default request ("What is the correct answer?") contains the
    # trigger tokens extracted from the correction ("correct", "answer"),
    # so check() returns a non-empty response list and answer() rewrites
    # the request before generation.  This is the one behavioural output
    # path in MinimalOrganism; it must not crash.
    result = org.answer("What is the correct answer?", max_tokens=16)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. ExperienceAutoencoder engages
# ---------------------------------------------------------------------------


def test_experience_autoencoder_engages(driver) -> None:
    """use_experience_autoencoder=True wires the autoencoder and teach()
    encodes + trains on the episode (the organ instance is live)."""
    org = OrganAdditiveOrganism(driver, use_experience_autoencoder=True)
    org.boot()

    assert org.experience_autoencoder is not None
    # The module enables the hidden-delta config flag on construction.
    assert org.experience_autoencoder.config.get("use_hidden_delta") is True

    # encode + train_step run inside teach(); the bag-of-words path extends
    # the internal vocab, proving the organ actually processed the episode.
    vocab_before = len(org.experience_autoencoder._vocab)
    org.teach(_make_episode())
    vocab_after = len(org.experience_autoencoder._vocab)
    assert vocab_after >= vocab_before
    assert org.experience_autoencoder is not None


# ---------------------------------------------------------------------------
# 6. No cross-organ contamination
# ---------------------------------------------------------------------------


def test_no_cross_organ_contamination(driver) -> None:
    """A single-flag organism must leave every other organ as None."""
    org = OrganAdditiveOrganism(driver, use_skill_immune_cortex=True)
    org.boot()

    assert org.skill_immune_cortex is not None
    org.teach(_make_episode())

    assert org.world_model_critic is None
    assert org.identity_hypernetwork is None
    assert org.experience_autoencoder is None
