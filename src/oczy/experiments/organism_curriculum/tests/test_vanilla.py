"""Tests for the VanillaAgent no-learning baseline (S0.7) and its integration
with the curriculum harness and held-out probe split (S0.6)."""

from __future__ import annotations

from oczy.experiments.baselines import VanillaAgent
from oczy.experiments.organism import OrganismAgent
from oczy.experiments.organism_curriculum.dataset import build_curriculum, split_probes
from oczy.experiments.organism_curriculum.run_curriculum import run_battery
from oczy.experiments.organism_curriculum.scoring import categorize_results
from plastic_cortex import PlasticCortex


def _probe_id_map(stage):
    """Map each Probe object to its ``"episode_id|request|category"`` id."""
    mapping = {}
    for ep in stage.episodes:
        for probe in ep.probes:
            mapping[probe] = f"{ep.id}|{probe.request}|{probe.category}"
    return mapping


def test_vanilla_noop_learn() -> None:
    """``learn`` is a true no-op: a second ``answer`` is identical to the first,
    proving the baseline never mutates backend state from corrections."""
    agent = VanillaAgent({"use_lm": False})
    request = "What is the opposite of hot?"
    before = agent.answer(request)
    agent.learn(request, "The opposite of hot is cold.")
    agent.correct("The opposite of hot is cold.", "cold")
    after = agent.answer(request)
    assert before == after


def test_vanilla_raw_backend_answers() -> None:
    """In raw mode VanillaAgent delegates to a fresh ``PlasticCortex`` with the
    same config, so both must return identical answers for the same request."""
    agent = VanillaAgent({"use_lm": False})
    cortex = PlasticCortex({})
    request = "What is the opposite of hot?"
    assert agent.answer(request) == cortex.answer(request)


def test_vanilla_matches_interface() -> None:
    """VanillaAgent exposes the six-method agent contract the harness relies on
    to drive it through ``run_battery`` / ``run_stage`` interchangeably."""
    agent = VanillaAgent()
    for name in ("answer", "learn", "correct", "consolidate", "memory_bytes", "profile_summary"):
        assert callable(getattr(agent, name, None)), f"missing interface method: {name}"


def test_vanilla_smoke_with_curriculum() -> None:
    """VanillaAgent runs end-to-end against a real curriculum stage and yields a
    computable per-category accuracy in [0.0, 1.0] alongside OrganismAgent."""
    (stage,) = build_curriculum(stage_names=("stage_0_grounding",))

    vanilla = VanillaAgent()
    organism = OrganismAgent()

    vanilla_results = run_battery(vanilla, stage, stage.episodes, split_ids=None)
    organism_results = run_battery(organism, stage, stage.episodes, split_ids=None)

    assert vanilla_results, "vanilla battery produced no probe results"
    assert organism_results, "organism battery produced no probe results"

    vanilla_cats = categorize_results(vanilla_results)
    assert vanilla_cats, "categorize_results returned no categories for vanilla"
    for _cat, (correct, total, acc) in vanilla_cats.items():
        assert total > 0
        assert 0 <= correct <= total
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


def test_vanilla_with_split() -> None:
    """Running the battery with ``split_ids`` restricts probes to exactly the
    dev partition returned by ``split_probes``."""
    (stage,) = build_curriculum(stage_names=("stage_0_grounding",))
    dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2")

    assert dev_ids, "split produced an empty dev set"
    assert dev_ids.isdisjoint(holdout_ids)

    vanilla = VanillaAgent()
    results = run_battery(vanilla, stage, stage.episodes, split_ids=dev_ids)

    id_map = _probe_id_map(stage)
    for probe, _answer, _ok in results:
        assert id_map[probe] in dev_ids

    assert len(results) == len(dev_ids)
