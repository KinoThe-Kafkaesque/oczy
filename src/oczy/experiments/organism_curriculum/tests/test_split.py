"""Tests for the held-out probe split (S0.6) and its validator.

These tests exercise the real bundled curriculum stages plus synthetic edge
cases, defending the concrete contracts of ``split_probes`` and
``validate_split``:

* determinism — same inputs always yield the same partition
* salt sensitivity — changing the salt re-partitions probes
* the tiny-stage guarantee — stages with fewer than 4 probes still get a
  non-empty holdout
* fraction boundaries — 0.0 empties holdout, 1.0 empties dev (for stages large
  enough that the tiny-stage guarantee does not engage)
* the empty stage — no probes means no partition
* validation — a healthy curriculum reports no errors, and an empty holdout on
  a sufficiently large stage is flagged as an error
"""

from __future__ import annotations

import math

from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    Stage,
    build_curriculum,
    split_probes,
)
from oczy.experiments.organism_curriculum.validation import validate_split


def _synthetic_stage(n_probes: int, name: str = "synthetic") -> Stage:
    """Build a stage holding ``n_probes`` distinct retention probes in one episode."""
    probes = tuple(
        Probe(
            request=f"q{i}",
            expected=f"a{i}",
            category="retention",
            match_mode="exact",
        )
        for i in range(n_probes)
    )
    episode = Episode(
        id="e1",
        initial_request="q",
        default_response="d",
        correction_utterance="c",
        corrected_label="l",
        corrected_response="r",
        domain="d",
        probes=probes,
    )
    return Stage(
        name=name,
        description="",
        consolidate_before=False,
        consolidate_after=False,
        episodes=(episode,),
    )


def _largest_stage() -> Stage:
    """Return the bundled stage with the most probes (most stable split signal)."""
    return max(build_curriculum(), key=lambda s: sum(len(e.probes) for e in s.episodes))


def test_split_determinism() -> None:
    """Calling ``split_probes`` twice with identical params returns identical sets."""
    stage = _largest_stage()
    dev_a, holdout_a = split_probes(stage, fraction=0.3, salt="v2")
    dev_b, holdout_b = split_probes(stage, fraction=0.3, salt="v2")
    assert dev_a == dev_b
    assert holdout_a == holdout_b


def test_split_stability() -> None:
    """Changing the salt re-partitions probes: the dev sets must differ."""
    stage = _largest_stage()
    dev_v2, _ = split_probes(stage, fraction=0.3, salt="v2")
    dev_v3, _ = split_probes(stage, fraction=0.3, salt="v3")
    assert dev_v2 != dev_v3, "salt change must move at least one probe between splits"


def test_split_min_holdout() -> None:
    """A stage with fewer than 4 probes is guaranteed a non-empty holdout.

    With ``fraction=0.0`` every probe hashes into dev, so the only way holdout
    becomes non-empty is the tiny-stage guarantee forcing one probe over — this
    exercises that code path deterministically.
    """
    stage = _synthetic_stage(n_probes=3)
    dev, holdout = split_probes(stage, fraction=0.0, salt="v2")
    assert len(holdout) >= 1
    # The forced probe is removed from dev, so the partition is still disjoint.
    assert dev.isdisjoint(holdout)


def test_split_fraction() -> None:
    """fraction=0.0 empties holdout and fraction=1.0 empties dev for a large stage.

    A 20-probe stage is well past the tiny-stage guarantee threshold (<4), so
    the split is governed purely by the fraction comparison.
    """
    stage = _synthetic_stage(n_probes=20)

    dev_zero, holdout_zero = split_probes(stage, fraction=0.0, salt="v2")
    assert holdout_zero == set()
    assert len(dev_zero) == 20

    dev_one, holdout_one = split_probes(stage, fraction=1.0, salt="v2")
    assert dev_one == set()
    assert len(holdout_one) == 20


def test_split_empty_stage() -> None:
    """A stage with zero probes returns two empty sets."""
    stage = _synthetic_stage(n_probes=0)
    dev, holdout = split_probes(stage, fraction=0.3, salt="v2")
    assert dev == set()
    assert holdout == set()


def test_validate_split() -> None:
    """A healthy curriculum reports no errors; an empty holdout is flagged.

    The bundled stages have enough probes that, at fraction=0.5, every stage
    receives a non-empty holdout — so ``validate_split`` reports no errors.
    Separately, a synthetic stage with 5 probes at fraction=0.0 places every
    probe in dev (the tiny-stage guarantee only covers <4 probes), leaving
    holdout empty, which ``validate_split`` must surface as an error.

    Note: a 1-probe stage at fraction=0.0 cannot trigger this error because the
    tiny-stage guarantee forces its single probe into holdout; >=4 probes are
    required to reach an genuinely empty holdout.
    """
    stages = build_curriculum()
    healthy = validate_split(stages, fraction=0.5, salt="v2")
    assert healthy.ok
    assert healthy.errors == []

    bad_stage = _synthetic_stage(n_probes=5, name="empty_holdout")
    report = validate_split((bad_stage,), fraction=0.0, salt="v2")
    assert not report.ok
    assert any("empty_holdout" in msg for msg in report.errors)


def test_split_guarantee_engages_for_unlucky_large_stage() -> None:
    """Bundled stage 0 hashes all 8 probes into dev at fraction=0.3, salt="v2"
    — the degenerate split that invalidated the first S2.2 run. The
    generalized guarantee must promote ceil(fraction * total) probes instead
    of returning an empty holdout.
    """
    stage_0 = build_curriculum()[0]
    total = sum(len(e.probes) for e in stage_0.episodes)
    dev, holdout = split_probes(stage_0, fraction=0.3, salt="v2")
    assert len(holdout) == math.ceil(0.3 * total) > 0
    assert dev.isdisjoint(holdout)
    assert len(dev) + len(holdout) == total


def test_split_guarantee_never_alters_nonempty_holdouts() -> None:
    """The guarantee only engages on an EMPTY holdout: every other bundled
    stage keeps the exact partition it had before the guarantee was
    generalized, preserving comparability with all previously logged numbers.
    """
    expected_holdout_counts = {1: 1, 2: 3, 3: 3, 4: 4, 5: 3}
    stages = build_curriculum()
    for idx, count in expected_holdout_counts.items():
        _, holdout = split_probes(stages[idx], fraction=0.3, salt="v2")
        assert len(holdout) == count, f"stage {idx} holdout changed"


def test_validate_split_healthy_at_preregistered_fraction() -> None:
    """The Sprint-2 pre-registered split params (fraction=0.3, salt="v2") must
    validate cleanly across the whole bundled curriculum.
    """
    report = validate_split(build_curriculum(), fraction=0.3, salt="v2")
    assert report.ok
    assert report.errors == []
