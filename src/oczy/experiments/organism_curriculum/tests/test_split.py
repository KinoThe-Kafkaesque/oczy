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
    DEFAULT_SPLIT_SALT,
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


def test_v2_2_split_is_category_stratified() -> None:
    """Every multi-probe capability category appears in dev and holdout."""
    for stage in build_curriculum():
        dev, holdout = split_probes(stage)
        by_category: dict[str, set[str]] = {}
        for ep in stage.episodes:
            for probe in ep.probes:
                by_category.setdefault(probe.category, set()).add(
                    f"{ep.id}|{probe.request}|{probe.category}"
                )
        for category, ids in by_category.items():
            if len(ids) >= 2:
                assert ids & dev, f"{stage.name}/{category} missing from dev"
                assert ids & holdout, f"{stage.name}/{category} missing from holdout"


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
    """A >=4-probe stage whose probes ALL hash into dev (the degenerate split
    that invalidated the first S2.2 run, before the v2.1 expansion) must get
    ceil(fraction * total) promoted probes instead of an empty holdout.

    Uses a synthetic stage pinned to a (fraction, salt) combination verified
    to threshold every probe into dev, so the guarantee path is exercised
    deterministically regardless of bundled-eval content.
    """
    stage = _synthetic_stage(n_probes=8, name="unlucky")
    fraction, salt = 0.05, "v2"
    dev_raw = 0
    # Find the raw assignment first so the test is self-checking: if this
    # (fraction, salt) ever stops producing an empty raw holdout, fail loudly
    # rather than silently testing nothing.
    import hashlib as _h
    for ep in stage.episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            v = int.from_bytes(_h.sha256(f"{salt}:{pid}".encode()).digest()[:4], "big") / 0xFFFFFFFF
            if v >= fraction:
                dev_raw += 1
    assert dev_raw == 8, "fixture no longer degenerate; pick a new (fraction, salt)"
    dev, holdout = split_probes(stage, fraction=fraction, salt=salt)
    assert len(holdout) == math.ceil(fraction * 8) > 0
    assert dev.isdisjoint(holdout)
    assert len(dev) + len(holdout) == 8


def test_legacy_split_counts_locked_to_eval_v2_1() -> None:
    """Lock the bundled dev/holdout counts at the pre-registered params.

    Updated for the human-approved eval v2.1 expansion (2026-07-03): stages
    0/1/2 grew (new probes hash independently; existing probes keep their raw
    assignment). Note stage 0's holdout is now raw-hash assigned, so the
    emptiness guarantee no longer engages there — the 3 previously PROMOTED
    v2 probes reverted to dev (disclosed in the expansion log; S2.1 ran on
    the promoted trio, later experiments run on v2.1's holdout).
    """
    expected = {0: 3, 1: 9, 2: 4, 3: 3, 4: 4, 5: 3}
    stages = build_curriculum()
    for idx, count in expected.items():
        _, holdout = split_probes(stages[idx], fraction=0.3, salt="v2")
        assert len(holdout) == count, f"stage {idx} holdout count changed"


def test_split_counts_locked_to_eval_v2_2() -> None:
    """Lock the human-approved category-stratified v2.2 split counts."""
    expected = {0: 6, 1: 12, 2: 10, 3: 4, 4: 3, 5: 4}
    stages = build_curriculum()
    for idx, count in expected.items():
        _, holdout = split_probes(
            stages[idx], fraction=0.3, salt=DEFAULT_SPLIT_SALT
        )
        assert len(holdout) == count, f"stage {idx} holdout count changed"


def test_validate_split_healthy_at_preregistered_fraction() -> None:
    """The Sprint-2 pre-registered split params (fraction=0.3, salt="v2") must
    validate cleanly across the whole bundled curriculum.
    """
    report = validate_split(
        build_curriculum(), fraction=0.3, salt=DEFAULT_SPLIT_SALT
    )
    assert report.ok
    assert report.errors == []
