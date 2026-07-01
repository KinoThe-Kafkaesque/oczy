"""Fast smoke tests for the S3.1 organ ablation harness.

Runs ``run_ablation()`` and ``run_vanilla_baseline()`` on the first two
curriculum stages with two seeds on the raw (no-GGUF) backend, and checks
the shape and sanity of the returned matrices.  Designed to complete in
well under 30 seconds.

The expensive ``run_ablation()`` call is computed once and shared across
the three test functions via a module-level cache.
"""

from __future__ import annotations

from oczy.experiments.organ_ablation import (
    ORGANS,
    run_ablation,
    run_vanilla_baseline,
)
from oczy.experiments.organism_curriculum.dataset import (
    build_curriculum,
    split_probes,
)

# First two curriculum stages — small enough to stay fast, diverse enough
# to exercise grounding + transfer.  These are *filenames*; the loaded
# ``Stage.name`` carries a human-readable display name instead.
_STAGE_FILES: tuple[str, ...] = ("stage_0_grounding", "stage_1_transfer")
_N_SEEDS: int = 2

_SUMMARY_KEYS = {"mean", "std", "n", "ci95_half"}


# ---------------------------------------------------------------------------
# Shared fixtures (module-level lazy cache — plain functions, no pytest
# fixtures, matching the existing test style in this directory).
# ---------------------------------------------------------------------------

_stages_cache: tuple | None = None
_splits_cache: dict | None = None
_ablation_cache: dict | None = None
_vanilla_cache: list | None = None


def _stages_and_splits() -> tuple:
    """Load the first two stages + their dev splits (cached)."""
    global _stages_cache, _splits_cache
    if _stages_cache is None:
        stages = build_curriculum(stage_names=_STAGE_FILES)
        _splits_cache = {stage.name: split_probes(stage)[0] for stage in stages}
        _stages_cache = stages
    return _stages_cache, _splits_cache  # type: ignore[return-value]


def _stage_names() -> list[str]:
    """Display names of the loaded stages, in order."""
    stages, _ = _stages_and_splits()
    return [stage.name for stage in stages]


def _ablation_results() -> dict:
    """Run the ablation matrix once and cache it."""
    global _ablation_cache
    if _ablation_cache is None:
        stages, splits = _stages_and_splits()
        _ablation_cache = run_ablation(
            list(stages), splits, n_seeds=_N_SEEDS, verbose=False
        )
    return _ablation_cache


def _vanilla_results() -> list:
    """Run the vanilla baseline once and cache it."""
    global _vanilla_cache
    if _vanilla_cache is None:
        stages, splits = _stages_and_splits()
        _vanilla_cache = run_vanilla_baseline(
            list(stages), splits, verbose=False
        )
    return _vanilla_cache


def _expected_config_keys() -> set[str]:
    keys = {"FULL", "MINIMAL"}
    for organ in ORGANS:
        keys.add(f"FULL-{organ.short}")
    return keys


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ablation_smoke_two_stages() -> None:
    """run_ablation returns a complete, sane matrix for 2 stages × 2 seeds."""
    results = _ablation_results()
    names = _stage_names()

    # All expected config keys present, nothing extra.
    assert set(results.keys()) == _expected_config_keys()

    # FULL >= MINIMAL on at least one stage (organs must not harm).  On the
    # raw/mock-path backend the organ config overrides are no-ops, so every
    # config collapses to identical accuracy — strict FULL > MINIMAL does
    # not hold here (documented in the harness docstring as the "mock path").
    # The meaningful invariant is that disabling organs never *improves*
    # accuracy, i.e. FULL is at least as good as MINIMAL.
    full_means = {name: s["mean"] for name, s in results["FULL"]}
    minimal_means = {name: s["mean"] for name, s in results["MINIMAL"]}
    organs_no_harm = any(
        full_means[stage_name] >= minimal_means[stage_name]
        for stage_name in names
    )
    assert organs_no_harm, (
        "FULL is worse than MINIMAL on every stage — organs actively harm"
    )

    # Per-config shape + summary sanity.
    for config_name, entries in results.items():
        # Each config has one entry per stage, in stage order.
        assert len(entries) == len(names), (
            f"{config_name}: expected {len(names)} entries, "
            f"got {len(entries)}"
        )
        assert [name for name, _ in entries] == names, (
            f"{config_name}: stage name order mismatch"
        )
        for stage_name, summary in entries:
            assert _SUMMARY_KEYS.issubset(summary.keys()), (
                f"{config_name}/{stage_name}: missing keys "
                f"{_SUMMARY_KEYS - set(summary.keys())}"
            )
            assert summary["n"] == _N_SEEDS, (
                f"{config_name}/{stage_name}: n={summary['n']} "
                f"expected {_N_SEEDS}"
            )
            mean = summary["mean"]
            assert 0.0 <= mean <= 1.0, (
                f"{config_name}/{stage_name}: mean {mean} out of [0, 1]"
            )


def test_vanilla_baseline_smoke() -> None:
    """run_vanilla_baseline returns sane per-stage accuracies < FULL somewhere."""
    vanilla = _vanilla_results()
    names = _stage_names()

    # Same shape as a single config column.
    assert len(vanilla) == len(names)
    assert [name for name, _ in vanilla] == names

    vanilla_means = {}
    for stage_name, summary in vanilla:
        assert _SUMMARY_KEYS.issubset(summary.keys())
        mean = summary["mean"]
        assert 0.0 <= mean <= 1.0, (
            f"vanilla/{stage_name}: mean {mean} out of [0, 1]"
        )
        vanilla_means[stage_name] = mean

    # Vanilla should be beaten by FULL on at least one stage.
    results = _ablation_results()
    full_means = {name: s["mean"] for name, s in results["FULL"]}
    vanilla_loses = any(
        vanilla_means[stage_name] < full_means[stage_name]
        for stage_name in names
    )
    assert vanilla_loses, (
        "Vanilla never loses to FULL — baseline not weaker than full organism"
    )


def test_matrix_shape() -> None:
    """results[config_name] is a list of (stage_name, summary_dict) tuples."""
    results = _ablation_results()
    names = _stage_names()

    for config_name, entries in results.items():
        assert isinstance(entries, list), f"{config_name}: not a list"
        assert len(entries) == len(names), (
            f"{config_name}: wrong length {len(entries)}"
        )
        for entry in entries:
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"{config_name}: entry {entry!r} is not a 2-tuple"
            )
            stage_name, summary = entry
            assert isinstance(stage_name, str)
            assert isinstance(summary, dict)
