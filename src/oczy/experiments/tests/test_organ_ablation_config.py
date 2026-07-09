"""Unit tests for :func:`oczy.experiments.organ_ablation.build_configs`."""

from __future__ import annotations

import json
from typing import Any

from oczy.experiments.organ_ablation import ORGANS, build_configs


# ---------------------------------------------------------------------------
# Keys & shape
# ---------------------------------------------------------------------------

def test_build_configs_has_all_keys() -> None:
    configs = build_configs()
    expected = {"FULL", "MINIMAL"} | {f"FULL-{o.short}" for o in ORGANS}
    assert set(configs.keys()) == expected
    # Sanity: there really are 7 organs, so 9 keys total.
    assert len(ORGANS) == 7
    assert len(configs) == 9


# ---------------------------------------------------------------------------
# FULL — empty delta, all defaults
# ---------------------------------------------------------------------------

def test_full_config_is_empty() -> None:
    configs = build_configs()
    assert configs["FULL"] == {}


# ---------------------------------------------------------------------------
# MINIMAL — every organ disabled
# ---------------------------------------------------------------------------

def test_minimal_config_disables_all() -> None:
    configs = build_configs()
    minimal = configs["MINIMAL"]
    for organ in ORGANS:
        for key, value in organ.off_cfg.items():
            assert key in minimal, f"MINIMAL missing off-switch {key!r} for {organ.short}"
            assert minimal[key] == value, (
                f"MINIMAL[{key!r}] == {minimal[key]!r}, expected {value!r}"
            )
    # MINIMAL must not carry anything beyond the union of all off-switches.
    expected_keys = {k for o in ORGANS for k in o.off_cfg}
    assert set(minimal.keys()) == expected_keys


# ---------------------------------------------------------------------------
# FULL-X — exactly one organ flipped, nothing else
# ---------------------------------------------------------------------------

def test_each_ablation_flips_exactly_one_thing() -> None:
    configs = build_configs()
    # Map every off-switch key to the organ that owns it.
    key_to_organ = {k: o.short for o in ORGANS for k in o.off_cfg}

    for organ in ORGANS:
        name = f"FULL-{organ.short}"
        cfg = configs[name]
        # The config must equal exactly this organ's off_cfg and nothing more.
        assert cfg == organ.off_cfg, (
            f"{name} config {cfg!r} != expected {organ.off_cfg!r}"
        )
        # Exactly one key touched.
        assert len(cfg) == 1, f"{name} should touch exactly one key, got {len(cfg)}"

        key, value = next(iter(cfg.items()))
        owner = key_to_organ[key]
        assert owner == organ.short, (
            f"{name} touches {key!r} owned by {owner!r}, not {organ.short!r}"
        )

        if key == "scope_rerank_weight":
            assert value == 0.0, f"{name} scope_rerank_weight should be 0.0, got {value!r}"
        else:
            assert value is False, (
                f"{name}[{key!r}] should be False, got {value!r}"
            )


# ---------------------------------------------------------------------------
# Short-name uniqueness
# ---------------------------------------------------------------------------

def test_organs_have_unique_shorts() -> None:
    shorts = [o.short for o in ORGANS]
    assert len(shorts) == len(set(shorts)), f"duplicate organ shorts: {shorts}"


# ---------------------------------------------------------------------------
# JSON-serializability
# ---------------------------------------------------------------------------

def test_config_values_are_serializable() -> None:
    configs = build_configs()
    for name, cfg in configs.items():
        # Must round-trip through JSON; values restricted to bool/int/float/str.
        blob = json.dumps(cfg)
        round_tripped = json.loads(blob)
        assert round_tripped == cfg, f"{name} did not round-trip: {cfg!r}"

        for value in cfg.values():
            assert isinstance(value, (bool, int, float, str)), (
                f"{name} has non-serializable value {value!r} "
                f"of type {type(value).__name__}"
            )
