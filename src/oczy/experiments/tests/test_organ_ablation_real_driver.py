"""Unit tests for the --real-driver flag and policy-gate wiring in
organ_ablation.py.  No GGUF is loaded; we exercise config merging,
Namespace construction, and CLI flag parsing only."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from oczy.experiments.organ_ablation import (
    _build_minimal_args,
    parse_args,
    run_ablation,
    write_outputs,
)
from oczy.experiments.organism_curriculum.dataset import (
    build_curriculum,
    split_probes,
)

# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------

def test_parse_args_default_is_mock() -> None:
    """--real-driver is False by default."""
    args = parse_args([])
    assert args.real_driver is False
    assert args.seeds == 3


def test_parse_args_real_driver_flag() -> None:
    """--real-driver flag sets the attribute to True."""
    args = parse_args(["--real-driver"])
    assert args.real_driver is True


def test_parse_args_real_driver_with_seeds() -> None:
    """--real-driver can be combined with --seeds and --stages."""
    args = parse_args(["--real-driver", "--seeds", "5", "--stages", "stage_0_grounding"])
    assert args.real_driver is True
    assert args.seeds == 5
    assert args.stages == ["stage_0_grounding"]


# ---------------------------------------------------------------------------
# _build_minimal_args
# ---------------------------------------------------------------------------

def test_build_minimal_args_mock_default() -> None:
    """Default is mock path (no real driver)."""
    ns = _build_minimal_args(seeds=3)
    assert ns.use_real_driver is False
    assert ns.use_cortex_agent_mock is False
    assert ns.agent == "OrganismAgent"


def test_build_minimal_args_real_driver() -> None:
    """use_real_driver=True propagates to the namespace."""
    ns = _build_minimal_args(seeds=5, use_real_driver=True)
    assert ns.use_real_driver is True
    assert ns.seeds == 5


# ---------------------------------------------------------------------------
# run_ablation — config merging with real-driver gates
# ---------------------------------------------------------------------------

def _fake_stages_and_splits() -> tuple:
    """Load stage_0_grounding only with dev split to keep it fast."""
    stages = build_curriculum(stage_names=("stage_0_grounding",))
    splits = {s.name: split_probes(s)[0] for s in stages}
    return stages, splits


def test_run_ablation_mock_path_still_works() -> None:
    """Smoke: run_ablation with use_real_driver=False still works."""
    stages, splits = _fake_stages_and_splits()
    results = run_ablation(
        stages, splits, n_seeds=1, use_real_driver=False, verbose=False,
    )
    assert "FULL" in results
    assert "MINIMAL" in results
    assert len(results["FULL"]) == 1
    assert results["FULL"][0][0] is not None


# ---------------------------------------------------------------------------
# write_outputs — real-driver label
# ---------------------------------------------------------------------------

def _quick_results(stages, splits):
    """One-seed ablation results for write_outputs tests."""
    return run_ablation(
        stages, splits, n_seeds=1, use_real_driver=False, verbose=False,
    )


def test_write_outputs_mock_path_label() -> None:
    """write_outputs with use_real_driver=False uses mock labels."""
    stages, splits = _fake_stages_and_splits()
    results = _quick_results(stages, splits)
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        md_path, json_path = write_outputs(
            results, None, out_dir, "test_mock",
            use_real_driver=False,
        )
        md_text = md_path.read_text()
        assert "mock (raw backend)" in md_text
        assert "raw (no cortex)" in md_text

        j = json.loads(json_path.read_text())
        assert j["path"] == "mock"


def test_write_outputs_real_driver_label() -> None:
    """write_outputs with use_real_driver=True uses real-driver labels."""
    stages, splits = _fake_stages_and_splits()
    results = _quick_results(stages, splits)
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        md_path, json_path = write_outputs(
            results, None, out_dir, "test_real",
            use_real_driver=True,
        )
        md_text = md_path.read_text()
        assert "real-driver (LlamaCVecDriver GGUF)" in md_text
        assert "real GGUF + CortexAgent" in md_text
        assert "MOCK-PATH CAVEAT" not in md_text

        j = json.loads(json_path.read_text())
        assert j["path"] == "real"


def test_write_outputs_per_organ_delta_table() -> None:
    """write_outputs includes a per-organ delta table (FULL minus FULL-organ)."""
    stages, splits = _fake_stages_and_splits()
    results = _quick_results(stages, splits)
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        md_path, _ = write_outputs(
            results, None, out_dir, "test_delta",
            use_real_driver=False,
        )
        md_text = md_path.read_text()
        assert "## Per-Organ \u0394 (FULL minus FULL-organ)" in md_text
        for organ_short in ("hippocampus", "critic", "identity", "immune",
                            "autoencoder", "dsi", "scope_slot_reranker"):
            assert f"FULL-{organ_short}" in results
