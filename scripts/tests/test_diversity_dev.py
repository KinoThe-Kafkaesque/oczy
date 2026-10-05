"""Scientific controls for the next fixed-budget curriculum comparison."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.diversity_dev.contract import admission, data, reference_prompt
from scripts.diversity_dev.prepare import build_data

ROOT = Path(__file__).resolve().parents[2] / "experiments/context-preservation-dev-v2"


def test_rotating_curriculum_keeps_update_counts_and_crossed_contexts():
    d = build_data(ROOT)
    groups = d["capacity_training"]["groups"]
    assert len(groups) == 3 and all(len(g) == 9 for g in groups)
    exposure = Counter((r["category"], r["input"]) for step in range(48) for r in groups[step % 3])
    assert len(exposure) == 27 and set(exposure.values()) == {16}
    for word in {r["input"] for r in d["capacity_training"]["rows"]}:
        assert len({r["target"] for r in d["capacity_training"]["rows"] if r["input"] == word}) == 3
    assert groups[0] == json.loads((ROOT / "capacity_training/data.json").read_text())["rows"]


def test_new_confirmation_excludes_previous_probe_and_training_words():
    d = build_data(ROOT)
    new = {r["input"] for r in d["probes"]["confirmation"][0]["rows"]}
    old = {"pear", "plum", "kiwi", "fig", "lime", "melon", "grape", "peach", "apple", "mango", "lemon", "guava"}
    taught = {r["input"] for r in d["capacity_training"]["rows"]}
    assert not new & (old | taught)
    assert len(d["probes"]["confirmation"][0]["rows"]) == 16


@pytest.mark.parametrize("role", ["probes", "oracle", "correction_training"])
def test_training_cannot_read_measurements(tmp_path, role):
    with pytest.raises(ValueError, match="cannot read"):
        data(tmp_path, {}, "train", role)


def test_every_reference_excludes_transformed_confirmation_target():
    d = build_data(ROOT)
    for row in d["probes"]["confirmation"][0]["rows"]:
        for condition in ("no_context", "direct_oracle", "complete_table", "text_control", "text_diversity", "chat_control", "chat_diversity", "zlib_control", "zlib_diversity"):
            prompt = reference_prompt(row, condition, d["capacity_training"], d["oracle"]["stages"][0]["table"])
            if row["category"] in ("amber", "cobalt"):
                assert row["target"] not in json.dumps(prompt)


def test_acquisition_requires_complete_fits_coverage_and_all_correct():
    d = build_data(ROOT)
    probes = d["probes"]["confirmation"][0]["rows"]
    manifest = {"training": {"seeds": [0, 1, 2]}}
    rows = [{**r, "seed": seed, "split": "confirmation", "condition": "diversity", "correct": True} for seed in range(3) for r in probes]
    fits = [{"seed": seed, "training_fit": [{**r, "correct": True} for r in d["capacity_training"]["rows"]]} for seed in range(3)]
    assert admission(rows, fits, manifest, probes)
    assert not admission(rows[:-1], fits, manifest, probes)
    assert not admission(rows + [rows[0]], fits, manifest, probes)
    changed = copy.deepcopy(rows)
    changed[0]["correct"] = False
    assert not admission(changed, fits, manifest, probes)
    changed = copy.deepcopy(fits)
    changed[0]["training_fit"].pop()
    assert not admission(rows, changed, manifest, probes)
