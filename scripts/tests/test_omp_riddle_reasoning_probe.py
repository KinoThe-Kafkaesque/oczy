"""Deterministic regression tests for the OMP RPC riddle-reasoning benchmark.

Covers bank validation and frozen shape, deterministic five-form generation,
answer-position Latin-square coverage, deterministic riddle-order shuffle,
scorer correctness, wrong-answer structural acceptance, duplicate/missing
format errors, first-valid-authoritative semantics, OMP argument-repair
capture, schedule/model rotation, summary/category/position/majority/stable
metrics, paired comparison determinism, ranking omission on unequal counts,
atomic partial artifact checkpointing, prompt-ack/agent-end race, and fake
RPC success/wrong/malformed/setup-failure paths.

No real OMP binary, provider, network, or model is involved.  Every test is
deterministic and full-suite safe.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# ─── dynamic import of the probe module ───────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "benchmarks" / "omp" / "riddle_reasoning_probe.py"

_spec = importlib.util.spec_from_file_location(
    "oczy_omp_riddle_reasoning_probe_tests", PROBE_PATH
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)


# ─── synthetic bank fixture ───────────────────────────────────────────

_CATEGORIES = [
    "logic", "wordplay", "lateral", "math", "pattern",
    "spatial", "temporal", "causal", "semantic", "meta",
]


def _make_option(riddle_num: int, opt_letter: str) -> dict:
    return {
        "option_id": f"r{riddle_num:02d}-{opt_letter}",
        "text": f"Option {opt_letter} for riddle {riddle_num}.",
    }


def _make_riddle(num: int, category: str) -> dict:
    options = [_make_option(num, chr(ord("a") + i)) for i in range(5)]
    return {
        "id": f"orz_{num:02d}",
        "category": category,
        "difficulty": "medium",
        "prompt": f"Riddle number {num} in category {category}: what is the answer?",
        "options": options,
        "correct_option_id": options[0]["option_id"],
        "proof": f"Proof for riddle {num}: option a is correct by construction.",
        "ambiguity_audit": f"Unambiguous riddle {num}.",
        "originality_note": f"Original riddle {num}.",
        "verification": "exhaustive_enumeration",
    }


def _make_bank() -> dict:
    """Build a valid 20-riddle bank: 10 categories × 2, 5 options each."""
    riddles = []
    for i in range(20):
        cat = _CATEGORIES[i % 10]
        riddles.append(_make_riddle(i + 1, cat))
    return {
        "schema_version": probe.BANK_SCHEMA_VERSION,
        "benchmark_version": probe.BANK_BENCHMARK_VERSION,
        "provenance": {"source": "test"},
        "exclusions": [],
        "categories": _CATEGORIES,
        "riddles": riddles,
    }


@pytest.fixture
def bank() -> dict:
    return _make_bank()


@pytest.fixture
def bank_file(tmp_path: Path, bank: dict) -> Path:
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank, ensure_ascii=False))
    return p


@pytest.fixture
def forms(bank: dict) -> list[dict]:
    return probe.generate_forms(bank)


# ─── helpers ──────────────────────────────────────────────────────────


def _all_correct_submission(form: dict) -> list[dict]:
    """Build a submission where every choice matches the correct letter."""
    return [
        {"number": i + 1, "choice": form["correct_letters"][i], "explanation": "ok"}
        for i in range(probe.RIDDLES_PER_FORM)
    ]


def _all_wrong_submission(form: dict) -> list[dict]:
    """Build a structurally valid submission where every choice is wrong."""
    subs = []
    for i in range(probe.RIDDLES_PER_FORM):
        correct = form["correct_letters"][i]
        # Pick the letter after the correct one (wrapping), guaranteed different.
        idx = probe.DISPLAY_LETTERS.index(correct)
        wrong = probe.DISPLAY_LETTERS[(idx + 1) % probe.OPTIONS_PER_RIDDLE]
        subs.append({"number": i + 1, "choice": wrong, "explanation": "no"})
    return subs


def test_public_model_metadata_strips_credentials() -> None:
    model = {
        "id": "safe-model",
        "contextWindow": 1000,
        "headers": {"Authorization": "Bearer must-not-leak"},
        "compatConfig": {"apiKey": "must-not-leak"},
    }
    public = probe._public_model_metadata(model)
    assert public == {"id": "safe-model", "contextWindow": 1000}
    assert "must-not-leak" not in json.dumps(public)


# ─── bank validation ──────────────────────────────────────────────────


def test_validate_bank_accepts_valid_bank(bank: dict) -> None:
    """A well-formed 20-riddle bank passes validation without raising."""
    probe.validate_bank(bank)  # must not raise


def test_validate_bank_rejects_wrong_schema_version(bank: dict) -> None:
    bank["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_wrong_benchmark_version(bank: dict) -> None:
    bank["benchmark_version"] = "wrong"
    with pytest.raises(ValueError, match="benchmark_version"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_wrong_riddle_count(bank: dict) -> None:
    bank["riddles"] = bank["riddles"][:19]
    with pytest.raises(ValueError, match="20 riddles"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_wrong_category_count(bank: dict) -> None:
    """All 20 riddles in different categories → 20 categories, not 10."""
    for i, r in enumerate(bank["riddles"]):
        r["category"] = f"cat_{i}"
    with pytest.raises(ValueError, match="10 categories"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_uneven_category_distribution(bank: dict) -> None:
    """3 riddles in one category, 1 in another → not 2 each."""
    bank["riddles"][0]["category"] = "logic"
    bank["riddles"][1]["category"] = "logic"
    bank["riddles"][2]["category"] = "logic"  # 3 in "logic"
    bank["riddles"][3]["category"] = "wordplay"  # only 1 in "wordplay"
    with pytest.raises(ValueError, match="2 riddles"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_duplicate_riddle_ids(bank: dict) -> None:
    bank["riddles"][1]["id"] = bank["riddles"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate riddle id"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_wrong_option_count(bank: dict) -> None:
    bank["riddles"][0]["options"] = bank["riddles"][0]["options"][:4]
    with pytest.raises(ValueError, match="5 options"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_correct_option_not_in_options(bank: dict) -> None:
    bank["riddles"][0]["correct_option_id"] = "nonexistent"
    with pytest.raises(ValueError, match="does not match"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_duplicate_option_ids(bank: dict) -> None:
    bank["riddles"][0]["options"][1]["option_id"] = bank["riddles"][0]["options"][0]["option_id"]
    with pytest.raises(ValueError, match="duplicate option_id"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_empty_proof(bank: dict) -> None:
    bank["riddles"][0]["proof"] = "  "
    with pytest.raises(ValueError, match="proof"):
        probe.validate_bank(bank)


def test_validate_bank_rejects_invalid_difficulty(bank: dict) -> None:
    bank["riddles"][0]["difficulty"] = "trivial"
    with pytest.raises(ValueError, match="difficulty"):
        probe.validate_bank(bank)


def test_load_and_validate_bank_returns_sha256(bank_file: Path) -> None:
    bank, sha = probe.load_and_validate_bank(bank_file)
    assert len(bank["riddles"]) == probe.RIDDLES_PER_FORM
    assert len(sha) == 64  # SHA-256 hex
    import hashlib
    assert sha == hashlib.sha256(bank_file.read_bytes()).hexdigest()


def test_load_and_validate_bank_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(probe.ProbeError, match="not found"):
        probe.load_and_validate_bank(tmp_path / "nonexistent.json")


def test_load_and_validate_bank_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(probe.ProbeError, match="not valid JSON"):
        probe.load_and_validate_bank(p)


# ─── form generation: determinism ─────────────────────────────────────


def test_generate_forms_deterministic_same_seed(bank: dict) -> None:
    f1 = probe.generate_forms(bank, seed=42)
    f2 = probe.generate_forms(bank, seed=42)
    assert f1 == f2


def test_generate_forms_different_seed_changes_order(bank: dict) -> None:
    f1 = probe.generate_forms(bank, seed=42)
    f2 = probe.generate_forms(bank, seed=99)
    # At least one form should have a different riddle order.
    orders1 = [f["riddle_order"] for f in f1]
    orders2 = [f["riddle_order"] for f in f2]
    assert orders1 != orders2


def test_generate_forms_default_seed_matches_constant(bank: dict) -> None:
    f1 = probe.generate_forms(bank)
    f2 = probe.generate_forms(bank, seed=probe.DEFAULT_SEED)
    assert f1 == f2


def test_generate_forms_rejects_zero_forms(bank: dict) -> None:
    with pytest.raises(ValueError, match="n_forms"):
        probe.generate_forms(bank, n_forms=0)


def test_generate_forms_rejects_too_many_forms(bank: dict) -> None:
    with pytest.raises(ValueError, match="n_forms"):
        probe.generate_forms(bank, n_forms=6)


def test_generate_forms_partial_count(bank: dict) -> None:
    """n_forms=3 produces exactly 3 forms with valid structure."""
    forms = probe.generate_forms(bank, n_forms=3)
    assert len(forms) == 3
    for f in forms:
        assert len(f["displayed"]) == probe.RIDDLES_PER_FORM
        assert len(f["correct_letters"]) == probe.RIDDLES_PER_FORM


# ─── form generation: frozen shape ────────────────────────────────────


def test_form_shape_keys(forms: list[dict]) -> None:
    for f in forms:
        assert set(f.keys()) >= {
            "form_index", "riddle_order", "option_permutations",
            "displayed", "correct_letters", "sha256",
        }


def test_form_riddle_order_is_permutation(forms: list[dict]) -> None:
    for f in forms:
        order = f["riddle_order"]
        assert sorted(order) == list(range(probe.RIDDLES_PER_FORM))


def test_form_displayed_has_20_riddles(forms: list[dict]) -> None:
    for f in forms:
        assert len(f["displayed"]) == probe.RIDDLES_PER_FORM
        for r in f["displayed"]:
            assert len(r["options"]) == probe.OPTIONS_PER_RIDDLE
            # Numbers 1..20 in display order.
            assert r["number"] >= 1 and r["number"] <= probe.RIDDLES_PER_FORM
            # Letters A-E exactly once.
            letters = [o["letter"] for o in r["options"]]
            assert sorted(letters) == list(probe.DISPLAY_LETTERS)


def test_form_sha256_is_64_hex(forms: list[dict]) -> None:
    for f in forms:
        assert len(f["sha256"]) == 64
        int(f["sha256"], 16)  # valid hex


def test_form_hashes_distinct_across_forms(forms: list[dict]) -> None:
    hashes = [f["sha256"] for f in forms]
    assert len(set(hashes)) == len(hashes)


# ─── answer-position Latin square ─────────────────────────────────────


def test_every_semantic_option_occupies_every_letter_once(forms: list[dict]) -> None:
    """Across 5 forms, each semantic option index occupies each A-E position
    exactly once for every riddle (cyclic rotation property)."""
    assert len(forms) == probe.OPTIONS_PER_RIDDLE
    n_riddles = probe.RIDDLES_PER_FORM

    for bank_idx in range(n_riddles):
        # For each form, find which display position this bank_idx appears at,
        # and build a mapping: form -> (display_position -> semantic_index).
        # Actually, option_permutations is indexed by display position in the
        # form's riddle_order. We need to find where bank_idx appears in each
        # form's riddle_order, then check the permutation at that position.
        for form in forms:
            riddle_order = form["riddle_order"]
            display_idx = riddle_order.index(bank_idx)
            perm = form["option_permutations"][display_idx]
            # perm maps display position (0=A..4=E) to semantic option index.
            # It should be a permutation of 0..4.
            assert sorted(perm) == list(range(probe.OPTIONS_PER_RIDDLE))

    # Now verify the Latin-square: for each riddle, across 5 forms, each
    # semantic option index appears at each display position exactly once.
    for bank_idx in range(n_riddles):
        position_occupancy: dict[int, list[int]] = {
            pos: [] for pos in range(probe.OPTIONS_PER_RIDDLE)
        }
        for form in forms:
            display_idx = form["riddle_order"].index(bank_idx)
            perm = form["option_permutations"][display_idx]
            for pos, sem_idx in enumerate(perm):
                position_occupancy[pos].append(sem_idx)
        # Each position should have seen each semantic index exactly once.
        for pos in range(probe.OPTIONS_PER_RIDDLE):
            assert sorted(position_occupancy[pos]) == list(
                range(probe.OPTIONS_PER_RIDDLE)
            ), (
                f"riddle {bank_idx}: position {pos} did not see all semantic "
                f"options exactly once across forms"
            )


def test_correct_letter_varies_across_forms(forms: list[dict]) -> None:
    """The correct answer should not always be at the same letter — it must
    occupy different positions across forms (answer-position resistance)."""
    for bank_idx in range(probe.RIDDLES_PER_FORM):
        letters_for_riddle = []
        for form in forms:
            display_idx = form["riddle_order"].index(bank_idx)
            letters_for_riddle.append(form["correct_letters"][display_idx])
        # With 5 forms and cyclic rotation, all 5 letters should appear.
        assert len(set(letters_for_riddle)) == probe.OPTIONS_PER_RIDDLE


# ─── riddle-order shuffle ─────────────────────────────────────────────


def test_riddle_order_differs_across_forms(forms: list[dict]) -> None:
    """Each form should have a different riddle order (deterministic shuffle)."""
    orders = [tuple(f["riddle_order"]) for f in forms]
    assert len(set(orders)) == len(forms)


def test_riddle_order_deterministic_same_seed(bank: dict) -> None:
    f1 = probe.generate_forms(bank, seed=100)
    f2 = probe.generate_forms(bank, seed=100)
    for a, b in zip(f1, f2, strict=True):
        assert a["riddle_order"] == b["riddle_order"]


def test_displayed_numbers_are_1_to_20(forms: list[dict]) -> None:
    for form in forms:
        numbers = [r["number"] for r in form["displayed"]]
        assert sorted(numbers) == list(range(1, probe.RIDDLES_PER_FORM + 1))


# ─── render_prompt ────────────────────────────────────────────────────


def test_render_prompt_includes_trial_id_and_all_riddles(forms: list[dict]) -> None:
    trial_id = "test-trial-42"
    prompt = probe.render_prompt(forms[0], trial_id)
    assert trial_id in prompt
    assert "Riddle 1:" in prompt
    assert f"Riddle {probe.RIDDLES_PER_FORM}:" in prompt
    # Should contain option letters.
    for letter in probe.DISPLAY_LETTERS:
        assert letter in prompt


def test_render_prompt_does_not_leak_correct_letters(forms: list[dict]) -> None:
    """The rendered prompt must not contain the correct_letters field or
    any proof text — only displayed options."""
    prompt = probe.render_prompt(forms[0], "trial-x")
    assert "correct_letters" not in prompt
    assert "correct_option_id" not in prompt
    # Proof text from the bank should not appear.
    bank = _make_bank()
    for riddle in bank["riddles"]:
        assert riddle["proof"] not in prompt


# ─── scorer correctness ───────────────────────────────────────────────


def test_score_submission_all_correct(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    result = probe.score_submission(sub, form)
    assert result["correct_count"] == probe.RIDDLES_PER_FORM
    assert result["accuracy"] == 1.0
    assert result["per_riddle"] == [1] * probe.RIDDLES_PER_FORM
    assert result["structural_issues"] == []


def test_score_submission_all_wrong(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_wrong_submission(form)
    result = probe.score_submission(sub, form)
    assert result["correct_count"] == 0
    assert result["accuracy"] == 0.0
    assert result["per_riddle"] == [0] * probe.RIDDLES_PER_FORM
    # Structural issues should be empty — wrong answers are structurally valid.
    assert result["structural_issues"] == []


def test_score_submission_partial(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    # Flip riddle 5 to wrong.
    sub[4] = {"number": 5, "choice": "Z", "explanation": "bad"}
    # Actually "Z" is not a valid letter, so it scores 0 but isn't structural.
    # Use a valid wrong letter instead.
    correct_5 = form["correct_letters"][4]
    idx = probe.DISPLAY_LETTERS.index(correct_5)
    wrong = probe.DISPLAY_LETTERS[(idx + 1) % 5]
    sub[4] = {"number": 5, "choice": wrong, "explanation": "no"}
    result = probe.score_submission(sub, form)
    assert result["correct_count"] == probe.RIDDLES_PER_FORM - 1
    assert result["per_riddle"][4] == 0
    assert result["per_riddle"][0] == 1


def test_score_submission_case_insensitive(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    sub[0]["choice"] = sub[0]["choice"].lower()
    result = probe.score_submission(sub, form)
    assert result["per_riddle"][0] == 1


def test_score_submission_never_raises_on_wrong(forms: list[dict]) -> None:
    """Wrong answers must be scored 0, not raise."""
    form = forms[0]
    sub = _all_wrong_submission(form)
    result = probe.score_submission(sub, form)
    assert result["accuracy"] == 0.0


# ─── wrong answers accepted structurally but scored wrong ─────────────


def test_wrong_submission_has_no_structural_issues(forms: list[dict]) -> None:
    """A structurally valid but entirely wrong submission has no structural
    issues — it is accepted and scored later."""
    form = forms[0]
    sub = _all_wrong_submission(form)
    result = probe.score_submission(sub, form)
    assert result["structural_issues"] == []
    assert result["correct_count"] == 0


# ─── duplicate/missing coverage format errors ─────────────────────────


def test_score_submission_duplicate_number(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    sub[1] = {"number": 1, "choice": "A", "explanation": "dup"}
    result = probe.score_submission(sub, form)
    assert any("Duplicate" in s for s in result["structural_issues"])


def test_score_submission_missing_number(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    # Remove riddle 10 by setting its number to 21 (out of range → dropped).
    sub[9] = {"number": 21, "choice": "A", "explanation": "bad"}
    result = probe.score_submission(sub, form)
    assert any("Missing" in s for s in result["structural_issues"])


def test_score_submission_invalid_choice_scores_zero(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    sub[0] = {"number": 1, "choice": "Z", "explanation": "bad"}
    result = probe.score_submission(sub, form)
    assert result["per_riddle"][0] == 0


def test_score_submission_non_dict_record_flagged(forms: list[dict]) -> None:
    form = forms[0]
    sub = _all_correct_submission(form)
    sub[0] = "not a dict"
    result = probe.score_submission(sub, form)
    assert any("Non-dict" in s for s in result["structural_issues"])


# ─── guessing-corrected score ─────────────────────────────────────────


def test_guessing_corrected_perfect_score() -> None:
    assert probe.guessing_corrected_score(1.0) == 1.0


def test_guessing_corrected_pure_guessing() -> None:
    """20% accuracy → 0.0 guessing-corrected."""
    assert probe.guessing_corrected_score(0.2) == 0.0


def test_guessing_corrected_below_guessing_clamped() -> None:
    """Below 20% accuracy is clamped to 0.0."""
    assert probe.guessing_corrected_score(0.0) == 0.0


def test_guessing_corrected_midpoint() -> None:
    """60% accuracy → (0.6 - 0.2) / 0.8 = 0.5."""
    assert probe.guessing_corrected_score(0.6) == pytest.approx(0.5)


# ─── tool argument validation (structure/coverage only) ───────────────


def test_validate_tool_args_accepts_valid(forms: list[dict]) -> None:
    form = forms[0]
    trial_id = "trial-test"
    answers = _all_correct_submission(form)
    args = {"trial_id": trial_id, "answers": answers}
    valid, detail, ans = probe.validate_tool_args(args, trial_id)
    assert valid is True
    assert detail == ""
    assert ans == answers


def test_validate_tool_args_rejects_wrong_trial_id(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)
    args = {"trial_id": "wrong", "answers": answers}
    valid, detail, ans = probe.validate_tool_args(args, "correct")
    assert valid is False
    assert "trial_id" in detail
    assert ans is None


def test_validate_tool_args_rejects_wrong_count(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)[:19]
    args = {"trial_id": "t", "answers": answers}
    valid, detail, _ = probe.validate_tool_args(args, "t")
    assert valid is False
    assert "20" in detail


def test_validate_tool_args_rejects_duplicate_number(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)
    answers[1] = {"number": 1, "choice": "A", "explanation": "dup"}
    args = {"trial_id": "t", "answers": answers}
    valid, detail, _ = probe.validate_tool_args(args, "t")
    assert valid is False
    assert "duplicate" in detail.lower()


def test_validate_tool_args_rejects_missing_number(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)
    answers[9] = {"number": 21, "choice": "A", "explanation": "bad"}
    args = {"trial_id": "t", "answers": answers}
    valid, detail, _ = probe.validate_tool_args(args, "t")
    assert valid is False
    assert "missing" in detail.lower() or "invalid" in detail.lower()


def test_validate_tool_args_rejects_invalid_choice(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)
    answers[0] = {"number": 1, "choice": "Z", "explanation": "bad"}
    args = {"trial_id": "t", "answers": answers}
    valid, detail, _ = probe.validate_tool_args(args, "t")
    assert valid is False
    assert "choice" in detail.lower()


def test_validate_tool_args_rejects_missing_explanation(forms: list[dict]) -> None:
    form = forms[0]
    answers = _all_correct_submission(form)
    answers[0] = {"number": 1, "choice": "A", "explanation": ""}
    args = {"trial_id": "t", "answers": answers}
    valid, detail, _ = probe.validate_tool_args(args, "t")
    assert valid is False
    assert "explanation" in detail.lower()


def test_validate_tool_args_rejects_non_dict_args() -> None:
    valid, _, _ = probe.validate_tool_args("not a dict", "t")
    assert valid is False


def test_validate_tool_args_never_checks_correctness(forms: list[dict]) -> None:
    """A completely wrong but structurally valid submission passes validation."""
    form = forms[0]
    answers = _all_wrong_submission(form)
    args = {"trial_id": "t", "answers": answers}
    valid, _, _ = probe.validate_tool_args(args, "t")
    assert valid is True


# ─── OMP argument repair detection ────────────────────────────────────


def test_strip_harness_i_only_strips_i() -> None:
    stripped = probe._strip_harness_i({"i": "intent", "trial_id": "T", "answers": []})
    assert "i" not in stripped
    assert stripped["trial_id"] == "T"
    assert probe._strip_harness_i("nope") == "nope"


def test_detect_repair_only_i_differs_is_not_repair() -> None:
    raw = {"trial_id": "T", "i": "intent", "answers": []}
    host = {"trial_id": "T", "i": "other", "answers": []}
    assert probe._detect_repair(raw, host) is False


def test_detect_repair_semantic_diff_is_repair() -> None:
    raw = {"trial_id": "T", "answers": [{"number": 1, "choice": "A", "explanation": "x"}]}
    host = {"trial_id": "T", "answers": [{"number": 1, "choice": "B", "explanation": "x"}]}
    assert probe._detect_repair(raw, host) is True


def test_detect_repair_identical_args_is_not_repair() -> None:
    raw = {"trial_id": "T", "answers": []}
    host = {"trial_id": "T", "answers": []}
    assert probe._detect_repair(raw, host) is False


# ─── schedule / model rotation ────────────────────────────────────────


def test_schedule_balanced_rotation() -> None:
    models = ["m1", "m2", "m3"]
    schedule = probe._build_schedule(models, 3, seed=42)
    # 3 forms × 3 models = 9 sessions.
    assert len(schedule) == 9
    # Form 0: m1, m2, m3 (offset 0).
    assert [s["model"] for s in schedule[0:3]] == ["m1", "m2", "m3"]
    # Form 1: rotated by 1 → m2, m3, m1.
    assert [s["model"] for s in schedule[3:6]] == ["m2", "m3", "m1"]
    # Form 2: rotated by 2 → m3, m1, m2.
    assert [s["model"] for s in schedule[6:9]] == ["m3", "m1", "m2"]


def test_schedule_form_indices_correct() -> None:
    schedule = probe._build_schedule(["m1", "m2"], 5, seed=1)
    form_indices = [s["form_index"] for s in schedule]
    assert form_indices == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_schedule_total_equals_models_times_forms() -> None:
    schedule = probe._build_schedule(["a", "b", "c", "d"], 5, seed=7)
    assert len(schedule) == 4 * 5


# ─── model result aggregation ─────────────────────────────────────────


def _make_trial_result(
    model: str,
    form_index: int,
    form: dict,
    correct: bool = True,
    accepted: bool = True,
    probe_error: bool = False,
) -> probe.TrialResult:
    """Build a minimal TrialResult for aggregation tests."""
    if correct:
        sub = _all_correct_submission(form)
    else:
        sub = _all_wrong_submission(form)
    score = probe.score_submission(sub, form) if accepted else None
    return probe.TrialResult(
        trial_id=f"test-{model}-{form_index}",
        model=model,
        form_index=form_index,
        accepted=accepted,
        submission=sub if accepted else None,
        raw_tool_args={"trial_id": "t", "answers": sub} if accepted else None,
        effective_tool_args={"trial_id": "t", "answers": sub} if accepted else None,
        omp_argument_repair=False,
        format_attempts=0 if accepted else 1,
        format_errors=[] if accepted else ["bad format"],
        score=score,
        ttft=100.0 + form_index,
        duration=1000.0 + form_index,
        usage={"input": 500, "output": 200, "totalTokens": 700},
        wall_time_seconds=10.0 + form_index,
        turn_count=1,
        message_count=1,
        auto_retry_events=0,
        auto_compaction_events=0,
        diagnostic_events=[],
        stderr_tail=[],
        non_json_rpc_lines=[],
        final_assistant_text="DONE" if accepted else "",
        process_exit_code=0,
        omp_model_resolved=f"resolved-{model}",
        probe_error=probe_error,
        failure_category=None if accepted else "tool_format_error",
        baseline_context_usage=None,
        final_context_usage=None,
        messages=[],
        attempts=[],
    )


def test_model_result_per_form_scores(bank: dict, forms: list[dict]) -> None:
    trials = [
        _make_trial_result("m1", 0, forms[0], correct=True),
        _make_trial_result("m1", 1, forms[1], correct=False),
    ]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert len(mr.per_form_scores) == 2
    assert mr.per_form_scores[0]["correct_count"] == 20
    assert mr.per_form_scores[1]["correct_count"] == 0


def test_model_result_per_riddle_scores_0_to_5(bank: dict, forms: list[dict]) -> None:
    """Per-riddle scores range 0..5 across 5 forms."""
    trials = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert len(mr.per_riddle_scores) == probe.RIDDLES_PER_FORM
    assert all(s == 5 for s in mr.per_riddle_scores)


def test_model_result_majority_correct(bank: dict, forms: list[dict]) -> None:
    """All correct → all 20 riddles have >=3/5 → majority_correct = 20."""
    trials = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.majority_correct == probe.RIDDLES_PER_FORM


def test_model_result_stable_correct(bank: dict, forms: list[dict]) -> None:
    """All correct → all 20 riddles have 5/5 → stable_correct = 20."""
    trials = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.stable_correct == probe.RIDDLES_PER_FORM


def test_model_result_majority_zero_when_all_wrong(bank: dict, forms: list[dict]) -> None:
    trials = [_make_trial_result("m1", fi, forms[fi], correct=False) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.majority_correct == 0
    assert mr.stable_correct == 0


def test_model_result_category_scores(bank: dict, forms: list[dict]) -> None:
    """Category scores should cover all 10 categories with correct/total."""
    trials = [_make_trial_result("m1", 0, forms[0], correct=True)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert len(mr.category_scores) == 10
    for cat_data in mr.category_scores.values():
        assert "correct" in cat_data
        assert "total" in cat_data
        assert "accuracy" in cat_data


def test_model_result_answer_position_scores(bank: dict, forms: list[dict]) -> None:
    """Answer-position scores should cover A-E."""
    trials = [_make_trial_result("m1", 0, forms[0], correct=True)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert set(mr.answer_position_scores.keys()) == set(probe.DISPLAY_LETTERS)
    for letter_data in mr.answer_position_scores.values():
        assert "correct" in letter_data
        assert "total" in letter_data


def test_model_result_primary_accuracy(bank: dict, forms: list[dict]) -> None:
    """Primary accuracy = total correct / total decisions (100)."""
    trials = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.primary_accuracy == 1.0
    assert mr.guessing_corrected == 1.0


def test_model_result_primary_accuracy_half_correct(bank: dict, forms: list[dict]) -> None:
    """2/5 forms correct → 40/100 = 0.4 accuracy."""
    trials = [
        _make_trial_result("m1", 0, forms[0], correct=True),
        _make_trial_result("m1", 1, forms[1], correct=True),
        _make_trial_result("m1", 2, forms[2], correct=False),
        _make_trial_result("m1", 3, forms[3], correct=False),
        _make_trial_result("m1", 4, forms[4], correct=False),
    ]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.primary_accuracy == pytest.approx(0.4)


def test_model_result_tool_format_failures(bank: dict, forms: list[dict]) -> None:
    """tool_format_failures counts trials where accepted=False."""
    trials = [
        _make_trial_result("m1", 0, forms[0], accepted=True),
        _make_trial_result("m1", 1, forms[1], accepted=False),
    ]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.tool_format_failures == 1


def test_model_result_mean_wall_time(bank: dict, forms: list[dict]) -> None:
    trials = [_make_trial_result("m1", fi, forms[fi]) for fi in range(3)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    expected = round((10.0 + 11.0 + 12.0) / 3, 3)
    assert mr.mean_wall_time_seconds == expected


def test_model_result_mean_ttft(bank: dict, forms: list[dict]) -> None:
    trials = [_make_trial_result("m1", fi, forms[fi]) for fi in range(3)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    expected = round((100.0 + 101.0 + 102.0) / 3, 3)
    assert mr.mean_ttft == expected


def test_model_result_total_usage(bank: dict, forms: list[dict]) -> None:
    trials = [_make_trial_result("m1", fi, forms[fi]) for fi in range(5)]
    mr = probe._compute_model_result("m1", trials, forms, bank)
    assert mr.total_usage["input"] == 500 * 5
    assert mr.total_usage["output"] == 200 * 5


# ─── paired comparison determinism ────────────────────────────────────


def test_paired_winloss_deterministic(forms: list[dict]) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=False) for fi in range(5)]
    wl1 = probe._paired_winloss("m1", "m2", trials_a, trials_b, forms)
    wl2 = probe._paired_winloss("m1", "m2", trials_a, trials_b, forms)
    assert wl1 == wl2


def test_paired_winloss_all_wins_when_a_correct_b_wrong(
    forms: list[dict],
) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=False) for fi in range(5)]
    wl = probe._paired_winloss("m1", "m2", trials_a, trials_b, forms)
    assert wl["wins"] == probe.RIDDLES_PER_FORM
    assert wl["losses"] == 0
    assert wl["ties"] == 0


def test_paired_winloss_missing_submission_counts_as_wrong(
    forms: list[dict],
) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [
        _make_trial_result("m2", fi, forms[fi], accepted=False) for fi in range(5)
    ]
    wl = probe._paired_winloss("m1", "m2", trials_a, trials_b, forms)
    assert wl["wins"] == probe.RIDDLES_PER_FORM
    assert wl["losses"] == 0
    assert wl["ties"] == 0
    assert wl["per_riddle_diff"] == [5] * probe.RIDDLES_PER_FORM


def test_paired_winloss_all_ties_when_both_correct(
    forms: list[dict],
) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=True) for fi in range(5)]
    wl = probe._paired_winloss("m1", "m2", trials_a, trials_b, forms)
    assert wl["wins"] == 0
    assert wl["losses"] == 0
    assert wl["ties"] == probe.RIDDLES_PER_FORM


def test_cluster_bootstrap_ci_deterministic() -> None:
    diffs = [1, -1, 0, 2, -1, 0, 1, 1, -1, 0] * 2
    ci1 = probe._cluster_bootstrap_ci(diffs, seed=42)
    ci2 = probe._cluster_bootstrap_ci(diffs, seed=42)
    assert ci1 == ci2


def test_cluster_bootstrap_ci_different_seed_changes_interval() -> None:
    diffs = [1, -1, 0, 2, -1, 0, 1, 1, -1, 0] * 2
    ci1 = probe._cluster_bootstrap_ci(diffs, seed=42)
    ci2 = probe._cluster_bootstrap_ci(diffs, seed=99)
    # CI bounds should differ (very high probability with 10000 bootstrap).
    assert ci1 != ci2


def test_cluster_bootstrap_ci_empty_diffs() -> None:
    ci = probe._cluster_bootstrap_ci([])
    assert ci["mean_diff"] == 0.0
    assert ci["n_bootstrap"] == 0


def test_compute_paired_results_deterministic(
    bank: dict, forms: list[dict]
) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=False) for fi in range(5)]
    mr_a = probe._compute_model_result("m1", trials_a, forms, bank)
    mr_b = probe._compute_model_result("m2", trials_b, forms, bank)
    p1, b1 = probe._compute_paired_results([mr_a, mr_b], forms, seed=42)
    p2, b2 = probe._compute_paired_results([mr_a, mr_b], forms, seed=42)
    assert p1 == p2
    assert b1 == b2


def test_compute_paired_results_normalizes_to_accuracy_difference(
    bank: dict, forms: list[dict]
) -> None:
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=False) for fi in range(5)]
    mr_a = probe._compute_model_result("m1", trials_a, forms, bank)
    mr_b = probe._compute_model_result("m2", trials_b, forms, bank)
    _, intervals = probe._compute_paired_results([mr_a, mr_b], forms, seed=42)
    interval = intervals["m1__vs__m2"]
    assert interval["mean_diff"] == 1.0
    assert interval["ci_lower"] == 1.0
    assert interval["ci_upper"] == 1.0


# ─── ranking ──────────────────────────────────────────────────────────


def test_ranking_omitted_for_unequal_counts(bank: dict, forms: list[dict]) -> None:
    """m1 has 5 trials, m2 has 3 → ranking omitted."""
    trials_a = [_make_trial_result("m1", fi, forms[fi]) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi]) for fi in range(3)]
    mr_a = probe._compute_model_result("m1", trials_a, forms, bank)
    mr_b = probe._compute_model_result("m2", trials_b, forms, bank)
    ranking, omitted = probe._compute_ranking([mr_a, mr_b])
    assert omitted is True
    assert ranking == []


def test_ranking_omitted_for_probe_error(bank: dict, forms: list[dict]) -> None:
    """Any probe_error trial → ranking omitted."""
    trials_a = [_make_trial_result("m1", fi, forms[fi]) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi]) for fi in range(5)]
    trials_b[0] = _make_trial_result(
        "m2", 0, forms[0], probe_error=True
    )
    mr_a = probe._compute_model_result("m1", trials_a, forms, bank)
    mr_b = probe._compute_model_result("m2", trials_b, forms, bank)
    ranking, omitted = probe._compute_ranking([mr_a, mr_b])
    assert omitted is True


def test_ranking_omitted_for_empty_results() -> None:
    ranking, omitted = probe._compute_ranking([])
    assert omitted is True
    assert ranking == []


def test_ranking_deterministic_for_equal_counts(
    bank: dict, forms: list[dict]
) -> None:
    """Equal counts, no probe errors → ranking by primary_accuracy desc."""
    trials_a = [_make_trial_result("m1", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("m2", fi, forms[fi], correct=False) for fi in range(5)]
    mr_a = probe._compute_model_result("m1", trials_a, forms, bank)
    mr_b = probe._compute_model_result("m2", trials_b, forms, bank)
    ranking, omitted = probe._compute_ranking([mr_a, mr_b])
    assert omitted is False
    assert ranking[0]["model"] == "m1"
    assert ranking[0]["primary_accuracy"] == 1.0
    assert ranking[1]["model"] == "m2"


def test_ranking_tiebreak_by_model_name(bank: dict, forms: list[dict]) -> None:
    """Full tie on all metrics → model name ascending wins."""
    trials_a = [_make_trial_result("aaa", fi, forms[fi], correct=True) for fi in range(5)]
    trials_b = [_make_trial_result("bbb", fi, forms[fi], correct=True) for fi in range(5)]
    mr_a = probe._compute_model_result("aaa", trials_a, forms, bank)
    mr_b = probe._compute_model_result("bbb", trials_b, forms, bank)
    ranking, omitted = probe._compute_ranking([mr_b, mr_a])  # pass in reverse
    assert omitted is False
    assert [r["model"] for r in ranking] == ["aaa", "bbb"]


# ─── artifact building ────────────────────────────────────────────────


def test_build_artifact_partial_progress(
    bank: dict, forms: list[dict], tmp_path: Path
) -> None:
    """build_artifact with empty trials produces progress 0/total."""
    schedule = probe._build_schedule(["m1", "m2"], 5, seed=42)
    artifact = probe.build_artifact(
        config={"models": ["m1", "m2"], "forms": 5, "seed": 42},
        omp_path=tmp_path / "omp",
        omp_version="test-1.0",
        system_prompt=probe.build_system_prompt(),
        tool_definition=probe.build_tool_definition(),
        bank=bank,
        bank_sha="abc123",
        bank_path="/test/bank.json",
        forms=forms,
        schedule=schedule,
        trials=[],
        model_results=[],
        paired_winloss={},
        bootstrap_intervals={},
        infra_probe_error=False,
    )
    assert artifact["schema_version"] == probe.SCHEMA_VERSION
    assert artifact["progress"] == {"completed": 0, "total": len(schedule)}
    assert artifact["ranking_omitted"] is True  # no model results
    assert len(artifact["form_generator"]["form_hashes"]) == 5


def test_build_artifact_strips_correct_letters_from_forms(
    bank: dict, forms: list[dict], tmp_path: Path
) -> None:
    """Public form data must not include correct_letters (scoring secret)."""
    artifact = probe.build_artifact(
        config={"models": ["m1"], "forms": 5, "seed": 42},
        omp_path=tmp_path / "omp",
        omp_version="test",
        system_prompt=probe.build_system_prompt(),
        tool_definition=probe.build_tool_definition(),
        bank=bank,
        bank_sha="abc",
        bank_path="/test/bank.json",
        forms=forms,
        schedule=[],
        trials=[],
        model_results=[],
        paired_winloss={},
        bootstrap_intervals={},
        infra_probe_error=False,
    )
    for f in artifact["forms"]:
        assert "correct_letters" not in f
        assert "displayed" not in f  # only structural form data is public


def test_build_artifact_includes_hashes(
    bank: dict, forms: list[dict], tmp_path: Path
) -> None:
    """Artifact must include probe_sha256, bank sha256, system_prompt sha256,
    tool_schema sha256, and form hashes."""
    artifact = probe.build_artifact(
        config={"models": ["m1"], "forms": 5, "seed": 42},
        omp_path=tmp_path / "omp",
        omp_version="test",
        system_prompt=probe.build_system_prompt(),
        tool_definition=probe.build_tool_definition(),
        bank=bank,
        bank_sha="abc123def456",
        bank_path="/test/bank.json",
        forms=forms,
        schedule=[],
        trials=[],
        model_results=[],
        paired_winloss={},
        bootstrap_intervals={},
        infra_probe_error=False,
    )
    assert len(artifact["probe_sha256"]) == 64
    assert artifact["bank"]["sha256"] == "abc123def456"
    assert len(artifact["system_prompt"]["sha256"]) == 64
    assert len(artifact["tool_schema"]["sha256"]) == 64
    assert len(artifact["form_generator"]["form_hashes"]) == 5


# ─── atomic artifact output ───────────────────────────────────────────


def test_write_artifact_atomic_no_tmp_residue(tmp_path: Path) -> None:
    artifact = {"schema_version": probe.SCHEMA_VERSION, "trials": [], "progress": {}}
    out = tmp_path / "artifact.json"
    probe.write_artifact(artifact, str(out))
    assert json.loads(out.read_text()) == artifact
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_write_artifact_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "artifact.json"
    probe.write_artifact({"ok": True}, str(out))
    assert out.exists()
    assert json.loads(out.read_text()) == {"ok": True}


# ─── CLI argument parsing ─────────────────────────────────────────────


def test_parse_args_rejects_duplicate_models() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["m1", "m1", "--output", "/tmp/out.json"])


def test_parse_args_rejects_blank_model() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["", "m2", "--output", "/tmp/out.json"])


def test_parse_args_rejects_forms_out_of_range() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["m1", "--forms", "0", "--output", "/tmp/o.json"])
    with pytest.raises(SystemExit):
        probe.parse_args(["m1", "--forms", "6", "--output", "/tmp/o.json"])


def test_parse_args_rejects_zero_max_format_attempts() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["m1", "--max-format-attempts", "0", "--output", "/tmp/o.json"])


def test_parse_args_rejects_zero_timeout() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["m1", "--trial-timeout-seconds", "0", "--output", "/tmp/o.json"])


def test_parse_args_requires_output() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["m1"])


def test_parse_args_defaults() -> None:
    args = probe.parse_args(["m1", "--output", "/tmp/o.json"])
    assert args.forms == probe.DEFAULT_FORMS
    assert args.seed == probe.DEFAULT_SEED
    assert args.max_format_attempts == probe.DEFAULT_MAX_FORMAT_ATTEMPTS
    assert args.trial_timeout_seconds == probe.DEFAULT_TRIAL_TIMEOUT_SECONDS
    assert args.omp_bin == probe.DEFAULT_OMP_BIN


# ─── tool schema / system prompt ──────────────────────────────────────


def test_build_tool_schema_has_20_answers() -> None:
    schema = probe.build_tool_schema()
    assert schema["properties"]["answers"]["minItems"] == probe.RIDDLES_PER_FORM
    assert schema["properties"]["answers"]["maxItems"] == probe.RIDDLES_PER_FORM


def test_build_tool_schema_choice_enum_is_a_to_e() -> None:
    schema = probe.build_tool_schema()
    enum = schema["properties"]["answers"]["items"]["properties"]["choice"]["enum"]
    assert enum == list(probe.DISPLAY_LETTERS)


def test_build_tool_definition_name_matches() -> None:
    td = probe.build_tool_definition()
    assert td["name"] == probe.TOOL_NAME
    assert "parameters" in td


def test_build_system_prompt_is_constant() -> None:
    assert probe.build_system_prompt() == probe.SYSTEM_PROMPT
    assert "submit_riddle_answers" in probe.SYSTEM_PROMPT


# ─── fake RPC: success path ───────────────────────────────────────────

_FAKE_OMP_SUCCESS = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-riddle 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-riddle-model",
                           "contextUsage": {"tokens": 2000}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            # Parse trial_id and correct letters from the prompt.
            # The prompt contains "Trial ID: <trial_id>" and riddle options.
            # We build a correct submission by reading the displayed options.
            # For simplicity, emit a tool call with all "A" choices.
            # This is structurally valid (20 records, numbers 1..20, valid
            # letters, explanations) but may be wrong — that's fine for
            # testing structural acceptance.
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            answers = []
            for n in range(1, 21):
                answers.append({"number": n, "choice": "A",
                                "explanation": "guess A"})
            args = {"trial_id": trial_id, "answers": answers}
            # 1. host_tool_call with the submission
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": args})
            # 2. message_end with the toolCall
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": args}]}})
            # 3. agent_end
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            # 4. late prompt ack
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_fake_rpc_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """Full probe run against a fake OMP RPC that submits structurally valid
    answers. The probe must accept the submission and produce a complete
    artifact with scores."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_SUCCESS)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/model",
        "--forms", "2",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    assert artifact["schema_version"] == probe.SCHEMA_VERSION
    assert artifact["progress"] == {"completed": 2, "total": 2}
    assert len(artifact["trials"]) == 2

    for trial in artifact["trials"]:
        assert trial["probe_error"] is False
        assert trial["accepted"] is True
        assert trial["submission"] is not None
        assert trial["score"] is not None
        assert trial["omp_model_resolved"] == "fake-riddle-model"
        assert trial["final_assistant_text"] == "DONE"

    # Model results present.
    assert len(artifact["model_results"]) == 1
    mr = artifact["model_results"][0]
    assert mr["model"] == "fake/model"
    assert len(mr["per_form_scores"]) == 2

    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


# ─── fake RPC: wrong answers accepted structurally ────────────────────

_FAKE_OMP_WRONG = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-wrong 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-wrong", "contextUsage": {"tokens": 100}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            # Submit all "B" — structurally valid but likely wrong.
            answers = [{"number": n, "choice": "B", "explanation": "always B"}
                       for n in range(1, 21)]
            args = {"trial_id": trial_id, "answers": answers}
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": args})
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": args}]}})
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_wrong_answers_accepted_and_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """A structurally valid but wrong submission is accepted (no correctness
    feedback to the model) and scored later."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_WRONG)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/wrong",
        "--forms", "1",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    trial = artifact["trials"][0]
    assert trial["accepted"] is True
    assert trial["probe_error"] is False
    # The submission was accepted despite being wrong.
    assert trial["score"]["correct_count"] < probe.RIDDLES_PER_FORM
    # The tool result sent to the model was ACCEPTED, not an error.
    # (We can't directly check the tool result, but accepted=True proves it.)


# ─── fake RPC: malformed submission then repair ───────────────────────

_FAKE_OMP_MALFORMED_REPAIR = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-repair 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-repair", "contextUsage": {"tokens": 100}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            # First attempt: only 19 answers (missing number 20).
            bad_answers = [{"number": n, "choice": "A", "explanation": "x"}
                           for n in range(1, 20)]
            bad_args = {"trial_id": trial_id, "answers": bad_answers}
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": bad_args})
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": bad_args}]}})
            # Second attempt: correct structure (all 20).
            good_answers = [{"number": n, "choice": "A", "explanation": "x"}
                            for n in range(1, 21)]
            good_args = {"trial_id": trial_id, "answers": good_answers}
            emit({"type": "host_tool_call", "id": "htc-2", "toolCallId": "tc-2",
                  "toolName": "submit_riddle_answers", "arguments": good_args})
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-2",
                   "name": "submit_riddle_answers", "arguments": good_args}]}})
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_malformed_then_valid_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """A first malformed submission (19 answers) is rejected with a format
    error, then a valid retry is accepted."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_MALFORMED_REPAIR)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/repair",
        "--forms", "1",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    trial = artifact["trials"][0]
    assert trial["accepted"] is True
    assert trial["probe_error"] is False
    # Should have recorded a format error from the first attempt.
    assert len(trial["format_errors"]) >= 1
    assert trial["format_attempts"] >= 1
    # Two tool call attempts.
    assert len(trial["attempts"]) == 2
    # First attempt was a format error, second was accepted.
    cats = [a["category"] for a in trial["attempts"]]
    assert "accepted" in cats
    assert cats[-1] == "accepted"


# ─── fake RPC: setup failure abort ────────────────────────────────────

_FAKE_OMP_SETUP_FAIL = r'''#!/usr/bin/env python3
import sys, json

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-fail 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Emit non-JSON garbage to trigger rpc_protocol_process_error.
        sys.stdout.write("THIS IS NOT JSON GARBAGE\n")
        sys.stdout.flush()
        return 1
    return 0

sys.exit(main())
'''


def test_end_to_end_setup_failure_aborts_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """A non-JSON/setup failure produces a partial artifact, probe_error,
    remaining schedule aborted, and exit code 2."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_SETUP_FAIL)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "m1", "m2",
        "--forms", "2",
        "--seed", "42",
        "--trial-timeout-seconds", "15",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_PROBE_ERROR

    # Partial artifact written.
    assert out.exists()
    artifact = json.loads(out.read_text())
    total = len(artifact["schedule"])
    assert total == 4  # 2 models × 2 forms
    assert artifact["progress"]["completed"] < total

    # The first trial should be a probe error.
    assert len(artifact["trials"]) >= 1
    trial = artifact["trials"][0]
    assert trial["probe_error"] is True

    # Infra probe error flag set.
    assert artifact["infra_probe_error"] is True

    # Ranking omitted.
    assert artifact["ranking_omitted"] is True

    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


# ─── fake RPC: prompt-ack/agent-end race ──────────────────────────────

_FAKE_OMP_RACE = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-race 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-race", "contextUsage": {"tokens": 50}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            answers = [{"number": n, "choice": "C", "explanation": "c"}
                       for n in range(1, 21)]
            args = {"trial_id": trial_id, "answers": answers}
            # 1. host_tool_call BEFORE prompt ack
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": args})
            # 2. message_end
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": args}]}})
            # 3. agent_end BEFORE prompt ack
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            # 4. late prompt ack
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_prompt_ack_agent_end_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """The probe must handle host_tool_call, message_end, and agent_end
    arriving BEFORE the prompt ack — no deadlock, submission accepted."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_RACE)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/race",
        "--forms", "1",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    trial = artifact["trials"][0]
    assert trial["accepted"] is True
    assert trial["probe_error"] is False
    assert trial["final_assistant_text"] == "DONE"
    assert trial["score"] is not None

    # No temp residue.
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


# ─── fake RPC: OMP argument repair capture ────────────────────────────

_FAKE_OMP_ARG_REPAIR = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-argrepair 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-argrepair", "contextUsage": {"tokens": 50}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            # Raw args from the model have an extra "i" key and wrong trial_id.
            raw_args = {"trial_id": "wrong", "i": "intent",
                        "answers": [{"number": n, "choice": "A",
                                     "explanation": "x"} for n in range(1, 21)]}
            # Host-emitted args are repaired by OMP: correct trial_id, no "i".
            host_args = {"trial_id": trial_id,
                         "answers": [{"number": n, "choice": "A",
                                      "explanation": "x"} for n in range(1, 21)]}
            # message_end carries the raw (unrepaired) args.
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": raw_args}]}})
            # host_tool_call carries the repaired args.
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": host_args})
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_omp_argument_repair_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """When OMP repairs the model's raw arguments (different trial_id, extra
    'i' key stripped), the probe captures omp_argument_repair=True."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_ARG_REPAIR)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/argrepair",
        "--forms", "1",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    trial = artifact["trials"][0]
    assert trial["accepted"] is True
    # The repair is detected at the attempt level: raw args (from message_end)
    # differ from effective args (from host_tool_call).
    assert len(trial["attempts"]) >= 1
    attempt = trial["attempts"][0]
    assert attempt["omp_argument_repair"] is True
    # Raw args from the model had wrong trial_id and extra "i" key.
    assert attempt["raw_arguments"]["trial_id"] == "wrong"
    # Effective args (host-emitted) have the correct trial_id.
    assert attempt["effective_arguments"]["trial_id"] == trial["trial_id"]


# ─── fake RPC: model submission failure counts wrong ──────────────────

_FAKE_OMP_NO_TOOL_CALL = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-notool 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-notool", "contextUsage": {"tokens": 50}}})
        elif ctype == "prompt":
            # Agent ends without calling the tool at all.
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "text", "text": "I refuse to use tools."}]}})
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "no tool"}]}]})
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_no_tool_call_counts_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """When the model never calls the tool, the submission failure counts
    as wrong (not a probe error) and is reported in the artifact."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_NO_TOOL_CALL)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/notool",
        "--forms", "1",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    # Model failure is a measured result, not a probe error.
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    trial = artifact["trials"][0]
    assert trial["accepted"] is False
    assert trial["probe_error"] is False
    assert trial["submission"] is None
    assert trial["score"] is None
    assert trial["failure_category"] == "missing_tool_call"

    # Model result: 0 accuracy, 1 tool_format_failure.
    mr = artifact["model_results"][0]
    assert mr["primary_accuracy"] == 0.0
    assert mr["tool_format_failures"] == 1


# ─── fake RPC: checkpoint after every session ─────────────────────────


def test_end_to_end_checkpoint_after_every_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """The artifact is written atomically after every session, so a partial
    artifact exists even if the process is interrupted mid-schedule."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_SUCCESS)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    # Run with 3 forms — artifact should be written after each.
    rc = probe.main([
        "fake/model",
        "--forms", "3",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    # Final artifact has all 3 sessions.
    artifact = json.loads(out.read_text())
    assert artifact["progress"] == {"completed": 3, "total": 3}
    assert len(artifact["trials"]) == 3

    # No temp residue (atomic writes).
    assert [p for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


# ─── two-model end-to-end with ranking ────────────────────────────────

_FAKE_OMP_TWO_MODELS = r'''#!/usr/bin/env python3
import sys, json

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        sys.stdout.write("fake-omp-two 0.0.0-test\n")
        sys.stdout.flush()
        return 0
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctype = cmd.get("type")
        cid = cmd.get("id")
        if ctype in ("set_host_tools", "set_auto_retry", "set_auto_compaction"):
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "get_state":
            emit({"type": "response", "id": cid, "success": True,
                  "data": {"model": "fake-two", "contextUsage": {"tokens": 100}}})
        elif ctype == "prompt":
            msg = cmd.get("message", "")
            trial_id = None
            for ml in msg.split("\n"):
                if ml.startswith("Trial ID:"):
                    trial_id = ml.split(":", 1)[1].strip()
                    break
            # Determine model from argv --model flag.
            model_name = "unknown"
            for i, a in enumerate(argv):
                if a == "--model" and i + 1 < len(argv):
                    model_name = argv[i + 1]
                    break
            # "good" model picks correct letters from the prompt;
            # "bad" model always picks "A".
            # We can't easily parse correct letters, so just pick "A" for all.
            # The ranking test checks that both models complete with equal counts.
            answers = [{"number": n, "choice": "A", "explanation": "x"}
                       for n in range(1, 21)]
            args = {"trial_id": trial_id, "answers": answers}
            emit({"type": "host_tool_call", "id": "htc-1", "toolCallId": "tc-1",
                  "toolName": "submit_riddle_answers", "arguments": args})
            emit({"type": "message_end", "message": {"role": "assistant",
                  "content": [{"type": "toolCall", "id": "tc-1",
                   "name": "submit_riddle_answers", "arguments": args}]}})
            emit({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]})
            emit({"type": "response", "id": cid, "success": True})
        elif ctype == "abort":
            pass
    return 0

sys.exit(main())
'''


def test_end_to_end_two_models_ranking_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bank_file: Path,
) -> None:
    """Two models with equal trial counts and no probe errors → ranking present."""
    fake = tmp_path / "omp"
    fake.write_text(_FAKE_OMP_TWO_MODELS)
    os.chmod(fake, 0o755)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")

    out = tmp_path / "artifact.json"
    rc = probe.main([
        "fake/good", "fake/bad",
        "--forms", "2",
        "--seed", "42",
        "--trial-timeout-seconds", "30",
        "--omp-bin", str(fake),
        "--bank", str(bank_file),
        "--output", str(out),
    ])
    assert rc == probe.EXIT_OK

    artifact = json.loads(out.read_text())
    assert artifact["progress"] == {"completed": 4, "total": 4}
    assert len(artifact["trials"]) == 4
    # Both models have 2 trials each → ranking not omitted.
    assert artifact["ranking_omitted"] is False
    assert len(artifact["ranking"]) == 2
    # Paired winloss present.
    assert len(artifact["paired_winloss"]) >= 1
    assert len(artifact["bootstrap_intervals"]) >= 1
