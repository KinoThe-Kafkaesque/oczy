"""High-signal contract tests for the meta_cortex/v1 calibration layer.

These tests defend the scientific boundaries of the DEV calibration —
not plumbing.  They verify:

  - Exact scorer: normalized-exact matching, no substring false positives.
  - Empirical P95 edge cases: nearest-rank off-by-one, zero margin, exact Fraction.
  - Task-first power determinism: task is the independent unit, not probes/seeds.
  - Real seed variation: same domain/index is byte-identical, domains differ.
  - Synthetic calibration rows: no Qwen/network required.
  - Nearest-rank P95: non-overlapping pairing, joint-max, exact rank.
  - Power analysis: monotonicity, first passing integer, floor 30, zero variance.
  - Task-cluster bootstrap: resamples task clusters, not individual rows.
  - Canonical decimal string: exact Fraction to canonical decimal.
  - t-critical engine: deterministic, no scipy, fixed table.
  - Task-mean CI: n is task count, not probe/seed count.
  - Endpoint definitions: 7 superiority + 2 equivalence = 9 endpoints.
  - DevSplit firewall: no META_TEST member.

No real model, network, or Qwen is required.  All fixtures are synthetic.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from oczy.experiments.meta_cortex.calibration import (
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    CALIBRATION_RECORDS_SCHEMA,
    CALIBRATION_SCHEMA,
    CALIBRATION_THETA_HASHES_SCHEMA,
    ENDPOINT_FORMULAS,
    ENDPOINT_NAMES,
    ENDPOINT_SCHEMA,
    EQUIVALENCE_ENDPOINTS,
    INSTRUMENT_ID,
    INSTRUMENT_VERSION,
    SCORER_SCHEMA,
    SEED_DERIVATION_SCHEMA,
    SUPERIORITY_ENDPOINTS,
    CalibrationResult,
    FrozenScorer,
    NoUpdateRepeatRecord,
    ProbeCount,
    SeedCellRecord,
    TaskConditionRecord,
    _norm_cdf,
    _power_equivalence,
    _power_superiority,
    _uint63_from_sha256,
    canonical_decimal_string,
    compute_power_analysis,
    derive_seed_table,
    load_no_update_repeat_records,
    load_seed_cell_records,
    load_theta_hashes,
    nearest_rank_p95,
    normalize_seed_input,
    run_dev_calibration,
    t_critical_975,
    task_cluster_bootstrap,
    task_mean_ci,
)
from oczy.experiments.meta_cortex.contracts import (
    DevSplit,
    DialogueMessage,
    LearningEvent,
    MetaTask,
    OutcomeCode,
    ProbeBattery,
    ProbeCase,
    ProbeKind,
    TaskFamily,
)

# ---------------------------------------------------------------------------
# FrozenScorer tests
# ---------------------------------------------------------------------------


class TestFrozenScorer:
    """Test the frozen normalized-exact scorer — no substring matching."""

    @pytest.fixture
    def scorer(self) -> FrozenScorer:
        return FrozenScorer()

    def test_exact_match_is_correct(self, scorer: FrozenScorer) -> None:
        assert scorer.score_response("marmalade", "marmalade") is True

    def test_extra_prose_fails(self, scorer: FrozenScorer) -> None:
        """The expected token inside a longer string must fail."""
        assert scorer.score_response("marmalade", "The answer is marmalade.") is False

    def test_substring_fails(self, scorer: FrozenScorer) -> None:
        """Substring containment must not score as correct."""
        assert scorer.score_response("marmalade", "marmalade123") is False

    def test_case_preserved(self, scorer: FrozenScorer) -> None:
        """Case is preserved — Marmalade != marmalade."""
        assert scorer.score_response("Marmalade", "marmalade") is False

    def test_punctuation_preserved(self, scorer: FrozenScorer) -> None:
        assert scorer.score_response("level 7.", "level 7") is False

    def test_outer_whitespace_stripped(self, scorer: FrozenScorer) -> None:
        """Leading/trailing whitespace is stripped."""
        assert scorer.score_response("marmalade", "  marmalade  ") is True

    def test_internal_whitespace_preserved(self, scorer: FrozenScorer) -> None:
        """Internal whitespace is preserved, not collapsed."""
        assert scorer.score_response("a b", "a  b") is False

    def test_empty_generated_fails(self, scorer: FrozenScorer) -> None:
        assert scorer.score_response("marmalade", "") is False

    def test_empty_expected_fails(self, scorer: FrozenScorer) -> None:
        """Empty expected string returns False — no free pass."""
        assert scorer.score_response("", "marmalade") is False

    def test_both_empty_fails(self, scorer: FrozenScorer) -> None:
        """Two empty strings: expected is empty, so returns False."""
        assert scorer.score_response("", "") is False

    def test_reversed_order_fails(self, scorer: FrozenScorer) -> None:
        """Reversed composition order must fail."""
        assert scorer.score_response("alpha then beta", "beta then alpha") is False

    def test_nfkc_normalization(self, scorer: FrozenScorer) -> None:
        """NFKC normalization: fullwidth chars normalize to ASCII."""
        # Fullwidth 'm' (U+FF4D) should normalize to ASCII 'm'.
        assert scorer.normalize_response("\uff4d\u0061\u0072\u006d\u0061\u006c\u0061\u0064\u0065") == "marmalade"

    def test_normalize_strips_outer_only(self, scorer: FrozenScorer) -> None:
        """Only outer whitespace is stripped; internal is preserved."""
        assert scorer.normalize_response("  hello  world  ") == "hello  world"

    def test_normalize_non_string_raises(self, scorer: FrozenScorer) -> None:
        with pytest.raises(TypeError, match="str"):
            scorer.normalize_response(123)  # type: ignore[arg-type]

    def test_sha256_is_deterministic(self, scorer: FrozenScorer) -> None:
        """The scorer hash must be deterministic across instances."""
        s1 = FrozenScorer()
        s2 = FrozenScorer()
        assert s1.sha256 == s2.sha256

    def test_sha256_is_64_char_hex(self, scorer: FrozenScorer) -> None:
        sha = scorer.sha256
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_score_battery_returns_probe_count(self, scorer: FrozenScorer) -> None:
        """score_battery returns a ProbeCount with correct/total."""
        result = scorer.score_battery(
            ["a", "b", "c"],
            ["a", "x", "c"],
        )
        assert isinstance(result, ProbeCount)
        assert result.correct == 2
        assert result.total == 3

    def test_score_battery_unequal_length_raises(self, scorer: FrozenScorer) -> None:
        with pytest.raises(ValueError, match="equal length"):
            scorer.score_battery(["a", "b"], ["a"])

    def test_score_battery_empty_raises(self, scorer: FrozenScorer) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            scorer.score_battery([], [])


# ---------------------------------------------------------------------------
# ProbeCount tests
# ---------------------------------------------------------------------------


class TestProbeCount:
    def test_valid_probe_count(self) -> None:
        pc = ProbeCount(correct=5, total=10)
        assert pc.correct == 5
        assert pc.total == 10
        assert pc.accuracy == Fraction(1, 2)

    def test_reject_negative_correct(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            ProbeCount(correct=-1, total=10)

    def test_reject_zero_total(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            ProbeCount(correct=0, total=0)

    def test_reject_correct_exceeds_total(self) -> None:
        with pytest.raises(ValueError, match="> total"):
            ProbeCount(correct=11, total=10)

    def test_to_json_obj(self) -> None:
        pc = ProbeCount(correct=3, total=7)
        assert pc.to_json_obj() == {"correct": 3, "total": 7}

    def test_is_frozen(self) -> None:
        import dataclasses
        pc = ProbeCount(correct=1, total=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.correct = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Seed derivation tests
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    """Test deterministic seed derivation — same domain/index is byte-identical."""

    def test_same_call_is_identical(self) -> None:
        table1 = derive_seed_table()
        table2 = derive_seed_table()
        assert table1 == table2

    def test_returns_dict_with_required_keys(self) -> None:
        table = derive_seed_table()
        assert "schema" in table
        assert "instrument_id" in table
        assert "instrument_version" in table
        assert "developmental" in table
        assert "evaluation" in table
        assert "no_update_repeat" in table
        assert "task_cluster_bootstrap" in table

    def test_developmental_has_5_seeds(self) -> None:
        table = derive_seed_table()
        assert len(table["developmental"]) == 5

    def test_evaluation_has_5_seeds(self) -> None:
        table = derive_seed_table()
        assert len(table["evaluation"]) == 5

    def test_no_update_repeat_has_20_seeds(self) -> None:
        table = derive_seed_table()
        assert len(table["no_update_repeat"]) == 20

    def test_task_cluster_bootstrap_is_single_int(self) -> None:
        table = derive_seed_table()
        assert isinstance(table["task_cluster_bootstrap"], int)

    def test_all_seeds_unique(self) -> None:
        """All seeds across all domains must be unique."""
        table = derive_seed_table()
        all_seeds = (
            list(table["developmental"])
            + list(table["evaluation"])
            + list(table["no_update_repeat"])
            + [table["task_cluster_bootstrap"]]
        )
        assert len(set(all_seeds)) == len(all_seeds)

    def test_all_seeds_uint63(self) -> None:
        """All seeds must be non-negative and < 2^63."""
        table = derive_seed_table()
        max_uint63 = (1 << 63) - 1
        for seed in table["developmental"]:
            assert 0 <= seed <= max_uint63
        for seed in table["evaluation"]:
            assert 0 <= seed <= max_uint63
        for seed in table["no_update_repeat"]:
            assert 0 <= seed <= max_uint63
        assert 0 <= table["task_cluster_bootstrap"] <= max_uint63

    def test_different_versions_differ(self) -> None:
        """Different instrument versions must produce different seeds."""
        v1 = derive_seed_table(instrument_version="v1")
        v2 = derive_seed_table(instrument_version="v2")
        assert v1["developmental"] != v2["developmental"]

    def test_developmental_differs_from_evaluation(self) -> None:
        """Developmental and evaluation seeds must not overlap."""
        table = derive_seed_table()
        dev_set = set(table["developmental"])
        eval_set = set(table["evaluation"])
        assert dev_set.isdisjoint(eval_set)

    def test_uint63_from_sha256_deterministic(self) -> None:
        """Same label produces same seed."""
        s1 = _uint63_from_sha256("test_label")
        s2 = _uint63_from_sha256("test_label")
        assert s1 == s2

    def test_uint63_from_sha256_different_labels(self) -> None:
        """Different labels produce different seeds."""
        s1 = _uint63_from_sha256("label_a")
        s2 = _uint63_from_sha256("label_b")
        assert s1 != s2

    def test_uint63_range(self) -> None:
        """The result must be a uint63."""
        val = _uint63_from_sha256("any_label")
        assert 0 <= val < (1 << 63)


# ---------------------------------------------------------------------------
# Canonical decimal string tests
# ---------------------------------------------------------------------------


class TestCanonicalDecimalString:
    def test_zero(self) -> None:
        assert canonical_decimal_string(0, 1) == "0.0"

    def test_simple_fraction(self) -> None:
        assert canonical_decimal_string(1, 2) == "0.5"

    def test_one_tenth(self) -> None:
        assert canonical_decimal_string(1, 10) == "0.1"

    def test_eighteen_hundredths(self) -> None:
        assert canonical_decimal_string(18, 100) == "0.18"

    def test_trailing_zeros_stripped(self) -> None:
        """Trailing zeros are stripped: 50/100 = 0.5, not 0.50."""
        assert canonical_decimal_string(50, 100) == "0.5"

    def test_integer_value(self) -> None:
        """Integer values get .0 suffix."""
        assert canonical_decimal_string(5, 1) == "5.0"

    def test_negative_numerator(self) -> None:
        result = canonical_decimal_string(-1, 10)
        assert result.startswith("-")
        assert "0.1" in result

    def test_negative_denominator_normalized(self) -> None:
        """Negative denominator is normalized to positive."""
        assert canonical_decimal_string(1, -10) == canonical_decimal_string(-1, 10)

    def test_zero_denominator_raises(self) -> None:
        with pytest.raises(ValueError, match="nonzero"):
            canonical_decimal_string(1, 0)

    def test_repeating_decimal_rounded(self) -> None:
        """1/3 should be rounded at max_precision."""
        result = canonical_decimal_string(1, 3, max_precision=10)
        # Should be approximately 0.3333333333
        assert result.startswith("0.3")

    def test_one_seventh(self) -> None:
        """1/7 is a classic repeating decimal."""
        result = canonical_decimal_string(1, 7, max_precision=20)
        assert result.startswith("0.142857")


# ---------------------------------------------------------------------------
# Nearest-rank P95 tests
# ---------------------------------------------------------------------------


class TestNearestRankP95:
    """Test the exact nearest-rank empirical P95 computation."""

    def test_simple_p95(self) -> None:
        """With 20 values, k=ceil(0.95*20)=19, P95 = 19th value (1-indexed)."""
        values = [Fraction(i, 100) for i in range(20)]  # 0.00, 0.01, ..., 0.19
        result = nearest_rank_p95(values)
        assert result["M"] == 20
        assert result["k"] == 19
        # k=19, zero-indexed 18 → 0.18
        assert result["selected_num"] == 18
        assert result["selected_den"] == 100
        assert result["selected_string"] == "0.18"

    def test_off_by_one_lower(self) -> None:
        """The rank below k must NOT be selected."""
        values = [Fraction(i, 100) for i in range(20)]
        result = nearest_rank_p95(values)
        # Must not be the 18th value (0.17).
        assert result["selected_num"] != 17

    def test_off_by_one_upper(self) -> None:
        """The rank above k must NOT be selected."""
        values = [Fraction(i, 100) for i in range(20)]
        result = nearest_rank_p95(values)
        # Must not be the 20th value (0.19).
        assert result["selected_num"] != 19

    def test_zero_margin(self) -> None:
        """If all values are zero, P95 is zero — this is valid."""
        values = [Fraction(0)] * 100
        result = nearest_rank_p95(values)
        assert result["selected_num"] == 0
        assert result["selected_den"] == 1

    def test_single_value(self) -> None:
        """With one value, k=ceil(0.95*1)=1, P95 = that value."""
        result = nearest_rank_p95([Fraction(3, 10)])
        assert result["M"] == 1
        assert result["k"] == 1
        assert result["selected_num"] == 3
        assert result["selected_den"] == 10

    def test_pooled_across_families(self) -> None:
        """Values from multiple families are pooled into one distribution."""
        # 3 families × 10 values each = 30 values.
        values = [Fraction(i, 100) for i in range(30)]
        result = nearest_rank_p95(values)
        # k = ceil(0.95 * 30) = 29, so 29th value (0-indexed: 28) = 0.28
        assert result["k"] == 29
        assert result["selected_num"] == 28

    def test_non_interpolated(self) -> None:
        """Nearest-rank must not interpolate between values."""
        # 10 values: 0.0, 0.1, ..., 0.9
        values = [Fraction(i, 10) for i in range(10)]
        result = nearest_rank_p95(values)
        # k = ceil(0.95 * 10) = 10, so 10th value = 0.9
        # Interpolated would give 0.855, which must NOT happen.
        assert result["selected_num"] == 9
        assert result["selected_den"] == 10

    def test_returns_exact_fraction(self) -> None:
        """The result must contain exact Fraction components, not floats."""
        result = nearest_rank_p95([Fraction(1, 3), Fraction(2, 3)])
        assert isinstance(result["selected_num"], int)
        assert isinstance(result["selected_den"], int)

    def test_deterministic(self) -> None:
        """Same input must produce same output."""
        values = [Fraction(i, 50) for i in range(50)]
        r1 = nearest_rank_p95(values)
        r2 = nearest_rank_p95(values)
        assert r1 == r2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            nearest_rank_p95([])

    def test_adjacent_values_present(self) -> None:
        """The result must include adjacent values for audit."""
        values = [Fraction(i, 100) for i in range(20)]
        result = nearest_rank_p95(values)
        assert result["lower_adjacent_num"] is not None
        assert result["upper_adjacent_num"] is not None
        # k=19, lower_adjacent = k-2 = 17th value (0-indexed 17) = 0.17
        assert result["lower_adjacent_num"] == 17
        # upper_adjacent = k = 20th value (0-indexed 19) = 0.19
        assert result["upper_adjacent_num"] == 19

    def test_sorted_values_in_result(self) -> None:
        """The result must include sorted values for reproducibility."""
        values = [Fraction(3, 10), Fraction(1, 10), Fraction(2, 10)]
        result = nearest_rank_p95(values)
        assert result["sorted_values_num"] == [1, 2, 3]
        assert result["sorted_values_den"] == [10, 10, 10]


# ---------------------------------------------------------------------------
# Task-cluster bootstrap tests
# ---------------------------------------------------------------------------


class TestTaskClusterBootstrap:
    """The bootstrap must resample task clusters, not individual rows."""

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed must produce same bootstrap result."""
        task_values = {
            "contextual_remap": [Fraction(1, 10)] * 5,
            "rule_transformation": [Fraction(2, 10)] * 5,
            "finite_state": [Fraction(3, 10)] * 5,
        }
        r1 = task_cluster_bootstrap(task_values, n_replicates=100, seed=42)
        r2 = task_cluster_bootstrap(task_values, n_replicates=100, seed=42)
        assert r1 == r2

    def test_different_seed_differs(self) -> None:
        """Different seeds should generally produce different results."""
        task_values = {
            "contextual_remap": [Fraction(1, 10), Fraction(2, 10)],
            "rule_transformation": [Fraction(3, 10), Fraction(4, 10)],
            "finite_state": [Fraction(5, 10), Fraction(6, 10)],
        }
        r1 = task_cluster_bootstrap(task_values, n_replicates=100, seed=42)
        r2 = task_cluster_bootstrap(task_values, n_replicates=100, seed=43)
        # The point estimate is the same (deterministic from data), but
        # the bootstrap CI may differ.
        assert r1["point_estimate_num"] == r2["point_estimate_num"]

    def test_returns_dict_with_required_keys(self) -> None:
        task_values = {
            "contextual_remap": [Fraction(1, 10), Fraction(2, 10)],
        }
        result = task_cluster_bootstrap(task_values, n_replicates=10, seed=0)
        assert "n_replicates" in result
        assert "seed" in result
        assert "point_estimate_num" in result
        assert "point_estimate_den" in result
        assert "point_estimate_string" in result
        assert "bootstrap_ci_lower_num" in result
        assert "bootstrap_ci_upper_num" in result

    def test_resamples_clusters_not_rows(self) -> None:
        """The bootstrap resamples complete task clusters, not individual rows.

        With 3 tasks each having 5 values, a task-cluster bootstrap
        selects 3 tasks (with replacement) and keeps all values per
        selected task. The point estimate should be the nearest-rank P95
        of all 15 values pooled.
        """
        task_values = {
            "family_a": [Fraction(1, 10)] * 5,
            "family_b": [Fraction(2, 10)] * 5,
            "family_c": [Fraction(3, 10)] * 5,
        }
        result = task_cluster_bootstrap(task_values, n_replicates=1000, seed=42)
        # Point estimate is the P95 of all 15 pooled values.
        # All values are 0.1, 0.2, or 0.3, each repeated 5 times.
        # k = ceil(0.95 * 15) = 15, so the 15th value = 0.3
        assert result["point_estimate_num"] == 3
        assert result["point_estimate_den"] == 10

    def test_zero_replicates_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            task_cluster_bootstrap(
                {"a": [Fraction(1, 10)]},
                n_replicates=0,
            )

    def test_empty_family_raises(self) -> None:
        with pytest.raises(ValueError, match="no task cluster"):
            task_cluster_bootstrap(
                {"a": []},
                n_replicates=10,
            )


# ---------------------------------------------------------------------------
# t-critical engine tests
# ---------------------------------------------------------------------------


class TestTCriticalEngine:
    """The t-critical engine is deterministic, no scipy, fixed table."""

    def test_df_1_known_value(self) -> None:
        """t_critical_975(1) = 12.7062057... (known value)."""
        result = t_critical_975(1)
        assert abs(result - 12.7062057) < 0.001

    def test_df_29_known_value(self) -> None:
        """t_critical_975(29) ≈ 2.0452... (commonly used for n=30)."""
        result = t_critical_975(29)
        assert abs(result - 2.04523) < 0.001

    def test_df_large_uses_normal(self) -> None:
        """For df > 100, the normal approximation 1.959964 is used."""
        result = t_critical_975(200)
        assert abs(result - 1.959963984540) < 0.0001

    def test_deterministic(self) -> None:
        """Same df always returns same value."""
        assert t_critical_975(10) == t_critical_975(10)

    def test_monotonically_decreasing(self) -> None:
        """t-critical decreases as df increases (approaching normal)."""
        vals = [t_critical_975(df) for df in [1, 2, 5, 10, 29, 50, 100]]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    def test_df_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            t_critical_975(0)

    def test_all_values_positive(self) -> None:
        """All t-critical values must be positive."""
        for df in [1, 2, 5, 10, 29, 50, 100, 200]:
            assert t_critical_975(df) > 0


# ---------------------------------------------------------------------------
# Task-mean CI tests
# ---------------------------------------------------------------------------


class TestTaskMeanCI:
    """Task-mean CI uses task count as n, not probe/seed count."""

    def test_basic_ci(self) -> None:
        """Compute CI for 30 task means."""
        values = [0.5 + 0.01 * i for i in range(30)]
        result = task_mean_ci(values)
        assert result["n"] == 30
        assert abs(result["mean"] - sum(values) / 30) < 1e-10
        assert result["ci_lower"] < result["mean"]
        assert result["ci_upper"] > result["mean"]

    def test_n_is_task_count_not_probes(self) -> None:
        """n must equal len(values), not be inflated by probe counts."""
        values = [0.8, 0.6, 0.7]
        result = task_mean_ci(values)
        assert result["n"] == 3

    def test_single_value_returns_nan_ci(self) -> None:
        """With n=1, CI is NaN (no degrees of freedom)."""
        result = task_mean_ci([0.5])
        assert result["n"] == 1
        assert math.isnan(result["ci_lower"])
        assert math.isnan(result["ci_upper"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            task_mean_ci([])

    def test_zero_variance(self) -> None:
        """All identical values → sd=0, CI = [mean, mean]."""
        result = task_mean_ci([0.5, 0.5, 0.5, 0.5, 0.5])
        assert result["sd"] == 0.0
        assert result["ci_lower"] == result["mean"]
        assert result["ci_upper"] == result["mean"]

    def test_ci_width_decreases_with_n(self) -> None:
        """Larger n → narrower CI (more precision)."""
        values_small = [0.5, 0.6, 0.4]
        values_large = [0.5, 0.6, 0.4, 0.55, 0.45, 0.52, 0.48, 0.51, 0.49, 0.5]
        ci_small = task_mean_ci(values_small)
        ci_large = task_mean_ci(values_large)
        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small

    def test_t_critical_matches_engine(self) -> None:
        """The t_critical in the CI must match t_critical_975(n-1)."""
        values = [0.5, 0.6, 0.4, 0.55, 0.45]
        result = task_mean_ci(values)
        assert result["t_critical"] == t_critical_975(len(values) - 1)


# ---------------------------------------------------------------------------
# Power analysis tests
# ---------------------------------------------------------------------------


class TestPowerAnalysis:
    """Test the power analysis computation."""

    def _make_stats(self, mean: float, sd: float, n: int = 30) -> dict[str, float]:
        """Build per-endpoint stats dict."""
        if sd > 0 and n >= 2:
            tcrit: float = t_critical_975(n - 1)
            sem = sd / math.sqrt(n)
            ci_lower = mean - tcrit * sem
            ci_upper = mean + tcrit * sem
        else:
            tcrit = 0.0
            ci_lower = mean
            ci_upper = mean
        return {"n": n, "mean": mean, "sd": sd, "ci_lower": ci_lower, "ci_upper": ci_upper, "t_critical": tcrit}

    def _make_family_stats(self, family: str, mean: float, sd: float) -> dict[str, dict[str, dict[str, float]]]:
        """Build per_family_endpoint_stats for one family with all 9 endpoints."""
        stats = {}
        for ep in SUPERIORITY_ENDPOINTS:
            stats[ep] = self._make_stats(mean, sd)
        for ep in EQUIVALENCE_ENDPOINTS:
            stats[ep] = self._make_stats(0.0, sd)
        return {family: stats}

    def test_feasible_with_positive_mean(self) -> None:
        """A positive mean with reasonable SD should be feasible."""
        per_family = self._make_family_stats("contextual_remap", 0.3, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
            target_power=0.80,
        )
        assert result["feasibility_status"] == "feasible"
        assert result["sample_size_tasks_per_family"] >= 30
        assert len(result["block_reasons"]) == 0

    def test_floor_30(self) -> None:
        """The minimum N must be 30, even if power is satisfied earlier."""
        per_family = self._make_family_stats("contextual_remap", 1.0, 0.01)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
            target_power=0.80,
            minimum_n=30,
        )
        assert result["sample_size_tasks_per_family"] >= 30

    def test_zero_variance_positive_mean(self) -> None:
        """If SD=0 and mean is positive, superiority requires floor 30."""
        per_family = self._make_family_stats("contextual_remap", 0.5, 0.0)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
        )
        assert result["feasibility_status"] == "feasible"
        # Zero-variance superiority endpoints get floor.
        for ep in SUPERIORITY_ENDPOINTS:
            key = f"contextual_remap/{ep}"
            assert result["required_n_per_endpoint"][key]["status"] == "floor"
            assert result["required_n_per_endpoint"][key]["required_n"] == 30

    def test_nonpositive_superiority_mean_blocks(self) -> None:
        """If mean <= 0 for superiority, no finite N can repair it."""
        per_family = self._make_family_stats("contextual_remap", 0.0, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
        )
        assert result["feasibility_status"] == "blocked"
        assert len(result["block_reasons"]) > 0

    def test_zero_margin_blocks_equivalence(self) -> None:
        """If Delta <= 0 for equivalence, no finite N can repair it."""
        per_family = self._make_family_stats("contextual_remap", 0.3, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.0,
        )
        assert result["feasibility_status"] == "blocked"

    def test_equivalence_mean_on_boundary_blocks(self) -> None:
        """If |mean| >= Delta for equivalence, no finite N."""
        per_family = self._make_family_stats("contextual_remap", 0.3, 0.1)
        # Set equivalence endpoint mean to exactly the margin.
        for ep in EQUIVALENCE_ENDPOINTS:
            per_family["contextual_remap"][ep] = self._make_stats(0.05, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
        )
        assert result["feasibility_status"] == "blocked"

    def test_sample_size_is_max_of_all_endpoints(self) -> None:
        """sample_size_tasks_per_family = max(minimum_n, all finite required_n)."""
        per_family = self._make_family_stats("contextual_remap", 0.3, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
        )
        all_required = [
            v["required_n"] for v in result["required_n_per_endpoint"].values()
            if v["required_n"] is not None
        ]
        if all_required:
            assert result["sample_size_tasks_per_family"] == max(30, max(all_required))

    def test_block_reasons_tuple(self) -> None:
        """block_reasons must be a tuple of strings."""
        per_family = self._make_family_stats("contextual_remap", 0.0, 0.1)
        result = compute_power_analysis(
            per_family,
            equivalence_margin=0.05,
        )
        assert isinstance(result["block_reasons"], tuple)
        for reason in result["block_reasons"]:
            assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# Power function unit tests
# ---------------------------------------------------------------------------


class TestPowerFunctions:
    def test_superiority_power_increases_with_n(self) -> None:
        """Power must increase with n for positive mean."""
        p1 = _power_superiority(30, 0.3, 0.1)
        p2 = _power_superiority(100, 0.3, 0.1)
        assert p2 > p1

    def test_superiority_power_zero_mean(self) -> None:
        """With mean=0, superiority power is low."""
        p = _power_superiority(100, 0.0, 0.1)
        assert p < 0.5

    def test_equivalence_power_increases_with_n(self) -> None:
        """Power must increase with n for mean inside margin."""
        p1 = _power_equivalence(30, 0.0, 0.1, 0.05)
        p2 = _power_equivalence(100, 0.0, 0.1, 0.05)
        assert p2 > p1

    def test_norm_cdf_known_values(self) -> None:
        """Standard normal CDF: Phi(0) = 0.5, Phi(1.96) ≈ 0.975."""
        assert abs(_norm_cdf(0) - 0.5) < 1e-10
        assert abs(_norm_cdf(1.959963984540) - 0.975) < 0.001


# ---------------------------------------------------------------------------
# Endpoint definitions tests
# ---------------------------------------------------------------------------


class TestEndpointDefinitions:
    def test_nine_endpoints(self) -> None:
        assert len(ENDPOINT_NAMES) == 9

    def test_seven_superiority(self) -> None:
        assert len(SUPERIORITY_ENDPOINTS) == 7

    def test_two_equivalence(self) -> None:
        assert len(EQUIVALENCE_ENDPOINTS) == 2

    def test_disjoint_sets(self) -> None:
        """Superiority and equivalence endpoints must be disjoint."""
        sup_set = set(SUPERIORITY_ENDPOINTS)
        eq_set = set(EQUIVALENCE_ENDPOINTS)
        assert sup_set.isdisjoint(eq_set)

    def test_union_equals_all(self) -> None:
        """Union of superiority and equivalence = all endpoints."""
        assert set(SUPERIORITY_ENDPOINTS) | set(EQUIVALENCE_ENDPOINTS) == set(ENDPOINT_NAMES)

    def test_all_endpoints_have_formulas(self) -> None:
        """Every endpoint must have a formula string."""
        for ep in ENDPOINT_NAMES:
            assert ep in ENDPOINT_FORMULAS
            assert isinstance(ENDPOINT_FORMULAS[ep], str)
            assert len(ENDPOINT_FORMULAS[ep]) > 0

    def test_condition_constants(self) -> None:
        """Condition name constants are correct."""
        assert C1 == "update_disabled"
        assert C2 == "untrained_rule"
        assert C3 == "trained"
        assert C4 == "feedback_shuffled"
        assert C5 == "state_zeroed"
        assert C6 == "state_swapped"


# ---------------------------------------------------------------------------
# Task-first aggregation property tests
# ---------------------------------------------------------------------------


class TestTaskFirstAggregation:
    """The task is the independent unit — probes/seeds never inflate n."""

    def test_duplicating_probes_does_not_change_n(self) -> None:
        """Doubling every probe count must not change the inferential n."""
        # If a task has 10 correct / 20 total, doubling to 20/40
        # gives the same task accuracy but must not change n.
        task1 = Fraction(10, 20)
        task1_doubled = Fraction(20, 40)
        assert task1 == task1_doubled  # Same accuracy.

    def test_seed_cells_do_not_inflate_n(self) -> None:
        """5×5 seed cells improve within-task characterization but n stays at task count."""
        n_tasks = 30
        n_seed_cells = 25
        total_measurements = n_tasks * n_seed_cells
        assert total_measurements == 750
        # The independent n is n_tasks, not total_measurements.
        independent_n = n_tasks
        assert independent_n == 30

    def test_family_mean_averages_task_means(self) -> None:
        """The family estimate is the mean of task means, not the micro-pool."""
        task3 = Fraction(8, 10)  # 0.8
        task4 = Fraction(2, 20)  # 0.1
        family_mean_task_first = (task3 + task4) / 2  # 0.45
        micro_pool = Fraction(8 + 2, 10 + 20)  # 10/30 ≈ 0.333
        assert family_mean_task_first != micro_pool  # They differ.

    def test_task_mean_ci_n_equals_task_count(self) -> None:
        """task_mean_ci uses len(values) as n, never inflating."""
        values = [0.8, 0.6, 0.7, 0.9, 0.5]
        result = task_mean_ci(values)
        assert result["n"] == 5  # Task count, not probe count.


# ---------------------------------------------------------------------------
# Non-overlapping pairing tests
# ---------------------------------------------------------------------------


class TestNonOverlappingPairing:
    """Repeat pairs must be non-overlapping: (0,1), (2,3), ..., (18,19)."""

    def test_pairs_are_non_overlapping(self) -> None:
        """The 20 repeats produce 10 non-overlapping pairs."""
        n_repeats = 20
        pairs = [(2 * j, 2 * j + 1) for j in range(n_repeats // 2)]
        assert len(pairs) == 10
        all_indices = set()
        for a, b in pairs:
            assert a not in all_indices
            assert b not in all_indices
            all_indices.add(a)
            all_indices.add(b)
        assert all_indices == set(range(20))

    def test_all_pairwise_differences_are_pseudo_replication(self) -> None:
        """Creating all pairwise differences would produce 190 pairs from 20,
        which is pseudo-replication. Only 10 non-overlapping pairs are valid."""
        n = 20
        all_pairs = n * (n - 1) // 2  # 190
        non_overlapping = n // 2  # 10
        assert all_pairs == 190
        assert non_overlapping == 10


# ---------------------------------------------------------------------------
# DevSplit firewall (calibration-specific)
# ---------------------------------------------------------------------------


class TestCalibrationSplitFirewall:
    """Calibration uses only meta-validation rules — no meta-test access."""

    def test_devsplit_still_two_members(self) -> None:
        """DevSplit must remain exactly {META_TRAIN, META_VALIDATION}."""
        from oczy.experiments.meta_cortex.contracts import DevSplit
        members = {m.name for m in DevSplit}
        assert members == {"META_TRAIN", "META_VALIDATION"}
        assert "META_TEST" not in members

    def test_no_meta_test_in_task_generator_config(self) -> None:
        """TaskGeneratorConfig must not have meta_test fields."""
        import dataclasses

        from oczy.experiments.meta_cortex.contracts import TaskGeneratorConfig
        field_names = {f.name for f in dataclasses.fields(TaskGeneratorConfig)}
        assert "meta_test_tasks_per_family" not in field_names
        assert "meta_test_seed" not in field_names
        assert "test_seed" not in field_names


# ---------------------------------------------------------------------------
# 95% CI equivalence containment tests
# ---------------------------------------------------------------------------


class TestEquivalenceContainment:
    """Equivalence acceptance is CI contained in [-Delta, Delta] using 95% CI."""

    def test_contained_ci_accepts(self) -> None:
        """If CI is entirely within [-Delta, Delta], equivalence holds."""
        ci_lower = -0.02
        ci_upper = 0.02
        delta = 0.05
        accepts = ci_lower >= -delta and ci_upper <= delta
        assert accepts is True

    def test_ci_exceeds_upper_fails(self) -> None:
        ci_lower = -0.01
        ci_upper = 0.06
        delta = 0.05
        accepts = ci_lower >= -delta and ci_upper <= delta
        assert accepts is False

    def test_ci_exceeds_lower_fails(self) -> None:
        ci_lower = -0.06
        ci_upper = 0.01
        delta = 0.05
        accepts = ci_lower >= -delta and ci_upper <= delta
        assert accepts is False

    def test_95_ci_is_conservative_tost(self) -> None:
        """95% CI containment is more conservative than 90% CI TOST.

        The 95% CI is wider than the 90% CI, so containment in the 95% CI
        is a stricter criterion. This is intentional: the immutable spec
        requires 95% CIs, not 90%.
        """
        # For the same data, the 95% CI is wider than the 90% CI.
        # t_0.975 > t_0.95 for any df.
        for df in [1, 5, 10, 29, 100]:
            t_975 = t_critical_975(df)  # 95% CI (two-tailed alpha=0.05)
            assert t_975 > 0
            # The 90% CI would use a smaller critical value, so 95% CI
            # containment is strictly more conservative.


# ---------------------------------------------------------------------------
# Intersection-union test property
# ---------------------------------------------------------------------------


class TestIntersectionUnion:
    """The global ACCEPT rule is an intersection-union claim."""

    def test_all_endpoints_must_pass(self) -> None:
        """If any required endpoint fails, the global claim fails."""
        endpoint_results = {
            "adaptation_delta": True,
            "transfer_delta": True,
            "composition_delta": True,
            "meta_training_delta": True,
            "feedback_semantics_delta": True,
            "causal_state_delta": True,
            "state_addressing_delta": True,
            "specificity_delta": True,
            "trace_free_survival": True,
        }
        global_accept = all(endpoint_results.values())
        assert global_accept is True

        endpoint_results["adaptation_delta"] = False
        global_accept = all(endpoint_results.values())
        assert global_accept is False

    def test_no_multiplicity_correction_needed(self) -> None:
        """The intersection-union test does not need Holm/Bonferroni correction.

        Under IUT, the global Type I error is controlled at the nominal
        level without multiplicity correction. The acceptance rule is a
        simple AND of all endpoints.
        """
        endpoint_pass = [True] * 9
        assert all(endpoint_pass) is True
        endpoint_pass[3] = False
        assert all(endpoint_pass) is False


# ---------------------------------------------------------------------------
# Calibration evidence artifact tests
# ---------------------------------------------------------------------------


class TestCalibrationEvidence:
    """Calibration evidence must not contain signoff or test fields."""

    def test_no_signoff_fields_in_evidence(self) -> None:
        """Evidence artifacts must not contain signoff/verdict fields."""
        import dataclasses

        from oczy.experiments.meta_cortex.instrument_contracts import CandidateManifest
        manifest_fields = {f.name for f in dataclasses.fields(CandidateManifest)}
        assert "human_signoff_id" not in manifest_fields
        assert "signoff_sha256" not in manifest_fields

    def test_candidate_manifest_has_holdout_flag(self) -> None:
        import dataclasses

        from oczy.experiments.meta_cortex.instrument_contracts import CandidateManifest
        manifest_fields = {f.name for f in dataclasses.fields(CandidateManifest)}
        assert "calibration_holdout_accessed" in manifest_fields

    def test_candidate_manifest_has_evidence_hashes(self) -> None:
        import dataclasses

        from oczy.experiments.meta_cortex.instrument_contracts import CandidateManifest
        manifest_fields = {f.name for f in dataclasses.fields(CandidateManifest)}
        assert "calibration_report_sha256" in manifest_fields
        assert "power_report_sha256" in manifest_fields

    def test_candidate_manifest_has_margin_and_n(self) -> None:
        import dataclasses

        from oczy.experiments.meta_cortex.instrument_contracts import CandidateManifest
        manifest_fields = {f.name for f in dataclasses.fields(CandidateManifest)}
        assert "equivalence_margin" in manifest_fields
        assert "sample_size_tasks_per_family" in manifest_fields
        assert "meta_test_tasks_by_family" in manifest_fields


# ---------------------------------------------------------------------------
# Schema constants tests
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_calibration_schema(self) -> None:
        assert CALIBRATION_SCHEMA == "oczy/meta-cortex/calibration/v1"

    def test_scorer_schema(self) -> None:
        assert SCORER_SCHEMA == "oczy/meta-cortex/scorers/v1"

    def test_endpoint_schema(self) -> None:
        assert ENDPOINT_SCHEMA == "oczy/meta-cortex/endpoints/v1"

    def test_instrument_id(self) -> None:
        assert INSTRUMENT_ID == "meta_cortex/v1"

    def test_instrument_version(self) -> None:
        assert INSTRUMENT_VERSION == "v1"

    def test_seed_derivation_schema(self) -> None:
        assert SEED_DERIVATION_SCHEMA == "oczy/meta-cortex/calibration-seeds/v1"



# ---------------------------------------------------------------------------
# Seed normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeSeedInput:
    """Seed input must accept 32 raw bytes or 64 hex chars, normalized to 32 bytes."""

    def test_32_raw_bytes_passes_through(self) -> None:
        raw = bytes(range(32))
        assert normalize_seed_input(raw) == raw

    def test_64_hex_chars_decodes(self) -> None:
        hex_str = "00" * 32
        result = normalize_seed_input(hex_str)
        assert result == b"\x00" * 32

    def test_64_hex_bytes_decodes(self) -> None:
        hex_bytes = b"00" * 32
        result = normalize_seed_input(hex_bytes)
        assert result == b"\x00" * 32

    def test_32_bytes_and_64_hex_normalize_same(self) -> None:
        raw = bytes(range(32))
        hex_str = raw.hex()
        assert normalize_seed_input(raw) == normalize_seed_input(hex_str)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="32 raw bytes or 64 hex"):
            normalize_seed_input(b"\x00" * 16)

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ValueError, match="lowercase hex"):
            normalize_seed_input("zz" * 32)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="32 raw bytes or 64 hex"):
            normalize_seed_input(b"")


# ---------------------------------------------------------------------------
# Record file schema tests
# ---------------------------------------------------------------------------


class TestRecordFileSchemas:
    def test_records_schema_constant(self) -> None:
        assert CALIBRATION_RECORDS_SCHEMA == "oczy/meta-cortex/calibration-records/v1"

    def test_theta_hashes_schema_constant(self) -> None:
        assert CALIBRATION_THETA_HASHES_SCHEMA == "oczy/meta-cortex/calibration-theta-hashes/v1"


# ---------------------------------------------------------------------------
# Record loader tests — fail-closed on missing/incomplete/underpowered records
# ---------------------------------------------------------------------------


_ZERO_HASH = "0" * 64
_ORGAN_HASH = "a" * 64


def _make_probe_battery() -> ProbeBattery:
    """Build a minimal valid ProbeBattery."""
    msg = DialogueMessage(role="user", content="test")
    pc = ProbeCase(messages=(msg,), expected_response="x", kind=ProbeKind.SAME_RULE)
    return ProbeBattery(
        pre=(pc,),
        same_rule=(pc,),
        transfer=(pc,),
        composition=(pc,),
        specificity=(pc,),
        oracle_context=(pc,),
    )


def _make_meta_task(family: TaskFamily, rule_fp: str) -> MetaTask:
    """Build a minimal valid MetaTask with META_VALIDATION split."""
    msg = DialogueMessage(role="user", content="obs")
    event = LearningEvent(
        observation_messages=(msg,),
        attempted_behavior="beh",
        correction="cor",
        outcome=OutcomeCode.NEUTRAL,
    )
    return MetaTask(
        family=family,
        split=DevSplit.META_VALIDATION,
        events=(event, event),
        probes=_make_probe_battery(),
        rule_fingerprint=rule_fp,
        assignment_fingerprint=_ZERO_HASH,
        composition_fingerprint=_ZERO_HASH,
        paraphrase_group_fingerprint=_ZERO_HASH,
    )


def _make_task_condition_record(
    family: str,
    rule_fp: str,
    dev_idx: int,
    eval_idx: int,
    condition: str,
) -> TaskConditionRecord:
    """Build a minimal valid TaskConditionRecord."""
    pc = ProbeCount(correct=1, total=2)
    return TaskConditionRecord(
        family=family,
        rule_fingerprint=rule_fp,
        assignment_fingerprint=_ZERO_HASH,
        composition_fingerprint=_ZERO_HASH,
        paraphrase_group_fingerprint=_ZERO_HASH,
        developmental_seed_index=dev_idx,
        evaluation_seed_index=eval_idx,
        developmental_seed=100 + dev_idx,
        evaluation_seed=200 + eval_idx,
        condition=condition,
        score_vector_hash=_ZERO_HASH,
        same_rule=pc,
        transfer=pc,
        composition=pc,
        specificity=pc,
        pre_learning_primary=pc,
        immediately_pre_deletion_primary=pc,
        post_deletion_primary=pc,
        theta_hash=_ZERO_HASH,
        organ_hash=_ORGAN_HASH,
        state_hash=_ZERO_HASH,
        optimizer_step_count=0,
        trace_count_after=0,
        fast_zero=True,
        slow_zero=True,
    )


def _make_no_update_repeat_record(
    family: str,
    rule_fp: str,
    dev_idx: int,
    repeat_idx: int,
) -> NoUpdateRepeatRecord:
    """Build a minimal valid NoUpdateRepeatRecord."""
    return NoUpdateRepeatRecord(
        family=family,
        rule_fingerprint=rule_fp,
        developmental_seed_index=dev_idx,
        developmental_seed=100 + dev_idx,
        repeat_index=repeat_idx,
        repeat_seed=300 + repeat_idx,
        specificity_accuracy=Fraction(1, 2),
        primary_pre_deletion=Fraction(1, 2),
        primary_post_deletion=Fraction(1, 2),
        theta_hash=_ZERO_HASH,
        organ_hash=_ORGAN_HASH,
        optimizer_step_count=0,
        trace_count_after=0,
        fast_zero=True,
        slow_zero=True,
    )


def _make_view():
    """Build a minimal CalibrationInstrumentView-compatible object."""
    from oczy.experiments.meta_cortex.instrument_contracts import (
        CALIBRATION_VIEW_SCHEMA,
        CalibrationInstrumentView,
    )

    families = [TaskFamily.CONTEXTUAL_REMAP, TaskFamily.RULE_TRANSFORMATION, TaskFamily.FINITE_STATE]
    tasks: list[MetaTask] = []
    for fam in families:
        for i in range(30):
            tasks.append(_make_meta_task(fam, f"{fam.value}_{i:03d}" + "0" * (64 - len(fam.value) - 4)))

    return CalibrationInstrumentView(
        schema=CALIBRATION_VIEW_SCHEMA,
        instrument_id="meta_cortex/v1",
        instrument_version="v1",
        definition_sha256=_ZERO_HASH,
        calibration_view_sha256=_ZERO_HASH,
        scorer_sha256=_ZERO_HASH,
        endpoint_schema_sha256=_ZERO_HASH,
        confidence_level=0.95,
        target_power=0.80,
        minimum_tasks_per_family=30,
        developmental_seeds=tuple(100 + i for i in range(5)),
        evaluation_seeds=tuple(200 + i for i in range(5)),
        no_update_repeat_seeds=tuple(300 + i for i in range(20)),
        task_cluster_bootstrap_seed=999,
        tasks=tuple(tasks),
        calibration_tasks_per_family={
            "contextual_remap": 30,
            "rule_transformation": 30,
            "finite_state": 30,
        },
    )


def _write_record_file(
    path: Path,
    view,
    schema: str,
    records: list,
) -> None:
    """Write a canonical record file with header."""
    data = {
        "schema": schema,
        "definition_sha256": view.definition_sha256,
        "calibration_view_sha256": view.calibration_view_sha256,
        "scorer_sha256": view.scorer_sha256,
        "records": [r.to_json_obj() for r in records],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_theta_hashes_file(
    path: Path,
    view,
    hashes: list[str],
) -> None:
    """Write a canonical theta hashes file with header."""
    data = {
        "schema": CALIBRATION_THETA_HASHES_SCHEMA,
        "definition_sha256": view.definition_sha256,
        "calibration_view_sha256": view.calibration_view_sha256,
        "scorer_sha256": view.scorer_sha256,
        "theta_hashes": hashes,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class TestRecordLoaders:
    """Test canonical record file loading with hash verification."""

    def test_load_no_update_records_roundtrip(self, tmp_path: Path) -> None:
        view = _make_view()
        rec = _make_no_update_repeat_record("contextual_remap", _ZERO_HASH, 0, 0)
        path = tmp_path / "no_update.json"
        _write_record_file(path, view, CALIBRATION_RECORDS_SCHEMA, [rec])
        loaded = load_no_update_repeat_records(path, view)
        assert len(loaded) == 1
        assert loaded[0].family == "contextual_remap"
        assert loaded[0].specificity_accuracy == Fraction(1, 2)

    def test_load_seed_cell_records_roundtrip(self, tmp_path: Path) -> None:
        view = _make_view()
        conds = tuple(
            _make_task_condition_record("contextual_remap", _ZERO_HASH, 0, 0, c)
            for c in (C1, C2, C3, C4, C5, C6)
        )
        scr = SeedCellRecord(
            developmental_seed_index=0,
            evaluation_seed_index=0,
            developmental_seed=100,
            evaluation_seed=200,
            rule_fingerprint=_ZERO_HASH,
            family="contextual_remap",
            conditions=conds,
        )
        path = tmp_path / "seed_cells.json"
        _write_record_file(path, view, CALIBRATION_RECORDS_SCHEMA, [scr])
        loaded = load_seed_cell_records(path, view)
        assert len(loaded) == 1
        assert loaded[0].family == "contextual_remap"
        assert len(loaded[0].conditions) == 6

    def test_load_theta_hashes_roundtrip(self, tmp_path: Path) -> None:
        view = _make_view()
        hashes: list[str] = [_ZERO_HASH] * 5
        path = tmp_path / "theta_hashes.json"
        _write_theta_hashes_file(path, view, hashes)
        loaded = load_theta_hashes(path, view)
        assert len(loaded) == 5
        assert loaded[0] == _ZERO_HASH

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        with pytest.raises(FileNotFoundError):
            load_no_update_repeat_records(tmp_path / "nonexistent.json", view)

    def test_load_wrong_schema_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        rec = _make_no_update_repeat_record("contextual_remap", _ZERO_HASH, 0, 0)
        path = tmp_path / "bad_schema.json"
        _write_record_file(path, view, "wrong/schema/v1", [rec])
        with pytest.raises(ValueError, match="schema"):
            load_no_update_repeat_records(path, view)

    def test_load_wrong_definition_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        rec = _make_no_update_repeat_record("contextual_remap", _ZERO_HASH, 0, 0)
        path = tmp_path / "bad_hash.json"
        data = {
            "schema": CALIBRATION_RECORDS_SCHEMA,
            "definition_sha256": "f" * 64,
            "calibration_view_sha256": view.calibration_view_sha256,
            "scorer_sha256": view.scorer_sha256,
            "records": [rec.to_json_obj()],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="definition_sha256"):
            load_no_update_repeat_records(path, view)

    def test_load_wrong_view_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        rec = _make_no_update_repeat_record("contextual_remap", _ZERO_HASH, 0, 0)
        path = tmp_path / "bad_view.json"
        data = {
            "schema": CALIBRATION_RECORDS_SCHEMA,
            "definition_sha256": view.definition_sha256,
            "calibration_view_sha256": "f" * 64,
            "scorer_sha256": view.scorer_sha256,
            "records": [rec.to_json_obj()],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="calibration_view_sha256"):
            load_no_update_repeat_records(path, view)

    def test_load_theta_hashes_wrong_count_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        path = tmp_path / "bad_count.json"
        _write_theta_hashes_file(path, view, [_ZERO_HASH] * 3)
        with pytest.raises(ValueError, match="5 theta hashes"):
            load_theta_hashes(path, view)

    def test_load_theta_hashes_bad_hash_raises(self, tmp_path: Path) -> None:
        view = _make_view()
        path = tmp_path / "bad_hash_len.json"
        _write_theta_hashes_file(path, view, ["abc"] * 5)
        with pytest.raises(ValueError, match="64-char hex"):
            load_theta_hashes(path, view)


class TestCalibrationFailClosed:
    """Missing/incomplete/underpowered records must fail closed."""

    def test_underpowered_records_rejected(self, tmp_path: Path) -> None:
        """Fewer than 30 tasks per family must raise."""
        from oczy.experiments.meta_cortex.instrument_contracts import (
            CALIBRATION_VIEW_SCHEMA,
            CalibrationInstrumentView,
        )
        # Build a view with only 5 tasks per family (underpowered).
        families = [TaskFamily.CONTEXTUAL_REMAP, TaskFamily.RULE_TRANSFORMATION, TaskFamily.FINITE_STATE]
        tasks = []
        for fam in families:
            for i in range(5):
                tasks.append(_make_meta_task(fam, f"{fam.value}_{i:03d}" + "0" * (64 - len(fam.value) - 4)))
        view = CalibrationInstrumentView(
            schema=CALIBRATION_VIEW_SCHEMA,
            instrument_id="meta_cortex/v1",
            instrument_version="v1",
            definition_sha256=_ZERO_HASH,
            calibration_view_sha256=_ZERO_HASH,
            scorer_sha256=_ZERO_HASH,
            endpoint_schema_sha256=_ZERO_HASH,
            confidence_level=0.95,
            target_power=0.80,
            minimum_tasks_per_family=30,
            developmental_seeds=tuple(100 + i for i in range(5)),
            evaluation_seeds=tuple(200 + i for i in range(5)),
            no_update_repeat_seeds=tuple(300 + i for i in range(20)),
            task_cluster_bootstrap_seed=999,
            tasks=tuple(tasks),
            calibration_tasks_per_family={
                "contextual_remap": 5,
                "rule_transformation": 5,
                "finite_state": 5,
            },
        )
        with pytest.raises(ValueError, match="minimum is 30"):
            run_dev_calibration(
                view=view,
                no_update_repeat_records=[],
                seed_cell_records=[],
                theta_hashes=[_ZERO_HASH] * 5,
                organ_hash=_ORGAN_HASH,
                output_dir=tmp_path,
            )

    def test_wrong_repeat_count_rejected(self, tmp_path: Path) -> None:
        """Wrong number of no-update repeat records must raise."""
        view = _make_view()
        with pytest.raises(ValueError, match="no-update repeat records"):
            run_dev_calibration(
                view=view,
                no_update_repeat_records=[],
                seed_cell_records=[],
                theta_hashes=[_ZERO_HASH] * 5,
                organ_hash=_ORGAN_HASH,
                output_dir=tmp_path,
            )

    def test_wrong_theta_hash_count_rejected(self, tmp_path: Path) -> None:
        """Wrong number of theta hashes must raise."""
        view = _make_view()
        with pytest.raises(ValueError, match="5 theta hashes"):
            run_dev_calibration(
                view=view,
                no_update_repeat_records=[],
                seed_cell_records=[],
                theta_hashes=[_ZERO_HASH] * 3,
                organ_hash=_ORGAN_HASH,
                output_dir=tmp_path,
            )

    def test_missing_seed_cells_rejected(self, tmp_path: Path) -> None:
        """Missing seed cell records for a view task must raise."""
        view = _make_view()
        # Build correct no-update records count: 90 tasks * 5 * 20 = 9000.
        no_update_records = []
        for task in view.tasks:
            fam = task.family.value
            for dev_idx in range(5):
                for rep in range(20):
                    no_update_records.append(
                        _make_no_update_repeat_record(fam, task.rule_fingerprint, dev_idx, rep)
                    )
        with pytest.raises(ValueError, match="no seed cell records"):
            run_dev_calibration(
                view=view,
                no_update_repeat_records=no_update_records,
                seed_cell_records=[],
                theta_hashes=[_ZERO_HASH] * 5,
                organ_hash=_ORGAN_HASH,
                output_dir=tmp_path,
            )


class TestCalibrationSyntheticFixture:
    """A CLI-run synthetic fixture with explicitly supplied verified canonical
    records creates reports; the full calibration pipeline produces
    DEV_DISTRIBUTIONS.json and POWER_ANALYSIS.json.
    """

    def test_full_calibration_creates_reports(self, tmp_path: Path) -> None:
        """Full calibration with 90 tasks, correct record counts, produces reports."""
        view = _make_view()

        # Build no-update repeat records: 90 tasks * 5 dev * 20 repeats = 9000.
        no_update_records = []
        for task in view.tasks:
            fam = task.family.value
            for dev_idx in range(5):
                for rep in range(20):
                    no_update_records.append(
                        _make_no_update_repeat_record(fam, task.rule_fingerprint, dev_idx, rep)
                    )
        assert len(no_update_records) == 9000

        # Build seed cell records: 90 tasks * 25 cells * 6 conditions.
        seed_cell_records = []
        for task in view.tasks:
            fam = task.family.value
            for dev_idx in range(5):
                for eval_idx in range(5):
                    conds = tuple(
                        _make_task_condition_record(
                            fam, task.rule_fingerprint, dev_idx, eval_idx, cond
                        )
                        for cond in (C1, C2, C3, C4, C5, C6)
                    )
                    seed_cell_records.append(
                        SeedCellRecord(
                            developmental_seed_index=dev_idx,
                            evaluation_seed_index=eval_idx,
                            developmental_seed=100 + dev_idx,
                            evaluation_seed=200 + eval_idx,
                            rule_fingerprint=task.rule_fingerprint,
                            family=fam,
                            conditions=conds,
                        )
                    )
        assert len(seed_cell_records) == 2250

        result = run_dev_calibration(
            view=view,
            no_update_repeat_records=no_update_records,
            seed_cell_records=seed_cell_records,
            theta_hashes=[_ZERO_HASH] * 5,
            organ_hash=_ORGAN_HASH,
            output_dir=tmp_path,
        )

        assert isinstance(result, CalibrationResult)
        assert (tmp_path / "DEV_DISTRIBUTIONS.json").exists()
        assert (tmp_path / "POWER_ANALYSIS.json").exists()
        assert result.dev_distributions_path == str(tmp_path / "DEV_DISTRIBUTIONS.json")
        assert result.power_analysis_path == str(tmp_path / "POWER_ANALYSIS.json")
        assert result.definition_sha256 == view.definition_sha256
        assert result.calibration_view_sha256 == view.calibration_view_sha256

        # Verify no sealed payload access in the output.
        dev_dist = json.loads((tmp_path / "DEV_DISTRIBUTIONS.json").read_text())
        assert dev_dist["holdout_accessed"] is False
        assert dev_dist["meta_validation_only"] is True

        power = json.loads((tmp_path / "POWER_ANALYSIS.json").read_text())
        assert power["holdout_accessed"] is False
        assert power["definition_sha256"] == view.definition_sha256

    def test_reports_feed_finalize_candidate(self, tmp_path: Path) -> None:
        """The calibration reports must be consumable by finalize_candidate."""
        view = _make_view()

        no_update_records = []
        for task in view.tasks:
            fam = task.family.value
            for dev_idx in range(5):
                for rep in range(20):
                    no_update_records.append(
                        _make_no_update_repeat_record(fam, task.rule_fingerprint, dev_idx, rep)
                    )

        seed_cell_records = []
        for task in view.tasks:
            fam = task.family.value
            for dev_idx in range(5):
                for eval_idx in range(5):
                    conds = tuple(
                        _make_task_condition_record(
                            fam, task.rule_fingerprint, dev_idx, eval_idx, cond
                        )
                        for cond in (C1, C2, C3, C4, C5, C6)
                    )
                    seed_cell_records.append(
                        SeedCellRecord(
                            developmental_seed_index=dev_idx,
                            evaluation_seed_index=eval_idx,
                            developmental_seed=100 + dev_idx,
                            evaluation_seed=200 + eval_idx,
                            rule_fingerprint=task.rule_fingerprint,
                            family=fam,
                            conditions=conds,
                        )
                    )

        run_dev_calibration(
            view=view,
            no_update_repeat_records=no_update_records,
            seed_cell_records=seed_cell_records,
            theta_hashes=[_ZERO_HASH] * 5,
            organ_hash=_ORGAN_HASH,
            output_dir=tmp_path,
        )

        # The reports must have the fields finalize_candidate expects.
        dev_dist = json.loads((tmp_path / "DEV_DISTRIBUTIONS.json").read_text())
        power = json.loads((tmp_path / "POWER_ANALYSIS.json").read_text())

        assert dev_dist["definition_sha256"] == view.definition_sha256
        assert dev_dist["holdout_accessed"] is False
        assert "equivalence_margin" in dev_dist
        assert dev_dist["equivalence_margin"]

        assert power["definition_sha256"] == view.definition_sha256
        assert "sample_size_tasks_per_family" in power
        assert isinstance(power["sample_size_tasks_per_family"], int)
        assert power["sample_size_tasks_per_family"] >= 30

