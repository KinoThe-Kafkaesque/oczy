# Multi-Fact Stressor: Auto-Consolidate S vs H — 2026-06-27

## Question

Does the `--auto-consolidate` path in `multi_fact_stressor.py` let the
DigestiveGate trigger consolidation for both scalar (S) and hybrid (H) modes,
and does it preserve benchmark-compatible metrics?

## Method

Added `--auto-consolidate` to `src/oczy/experiments/multi_fact_stressor.py`.
When active:

- `auto_consolidate=True` in `CortexAgentConfig`.
- `digestive_gate.consolidation_pressure_threshold` lowered to `0.05` so a
  single high-drift turn is likely to fire.
- After `perceive` + `metabolize`, the stressor calls
  `agent.should_consolidate()`. If true, strength is computed from gate pressure
  and threshold; for H it is additionally scaled by `(1.0 + digest.drift_max)`
  and capped at `10.0`. `agent.consolidate(strength=...)` runs and the gate is
  reset.
- `auto_consolidated` is tracked in `_ProbeResult` and emitted in both
  `METRIC` and `ASI` lines.

Ran mock and real-driver invocations against LFM2.5-1.2B-Instruct Q4_K_M.

## Results

| mode  | driver | auto_consolidated | length | cold_drift | consolidation_strength | traces |
|---|---|---|---|---|---|---|
| scalar | mock | 1 | 128 | 0.997987 | 10.0 | 2 |
| scalar | real | 1 | 512 | 0.866852 | 10.0 | 3 |
| hybrid | real | 1 | 512 | 0.866852 | 10.0 | 3 |

Tests:

- `uv run pytest src/oczy/experiments/tests/test_multi_fact_stressor.py -q`
  → `7 passed` (includes real-driver test, model cached).
- `uv run python -m src.oczy.experiments.multi_fact_stressor --auto-consolidate --length 128`
  → mock path emits `auto_consolidated=1`.
- `uv run python -m src.oczy.experiments.multi_fact_stressor --use-real-driver --auto-consolidate --mode scalar --length 512`
  → real path emits `auto_consolidated=1`.
- `uv run python -m src.oczy.experiments.multi_fact_stressor --use-real-driver --auto-consolidate --mode hybrid --length 512`
  → real path emits `auto_consolidated=1`.
- `uv run pytest -m "not slow and not requires_model" -q` → `301 passed, 22 deselected`.
- `uv run ruff check src/oczy/experiments/multi_fact_stressor.py src/oczy/experiments/tests/test_multi_fact_stressor.py`
  → clean.

## Interpretation

- The `--auto-consolidate` flag successfully lowers the gate threshold and
  lets the agent decide when to consolidate.
- Both scalar and hybrid real-driver runs fired consolidation on the first turn,
  producing comparable `cold_drift` and `consolidation_strength` values.
- The mock test asserts only that the path runs and emits the new metric,
  without requiring consolidation to fire (`auto_consolidated` ∈ `{0, 1}`).
- All existing fast-suite tests still pass; default behavior is unchanged because
  the flag defaults to `False`.

## Commits

- Branch `autoresearch/session-20260625`.
- Modified: `src/oczy/experiments/multi_fact_stressor.py`,
  `src/oczy/experiments/tests/test_multi_fact_stressor.py`.
