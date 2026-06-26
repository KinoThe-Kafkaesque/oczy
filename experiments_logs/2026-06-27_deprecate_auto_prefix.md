# Deprecating `--auto-prefix` in the Multi-Fact Stressor — 2026-06-27

## Question

Now that `--use-agent-prefix` validates the live `CortexAgent` prefix path,
should the stressor-only `--auto-prefix` wrapper be deprecated?

## Decision

Yes. `--auto-prefix` is kept for backward compatibility but now emits a clear
deprecation ASI pointing users to `--use-agent-prefix`. The wrapper path will
likely be removed in a future cleanup.

## Changes

- Added deprecation ASI in `multi_fact_stressor.py` when `--auto-prefix` is used.
- Updated `--auto-prefix` help text to mark it `[DEPRECATED]`.
- Reconstructed the `argparse` block after earlier edits accidentally duplicated
  entries for `--use-agent-prefix` and `--auto-consolidate`.
- Updated tests:
  - `test_multi_fact_stressor_auto_prefix_mock` asserts the deprecation ASI.
  - `test_multi_fact_stressor_auto_prefix_empty_fallback_mock` asserts the
    deprecation ASI even when no prefix is derived.

## Results

- Stressor tests: 16 passed.
- Fast suite: 310 passed, 24 deselected.
- `ruff check` clean.
- Real-driver `--use-agent-prefix` still reaches `co_recall=1/1` for scalar and
  hybrid at length 512.
- Benchmark `code_qa_accuracy=1.0` (run #99).

## Rationale

- Avoid maintaining two prefix mechanisms.
- The live agent path is more representative of production behavior because it
  exercises `CortexAgent.articulate()` rather than the stressor wrapper.
- Deprecation rather than removal preserves any external scripts relying on
  `--auto-prefix` while signaling the migration path.

## Next steps

- In a future run, remove `_derive_prefix_from_hippocampus()` and the
  `--auto-prefix` code entirely once `--use-agent-prefix` has been the default
  path for several benchmark cycles.
- Continue with IdentityHypernetwork adapter effect measurement or generalize
  keyword extraction.

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`

## Commits

- `c6a5beb` — Deprecate `--auto-prefix`.
- `adb2d91` — Update SUMMARY.md with run #99 result.

## Run

Run #99: benchmark `code_qa_accuracy=1.0`, fast suite `310 passed`.
