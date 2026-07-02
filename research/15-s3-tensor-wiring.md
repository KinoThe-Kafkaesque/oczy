# 15 — Wire surviving organs to tensors (Sprint 3 / S3.3, Goal 3)

**Pre-registered 2026-07-02** (human-approved sprint setup).
Conditional spec: applies to exactly the components research/14 marks KEEP.
If research/14 marks nothing KEEP (all no-effect or retrieval), this spec is
recorded as VACUOUS — that outcome would itself close Goal 3's question
honestly. Agents MUST NOT edit this spec.

## Problem

Today organ outputs influence behavior by reranking label strings in
`organism.py:_rank_answer`. That is not the thesis. Goal 3 says organ outputs
must become *changed dynamics*: tensors entering the cortex state or the
driver's injection surfaces.

## Requirement (per surviving component)

1. Output enters one of the sanctioned tensor surfaces ONLY:
   cortex warm/cold state update, per-layer cvec, or written KV entries
   (`encode_kv`/`generate_with_kv`). No string reranking, no logit bias.
2. The string-ranker path for that component is deleted in the same change —
   not left as a fallback (dual paths make the ablation unattributable).
3. Unit tests prove the tensor path is exercised (mock driver assertion that
   the surface was written) and that no `_rank_answer` coupling remains.

## Hypothesis & acceptance (per component)

**H-WIRE(c):** component c wired to tensors preserves or improves its
research/14 additive effect.

Re-run the research/14 M2 additive arm for c (same protocol, seeds, split):

- **Accept:** Δ_tensor(c) ≥ Δ_string(c) − 0.05 AND Δ_tensor(c) > 0 with 95%
  CI excluding 0.
- **Refute:** otherwise — c is then either kept as RETRIEVAL-BASELINE (string
  path restored, honestly labeled) if it was one, or ARCHIVED. A refutation
  here is expected for components whose effect was really retrieval in
  disguise; that is the point of the test.

## Reporting

Per-component before/after table (string vs tensor Δ, mean±CI), diff summary
of removed string-ranker code, tests added; log to
`experiments_logs/<date>_s3_3_tensor_wiring.md` quoting this spec.
