> **INVALIDATION NOTICE — 2026-07-01**  
> **Classification: PARTIAL**  
> **Reason:** The bug diagnosis and fix description (Sections "Root Causes," "Verification," "Remaining Work," "Key Files Changed") are VALID. The "Curriculum Impact" table (line ~40) reports leakage-era numbers (Stage 2 scope=1.00, Stage 5 retention=1.00, etc.) that are SUPERSEDED by the honest post-leakage baseline — those claims included test-set leakage removed on 2026-07-01. The honest baseline measures the same configuration at Stage 2 scope=0.69, Stage 5 scope=0.92.  
> **See:** `2026-07-01_honest_post_leakage_baseline.md` for the current reference point; `2026-07-01_stage5_scope_dsi_benchmarks.md` for additional superseded Stage 5 claims.

# Scope-Slot Reranker Fix: Three Bugs Found and Fixed

**Date**: 2026-06-30
**Branch**: `autoresearch/session-20260625`
**Commits**: `43cfc9f`, `091046c`, `e316cb1`

## Summary

Three compounding bugs in the scope-slot reranker prevented it from ever functioning correctly. After fixing all three, the curriculum metrics improved dramatically across all stages, with Stage 5 cross-domain scope improving from **0.00 → 0.50** (3/6 probes passing).

## Root Causes

### Bug 1: `last_token_only=True` in `_scope_key` (collapse all slots into one)

`_scope_key()` called `driver.peek_embedding(request)` with the default `last_token_only=True`. This embeds only the **last token** of the request. All curriculum requests end with `.`, so every request got the identical embedding of `.` (cosine similarity = 1.0000). The slot store collapsed all 44+ episodes into a single slot, and each correction overwrote the previous label.

**Fix**: `peek_embedding(request, last_token_only=False)` uses the mean-pooled whole-request embedding. Different requests now get distinct embeddings (cosine sim ~0.6).

### Bug 2: `_MAX_SLOTS=16` too small for 6-stage curriculum

The slot store hit its capacity limit after Stage 1 (8 + 8 = 16 slots). Stage 2 corrections then overwrote Stage 0/1 labels via the nearest-slot averaging path. The Stage 2 technical labels ("system error log", "computer file", etc.) were lost, so Stage 5 scope probes couldn't find them.

**Fix**: `_MAX_SLOTS=64`. The Exp04 experiment is unaffected (it only uses Stage 2, max 16 slots).

### Bug 3: `_ALLOC_THRESHOLD=0.85` used for label retrieval (reranker never fired)

`_scope_label_for()` filtered retrieved labels by `sim >= _ALLOC_THRESHOLD` (0.85). But mean-pooled embeddings of related-but-different requests have cosine sim ~0.3-0.65 (e.g. "Log the server crash in the system." vs "Log the runtime error." = 0.6438). No labels ever passed the 0.85 threshold, so the reranker never boosted any candidate.

**Fix**: Added a separate `_RETRIEVE_THRESHOLD=0.3` for label retrieval. Slot allocation still uses 0.85 (prevents creating duplicate slots for near-identical requests). The top-k limit (`scope_rerank_topk=3`) and the similarity-weighted boost formula (`weight * scope_sim * overlap`) handle noise from unrelated labels.

### Config change: `scope_rerank_topk=1 → 3`

With `topk=1`, only the single most-cosinely-similar scope label was returned. This was often the wrong label (e.g. "ml experiment run" instead of "ml model" for "Train the model on the image dataset."). With `topk=3`, the correct technical sense label is included in the top-3 and can boost its candidate.

`sense_split=True` was tested but rejected: it improved Stage 5 scope equally but hurt Stage 5 retention (1.00 → 0.83) by excluding the ambiguous word from retention/transfer probe scoring.

## Curriculum Impact

Real LFM2.5-1.2B-Instruct Q4 GGUF driver, semantic scoring:

| Metric | Before fixes | After fixes | Change |
|--------|-------------|-------------|--------|
| Stage 0 retention | 0.12 | 0.88 | +0.76 |
| Stage 1 transfer | 0.25 | 0.75 | +0.50 |
| Stage 2 scope | 0.12 | 1.00 | +0.88 |
| Stage 2 retention | 0.25 | 0.88 | +0.63 |
| Stage 3 scope | 0.00 | 1.00 | +1.00 |
| Stage 3 transfer | 0.25 | 0.25 | — |
| Stage 4 retention | 0.10 | 1.00 | +0.90 |
| Stage 5 retention | 0.17 | 1.00 | +0.83 |
| **Stage 5 scope** | **0.00** | **0.50** | **+0.50** |

## Verification

- 7/7 experiments accepted (preserved throughout)
- 441 tests pass
- `scope_selectivity_index=0.625` (unchanged)
- ASI metrics stable

## Remaining Work

Stage 5 scope = 0.50 (3/6). The remaining 3 failing scope probes need either:
1. Higher `scope_rerank_weight` to overcome fast-weight bias from Stage 5 corrections
2. Sense-specific concept matching in the identity hypernetwork
3. Semantic concept-to-label projection (learned during corrections)

## Key Files Changed

- `src/oczy/experiments/organism.py`: `_scope_key` fix, `_scope_label_for` threshold, reranker defaults
- `src/oczy/experiments/scope_selectivity_stressor.py`: `_MAX_SLOTS=64`, `_RETRIEVE_THRESHOLD=0.3`
