# Knowledge Core Expansion: 280KB → 1M Parameters

**Date**: 2026-06-29
**Branch**: `autoresearch/session-20260625`

## What Changed

Expanded the organism's knowledge core from ~29K parameters (~280KB) to ~1.05M parameters (~8.4MB), a 36× growth.

### Dimension changes

| Component | Dimension | Old | New | Params |
|-----------|-----------|-----|-----|--------|
| Experience Autoencoder `_A` | (RESIDUAL_DIM, NUM_SOURCES×MAX_VOCAB) | (28, 1024) | (128, 8192) | 1,048,576 |
| Identity Hypernetwork `W` | (concepts, 4×latent_dim) | (14, 32) | (14, 128) | 1,792 |
| Plastic Cortex `hidden_dim` | RNN hidden state | 8 | 64 | ~5,000 |
| **Total unique params** | | ~29,000 | **~1,055,000** | |

### Constants changed

| Constant | Old | New | File |
|----------|-----|-----|------|
| `RESIDUAL_DIM` | 28 | 128 | `autoencoder.py` |
| `MAX_VOCAB` | 256 | 2048 | `autoencoder.py` |
| `LATENT_DIM` | 32 | 132 | `autoencoder.py` (auto-computed) |
| `DECODE_SPARSITY` | 10 | 30 | `autoencoder.py` |
| `HEBBIAN_LR` | 0.01 | 0.05 | `autoencoder.py` |
| `latent_dim` (identity) | 8 | 32 | `hypernet.py` |
| `hidden_dim` (plastic cortex) | 8 | 64 | `cortex.py` |

## Curriculum Results

### Stage-by-stage comparison

| Stage | 280KB core | 1M param core | Change |
|-------|-----------|---------------|--------|
| Stage 0 (8 episodes) | 8/8 | 8/8 | — |
| Stage 1 (8 episodes) | 8/8 | 8/8 | — |
| Stage 2 (8 episodes) | 8/8 | 8/8 | — |
| Stage 3 (4 episodes) | 4/4 | 4/4 | — |
| Stage 4 (10 episodes) | 10/10 | 10/10 | — |
| Stage 5 (6 episodes) | 6/6 | 6/6 | — |
| Stage 5 scope (pre) | 0.0 | 0.0 | — |
| Stage 5 retention (pre) | 0.0 | 0.0 | — |

### Memory footprint

| Stage | 280KB core | 1M param core |
|-------|-----------|---------------|
| After Stage 0 | 254,943 B | 8,478,377 B |
| After Stage 5 | 286,920 B | 8,517,457 B |
| Growth during curriculum | +47,726 B | +39,080 B |

## Analysis

### The 36× expansion had zero effect on curriculum performance

All stage scores, scope, and retention metrics are identical between the 280KB and 1M param cores. The 1M param core's 8.4MB sensing matrix provides no behavioral advantage over the 280KB core's 229KB matrix.

### Why: the bottleneck is architectural, not capacity

The experience autoencoder's sensing matrix `_A` is a random projection that compresses bag-of-words episode features into a residual vector. The expanded matrix gives 128 residual dimensions (vs 28) and 2048 vocab slots (vs 256), but:

1. **The downstream organs don't use the residual.** The identity hypernetwork learns through text-based concept extraction (`_extract_first_concept`), not through the autoencoder's Δz vector. The plastic cortex learns through token-label associations, not through the residual. The autoencoder's output flows nowhere behaviorally — it's a side channel for memory tracking.

2. **The scope-slot reranker uses LFM2.5 embeddings.** The reranker retrieves corrected labels via `driver.peek_embedding(request)` — the LFM2.5's own 2048-dim hidden states, not the autoencoder's 128-dim residual. Expanding the autoencoder doesn't affect the reranker.

3. **The Hebbian learning doesn't improve reconstruction.** With the larger matrix, the OMP decoder can't benefit from Hebbian updates because the 128-dim residual is underdetermined with 30-sparse recovery. The reconstruction error stays constant or increases, meaning the autoencoder's self-training loop is ineffective at this scale.

4. **The vocab cap (2048) is still not the bottleneck.** The curriculum uses ~30 unique words. Even the old 256-vocab cap was sufficient. The 2048 cap helps only if the curriculum vocabulary grows 8×, which it doesn't.

### What would actually help

The `scope=0.0` gap (no proactive cross-domain disambiguation) is caused by:

1. **The scope-slot reranker only fires after corrections.** It stores corrected labels keyed by request embedding, but can only retrieve them when the same (or similar) request is seen again. For novel cross-domain requests, there's no stored slot.

2. **The identity hypernetwork globally boosts concept tokens.** It doesn't distinguish between senses of the same word — "file" in "clerk" context vs "file" in "tool" context both map to the same concept token.

3. **The plastic cortex has only 2 labels.** It can't represent the 10+ senses needed for cross-domain disambiguation.

To improve `scope`, the architecture needs to change, not just the capacity:
- Store scope slots from earlier stages that generalize to cross-domain requests
- Add sense-specific concept tokens to the identity hypernetwork
- Expand the plastic cortex's label vocabulary
- Connect the autoencoder's residual to the identity hypernetwork's update path

## Verification

- 261 tests passed (6 warnings)
- `autoresearch.sh`: 7/7 experiments accepted
- `bounded_growth_m1_ratio`: 0.073 → 0.002 (still accepted; larger baseline makes relative growth smaller)
- Curriculum: all stages pass, Stage 5 = 6/6, scope = 0.0

## Update: scope-slot fix resolves architectural bottleneck (2026-06-30)

The report's central conclusion — that the bottleneck is architectural, not capacity — was correct. The 36× core expansion produced zero behavioral change because the downstream organs don't consume the autoencoder's residual. But the specific architectural bottleneck behind the `scope=0.0` gap has now been identified and fixed, and it was not where this report looked.

### The real bottleneck: three bugs in the scope-slot reranker

The scope-slot reranker (which uses LFM2.5 embeddings, as point 2 of the analysis correctly noted) was silently non-functional due to three compounding bugs:

1. **`_scope_key` used `last_token_only=True`** (commit `43cfc9f`): `peek_embedding()` with the default `last_token_only=True` embeds only the last token. All curriculum requests end with `.`, so every request got identical embeddings (cosine sim=1.0) and all 44+ episodes collapsed into a single slot. Fix: `last_token_only=False` for mean-pooled whole-request embeddings.

2. **`_MAX_SLOTS=16` too small** (commit `091046c`): the slot store filled after Stage 1 (8+8=16), so Stage 2 corrections overwrote Stage 0/1 labels. Fix: `_MAX_SLOTS=64`.

3. **`_ALLOC_THRESHOLD=0.85` used for label retrieval** (commit `091046c`): mean-pooled embeddings of related-but-different requests have cosine sim ~0.3–0.65, well below 0.85. No labels were ever returned, so the reranker never fired. Fix: a separate `_RETRIEVE_THRESHOLD=0.3` for label retrieval.

A fourth change (commit `e316cb1`) raised `scope_rerank_topk` from 1 to 3, giving the correct technical sense a chance against the single most-similar (often wrong) label.

### Updated curriculum results (real LFM2.5-1.2B Q4 driver, semantic scoring, topk=3)

| Stage | 1M core (2026-06-29) | 1M core + scope fix (2026-06-30) | Change |
|-------|---------------------|----------------------------------|--------|
| Stage 0 (8) | 8/8, retention=0.0 | 8/8, retention=0.88 | +0.88 |
| Stage 1 (8) | 8/8, transfer=0.25 | 7/8, transfer=0.75 | +0.50 |
| Stage 2 (8) | 8/8, scope=0.0, retention=0.0 | 8/8, scope=1.00, retention=0.88 | scope +1.00 |
| Stage 3 (4) | 4/4, scope=0.0 | 4/4, scope=1.00, transfer=0.25 | scope +1.00 |
| Stage 4 (10) | 10/10, retention=0.0 | 10/10, retention=1.00 | +1.00 |
| Stage 5 (6) | 6/6, scope=0.0, retention=0.0 | 6/6, scope=0.50, retention=1.00 | scope +0.50, retention +1.00 |

The `scope=0.0` gap that motivated the "What would actually help" list is gone: Stage 2 scope 0.0→1.00, Stage 3 scope 0.0→1.00, Stage 5 scope 0.0→0.50. Retention metrics that were flat at 0.0 are now 0.88–1.00 across stages.

### What this means for the 1M param core

The 1M param core itself still has no direct behavioral effect — the analysis above holds. But the downstream reranker that consumes LFM2.5 embeddings (not the autoencoder residual) now works correctly, and the curriculum metrics it influences have moved dramatically. The capacity expansion and the reranker fix are orthogonal: the core's residual remains a side channel for memory tracking, while the reranker's embedding-based scope slots are what drive cross-domain disambiguation.

### Verification

- 441 tests pass (up from 261).
- 7/7 experiments accepted (preserved).
- `scope_selectivity_index=0.625`, `bounded_growth_m1_ratio=0.002079`, `metabolism_drift_delta=0.1016`, `layer_l_silhouette_gap=0.116`, `marker_free_uptake_gap=1.0`.

See `2026-06-30_scope_slot_reranker_fix.md` for the detailed fix report.


## Update: bilinear policy head fix (2026-06-30)

The cortex dimension benchmark (extended to d_cortex ∈ {2,...,512})
revealed that `d_cortex` had no effect on curriculum performance because
the warm_state was architecturally disconnected from candidate
discrimination in the policy head. Four disconnections were fixed
(commit `76c6105`):

1. Scope-slot warm_state not restored before policy scoring.
2. Policy features not L2-normalized (2048-dim hidden drowned d_cortex-dim warm).
3. Linear policy head cannot discriminate — warm is constant across candidates. Fixed with bilinear term: `warm @ W_bilinear @ hidden_i`.
4. Warm_state not captured when `use_cortex_lm_answer=False` — all slots were None.

Unit tests confirm the bilinear term discriminates candidates and varies
with d_cortex. The 1M param core's residual still has no direct
behavioral effect — the policy head's bilinear term uses the cortex
warm_state (d_cortex dims), not the autoencoder residual. Curriculum
results unchanged because `policy_delta` (weight=1.0) is dominated by
scope-rerank boost (weight=2.0).

Full details: `2026-06-30_cortex_dim_benchmark.md` (Update section).