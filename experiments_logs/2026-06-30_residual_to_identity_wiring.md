# Residual-to-Identity Wiring Report

**Date**: 2026-06-30
**Segment**: `residual-to-identity-wiring` (segment 10)
**Runs**: #183–#188 (6 kept, 1 transient crash)

## Goal

Wire the experience autoencoder's residual vector into the identity hypernetwork's concept-scoring path so the 1M-param sensing matrix (`_A`, 128×8192) actually shapes behavior, and cross-domain scope improves from 0.0.

## What was built

### 1. `_W_residual` matrix in IdentityHypernetwork

- **`generate_adapters(residual=...)`**: projects residual through `_W_residual` and adds `residual_scale * (_W_residual @ residual)` to concept scores.
- **`update_identity(lesson, residual=...)`**: applies Hebbian learning to `_W_residual[target_idx]` — moves the row toward the residual direction so similar future residuals boost the same concept.
- **`grow_vocab`**: extends and prunes `_W_residual` in lockstep with `W`.
- **`__setstate__`**: backward-compatible defaults for all new fields (`_W_residual`, `_residual_dim`, `_residual_lr`, `_residual_scale`).
- **`residual_scale=0.1`**: controls contribution magnitude. Raw contribution was 3× existing scores; scaled to 30%.

### 2. Hidden-delta encoding (`use_hidden_delta=True`)

- Captures `_last_hidden` from `cortex_agent` at answer time (dimension gate ≥ 64 to filter shim's 8-dim random vectors).
- Passes `hidden_delta` to `autoencoder.encode()` at both answer and correction time.
- Autoencoder routes to `encode_hidden_delta()` when `hidden_delta` present, falls back to text path otherwise.
- Gates residual-to-identity path on hidden state availability — bag-of-words residuals add noise, not signal.
- `LMBackendAgent` explicitly passes `residual=None` (no contextualized hidden states).

### 3. Autoencoder training (`train_step`)

- Added `autoencoder.train_step(episode)` call in both `OrganismAgent` and `LMBackendAgent` correction paths.
- Trains `_A_hidden` sensing matrix via Hebbian learning so future residuals preserve discriminative structure instead of being random projections.

## Results

| Metric | Before | After |
|--------|--------|-------|
| `experiments_accepted_count` | 7/7 | 7/7 |
| `scope_selectivity_index` (Exp04) | 0.625 | 0.625 |
| `bounded_growth_m1_ratio` (Exp06) | 0.002 | 0.002 |
| Stage 5 scope (organism curriculum) | 0.0 | 0.0 |
| Stage 5 retention | 0.0 | 0.17 |
| Test suite | 283 passed | 283 passed |

## Key finding: the architectural bottleneck

The residual-to-identity wiring is **architecturally complete and correct** but does **not** improve cross-domain scope. The reason is a downstream disconnect:

1. The identity hypernetwork's `concept_scores` only boost labels whose tokens directly match concept names (e.g. "profile", "business", "vertical").
2. The curriculum's labels are natural-language phrases like "the captain's journal", "submit it officially", "the map legend" — none of which contain concept vocabulary tokens.
3. Therefore `concept_scores` have **zero effect** on the final answer ranking for these labels, regardless of how good the residual-to-concept projection is.

The `_rank_answer` method at line 376 does:
```python
for token in label_tokens:
    score += float(concept_scores.get(token, 0.0))
```

Since "captain", "journal", "submit", "officially", "map", "legend" are not in `CONCEPT_VOCABULARY`, the concept scores never influence the ranking.

## What would fix this

1. **Sense-specific concept vocabulary**: Add concepts like "journal", "submit", "legend", "cell", "record", "branch", "model", "run" to `CONCEPT_VOCABULARY` so the concept scores can actually match label tokens.
2. **Semantic concept matching**: Instead of exact token match, use embedding similarity between label tokens and concept names.
3. **Concept-to-label projection**: Add a matrix that maps concept scores to label-space scores, learned during corrections.

## Update: _extract_all_concepts fix

After implementing `_extract_all_concepts` (registers ALL valid tokens from the label as concepts, not just the first), the concept scores now match multi-word labels:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Stage 1 transfer | 0.12 | 0.25 |
| Stage 2 scope | 0.00 | 0.12 |
| Stage 5 scope | 0.00 | 0.00 |
| Stage 5 retention | 0.17 | 0.17 |

Stage 2 scope improved from 0.0 to 0.12 — the concept scores are now influencing label ranking for multi-word labels. Stage 5 scope remains 0.0, indicating the cross-domain disambiguation bottleneck is deeper than concept vocabulary mismatch.

## Conclusion

The 1M-param sensing matrix now flows into the identity hypernetwork's concept-scoring path — the wiring is complete. The residual shapes concept scores, the Hebbian learning trains the sensing matrix, and `_extract_all_concepts` ensures all label tokens are registered as concepts. Stage 2 scope improved from 0.0 to 0.12, confirming the concept scores now influence multi-word label ranking. Stage 5 scope remains 0.0, indicating the cross-domain disambiguation bottleneck requires deeper architectural changes (sense-specific concepts, semantic matching, or a concept-to-label projection matrix).

**Status**: Wiring complete, 7/7 preserved, Stage 2 scope improved 0.0→0.12, Stage 5 scope still 0.0.

## Update: scope-slot reranker fix (2026-06-30)

The downstream disconnect identified above was not purely architectural — the scope-slot reranker that consumes the concept scores was silently broken by three compounding bugs. The residual-to-identity wiring itself remains architecturally complete and correct; what was failing was the reranker that sits downstream of `concept_scores` and decides which stored label to apply for a given request.

### Bugs found and fixed

1. **`_scope_key` used `last_token_only=True`** (commit `43cfc9f`): `peek_embedding()` with the default `last_token_only=True` embeds only the last token. Every curriculum request ends with `.`, so all requests produced identical embeddings (cosine sim = 1.0) and all 44+ episodes collapsed into a single slot. Fix: `last_token_only=False` for mean-pooled whole-request embeddings.
2. **`_MAX_SLOTS=16` too small** (commit `091046c`): the slot store filled after Stage 1 (8 + 8 = 16), after which Stage 2 corrections overwrote Stage 0/1 labels. Fix: `_MAX_SLOTS=64`.
3. **`_ALLOC_THRESHOLD=0.85` reused for label retrieval** (commit `091046c`): mean-pooled embeddings of related-but-different requests have cosine sim ~0.3–0.65, well below 0.85, so `_scope_label_for` never returned a label and the reranker never fired. Fix: a separate `_RETRIEVE_THRESHOLD=0.3` for label retrieval, distinct from the allocation threshold.

A fourth change (commit `e316cb1`, test defaults in `9e8eef4`) raised `scope_rerank_topk` from 1 to 3, so the correct technical sense gets a chance instead of only the single most-similar label.

### Updated curriculum results

With the reranker actually firing, the concept-scoring path (`residual → _W_residual → concept_scores`) is now functional end-to-end:

| Stage | Before fix (stale) | After fix (2026-06-30) |
|-------|--------------------|------------------------|
| Stage 0 | 7/8, retention=0.12 | 8/8, retention=0.88 |
| Stage 1 | 1/8, transfer=0.25 | 7/8, transfer=0.75 |
| Stage 2 | 3/8, scope=0.12, retention=0.25 | 8/8, scope=1.00, retention=0.88 |
| Stage 3 | 1/4, scope=0.00 | 4/4, scope=1.00, transfer=0.25 |
| Stage 4 | 7/10, retention=0.10 | 10/10, retention=1.00 |
| Stage 5 | 1/6, scope=0.00, retention=0.17 | 6/6, scope=0.50, retention=1.00 |

Headline deltas: **Stage 2 scope 0.12 → 1.00**, **Stage 5 scope 0.0 → 0.50**, **Stage 5 retention 0.17 → 1.00**.

### Test suite

441 tests pass (up from 283 reported in the original section above).

### Remaining gap

The concept-scoring path is now functional — the residual shapes concept scores, the Hebbian learning trains the sensing matrix, `_extract_all_concepts` registers all label tokens, and the reranker now actually retrieves and applies those labels. Stage 2 scope reached 1.00 and Stage 5 scope improved from 0.0 to 0.50.

The remaining Stage 5 scope gap (0.50 vs 1.0) is the genuine residual architectural bottleneck called out in the "What would fix this" section above: cross-domain disambiguation between same-vocabulary different-sense requests still needs sense-specific concept vocabulary, semantic (embedding-similarity) concept matching, or a learned concept-to-label projection matrix. The wiring is no longer the blocker; the concept representation is.

The detailed fix report, including the per-bug diagnosis and commit references, is in `2026-06-30_scope_slot_reranker_fix.md`.

**Status (2026-06-30)**: Wiring complete and downstream reranker fixed, 7/7 preserved, 441 tests pass, Stage 2 scope 0.12→1.00, Stage 5 scope 0.0→0.50, Stage 5 retention 0.17→1.00.


## Update: bilinear policy head fix (2026-06-30)

The cortex dimension benchmark revealed that `d_cortex` had no effect on
curriculum performance because the warm_state was architecturally
disconnected from candidate discrimination. Four disconnections were
fixed (commit `76c6105`): scope-slot warm_state restoration before
policy scoring, L2-normalization of policy features, a bilinear
interaction term (`warm @ W_bilinear @ hidden_i`) replacing the
non-discriminating linear warm portion, and always-on warm_state
capture (previously gated behind `use_cortex_lm_answer`).

Unit tests confirm the bilinear term discriminates candidates and varies
with d_cortex. However, curriculum results are unchanged (Stage 5
scope=0.50) because the policy head is advisory — `policy_delta`
(softmax × weight=1.0) is dominated by the scope-rerank boost
(weight=2.0). The concept representation remains the bottleneck for
Stage 5 scope, not the policy head architecture.

Full details: `2026-06-30_cortex_dim_benchmark.md` (Update section).