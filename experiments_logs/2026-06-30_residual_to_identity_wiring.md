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

## Conclusion

The 1M-param sensing matrix now flows into the identity hypernetwork's concept-scoring path — the wiring is complete. The residual shapes concept scores, and the Hebbian learning trains the sensing matrix. However, the concept scores don't influence the final answer because the concept vocabulary doesn't match the curriculum's label vocabulary. This is an architectural limitation in `_rank_answer`'s token-overlap concept boost, not in the residual-to-identity wiring itself.

**Status**: Wiring complete, 7/7 preserved, scope improvement blocked by concept vocabulary mismatch.
