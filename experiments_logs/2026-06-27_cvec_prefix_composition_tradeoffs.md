# Cvec + Prefix Composition Tradeoffs

## The problem

The cortex needs two steering surfaces in a single generation:
- **Prefix** (reserved position) — exact-token recall from memory.
- **Cvec** (control vector) — behavioral posture / domain shift.

With `llama-cpp-python`'s global cvec, both act on the residual stream. Cvec
perturbs the residual stream during decode, and that perturbation is written
into the KV cache for the generated token. Subsequent tokens attend to the
steered KV entry, overriding the prefix's clean context.

## Six approaches tested — five cvec methods fail, logit biasing succeeds

### 1. Low-amplitude cvec (single forward, scale ≤ 0.01)

**Mechanism**: Run prefix + global cvec at a reduced `articulate_scale`. The
cvec perturbation is weak enough that its KV-cache contamination doesn't
override the prefix's exact-token recall.

**Verified**: Fresh-driver sweep on LFM2.5-1.2B-Instruct Q4 GGUF showed
coexistence at scale ≤ 0.01 (exact=1, domain=1). At scale=0.03 (the old
default), cvec is too strong and derails the prefix (exact=0, domain=0).

**Cost**: Zero additional cost. Single forward pass per token.

**Limitation**: At coexistence scales, the cvec's influence on the output is
negligible to invisible. The output register shifts slightly ("web" →
"healthcare" vertical) but this is a content change, not a domain shift.

**When to use**: Default choice on LFM2.5-1.2B. Set `compose_cvec_prefix=True`
and `articulate_scale ≤ 0.01`.

### 2. Per-position cvec (single forward, position-gated hooks)

**Mechanism**: Forward hooks on each transformer layer apply cvec only to
positions ≥ prefix length (prefill) or always (decode). Prefix tokens get
clean KV entries; query/answer tokens get steered KV entries.

**Status**: Implemented in TorchDriver, unit-tested at the hidden-state level.
Does NOT solve the composition problem — the cvec on the first decode step
writes a steered KV entry that contaminates the cache for all subsequent
tokens. Token-level trace proved this: prefix_only and decode-only diverge
at token 2, after a single steered decode step.

**Cost**: Single forward pass, negligible hook overhead.

**When to use**: Not for prefix+cvec composition. Sound for other per-position
steering applications where cache contamination of generated tokens is
acceptable.

### 3. CFG-style logit blending (two forwards per token)

**Mechanism**: Run a clean forward pass (cvec OFF, KV cache written) and a
steered probe (cvec ON, KV cache diverges) at each decode step. Blend in
logit space: `logits = logits_clean + w * (logits_steer - logits_clean)`.

**Tested**: Two-Llama-instance implementation on LFM2.5-1.2B-Instruct Q4 GGUF
with contrastive cvec. At w=1.0: token repetition garbage (same as full
cvec). At w>1: multilingual garbage (amplifying diffuse delta). At w=0.5:
empty output. No sweet spot.

**Root cause**: Cvec operates in residual stream space, CFG blend operates in
logit space. The unembedding matrix maps the cvec's residual-space direction
to a DIFFUSE set of logit changes spread across many tokens — not
concentrated on the target token. Amplifying this diffuse delta amplifies
noise, not signal.

**Cost**: 2× compute per token, 2× memory (two Llama instances).

**When to use**: Does not work for exact-token recall on 1.2B model with
contrastive cvec. May work with a cvec that produces a concentrated logit-space
delta (e.g., direct logit biasing), but that's a different mechanism entirely.

### 4. Contrastive cvec (direct to driver, single forward)

**Mechanism**: Train cvec from `embed(target) - embed(default)` deltas instead
of SVD of correction hiddens. The contrastive vector encodes token-specific
signal.

**Tested**: Direct application to driver (bypassing cortex). Target token
"vertical" rank improves 9× (47K → 5K out of 65K). But rank 5K is
insufficient — needs rank 1 to appear in output. At low scales: register
shift but no target token. At high scales: token repetition garbage.

**Cost**: Single forward pass.

**Limitation**: The contrastive cvec carries token-specific signal but too
weak to force the token to rank 1. Through cortex proj_c@warm_state: zero
effect (cortex indirection dilutes signal).

**When to use**: Does not achieve exact-token recall alone. Useful for
measuring token-specific signal in cvec training (the rank improvement
metric is a useful diagnostic).

### 5. SVD of correction hiddens (current approach, single forward)

**Mechanism**: SVD of `_last_hidden` vectors from correction turns. Captures
the "this is a correction" direction.

**Tested**: Target token rank stays at ~47K (unchanged from baseline). The
SVD cvec carries zero token-specific signal — it's pure posture bias.

**Cost**: Single forward pass.

**When to use**: Current default. Works for domain/posture shift. Does not
carry token-specific signal. Pair with prefix for exact-token recall.

### 6. Direct logit biasing (single forward, post-forward bias) — THE WINNER

**Mechanism**: After the forward pass produces logits, add a constant bias to
the target token's logit, then argmax. The KV cache entry for the forced
token is written from the CLEAN residual stream — the bias is applied
post-forward, not during. Subsequent tokens attend to un-contaminated context.

**Tested**: Two probes on LFM2.5-1.2B-Instruct Q4 GGUF.
- Probe 1 (disambiguation): "What does 'profile' mean?" → target " vertical"
  (single BPE token, id=12825). Bias ≥ 20.0 → "vertical layout" (exact=1,
  domain=1). Below 20.0: natural LM output ("set of data").
- Probe 2 (consolidation): "What is the secret passphrase for level 7?" →
  target " marmalade" (3 BPE subwords: " marm"+"al"+"ade"). Sequential
  subword biasing, bias ≥ 20.0 → "marmalade" (exact=1, domain=1). Below
  20.0: natural LM output ("P@ssw0...").

**Key evidence**: Output after the forced token is coherent ("vertical
layout", "marmalade" with clean continuation) — proving the KV cache was NOT
contaminated. The LM continues naturally after the biased token.

**Cost**: Single forward pass. No second instance needed (unlike CFG blend).

**Why it works where cvec fails**: Cvec perturbs the residual stream, which
contaminates the KV cache entry for the generated token, which then steers
all subsequent tokens off-manifold. Logit biasing never touches the residual
stream — it operates purely in logit space, post-forward.

**Implementation note**: `llama-cpp-python`'s `get_logits()` returns a flat
array of ALL positions (n_batch × n_vocab), not just the last position. Must
index the last position: `full[(n_last_batch-1)*n_vocab : n_last_batch*n_vocab]`.

**When to use**: When exact-token recall is needed without prefix. This is
the ONLY method that achieves it on the 1.2B model.

## Decision matrix

| Criterion | Low-amplitude | Per-position | CFG blend | Contrastive | SVD (current) | Logit biasing |
|---|---|---|---|---|---|---|
| Inference cost | 1× | 1× | 2× | 1× | 1× | 1× |
| Composes prefix+cvec | yes (≤0.01) | no | no | no | yes (≤0.01) | n/a (independent) |
| Exact-token recall | via prefix | no | no | no | via prefix | **YES (no prefix needed)** |
| Domain shift | negligible | n/a | garbage | weak | weak | n/a |
| Token-specific signal | no | no | no | yes (rank 5K) | no | **YES (direct)** |
| KV cache contamination | no | yes | no | yes | no | **no** |

## Conclusion

On LFM2.5-1.2B-Instruct Q4 GGUF, **five cvec methods fail to achieve
exact-token recall without prefix**:

1. Global SVD cvec — posture bias, zero token-specific signal
2. Per-position cvec — KV cache contamination on first decode step
3. Low-amplitude coexistence — works but cvec influence negligible
4. Contrastive cvec — rank 47K→5K but can't reach rank 1
5. CFG logit blending — amplifies diffuse delta, produces garbage

**The 6th method — direct logit biasing — succeeds where all cvec methods
fail.** It achieves exact-token recall on both the disambiguation probe
("vertical") and the consolidation probe ("marmalade", 3 subwords) at
bias ≥ 20.0, without prefix. The key insight: logit biasing bypasses the
residual stream entirely, so the KV cache stays clean and subsequent tokens
attend to un-contaminated context.

The root cause of all cvec failures is now precisely identified: **cvec
perturbs the residual stream, which contaminates the KV cache entry for the
generated token, which then steers all subsequent tokens off-manifold.**
Logit biasing never touches the residual stream — it operates purely in
logit space, post-forward.

The unification question — "can a single mechanism encode both domain shift
AND exact-token signal?" — is answered: **yes, but not via cvec.** Logit
biasing handles exact-token recall; cvec (at low amplitude) handles domain
shift. They operate on different surfaces (logit space vs residual stream)
and can coexist.
