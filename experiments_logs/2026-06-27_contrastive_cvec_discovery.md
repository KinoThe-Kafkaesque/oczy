# Contrastive Cvec Discovery

## Date: 2026-06-27

## Finding

The current SVD cvec training (SVD of correction `_last_hidden` vectors) carries
**zero token-specific signal**. A contrastive vector (`embed(target) - embed(default)`)
jumps the target token's ranking **9×** in the LM's logit distribution.

## Experiment

Probe: `"'Profile' here means business _______."` with target `"vertical"`.
Measured the rank of the target token `" vertical"` (id=12825) in the LM's
logit distribution after the probe, under different cvec training conditions.

### Results

| Cvec training | "vertical" rank | logit | top tokens |
|---|---|---|---|
| No cvec | 47,251 | -4.65 | `**`, `1`, `Question` (formatting) |
| SVD of correction hiddens (current) | 46,322 | -4.57 | `**`, `1`, `Question` (unchanged) |
| Contrastive (vertical - profile) | 22,229 | +1.83 | `Position`, `Argument`, `position` |
| **Contrastive (vertical - business)** | **5,212** | **+2.94** | `Separ`, `Revolutionary`, `separation` |

### Key observations

1. **SVD cvec does nothing to target token ranking**: rank stays at ~47K out of
   65,536. The SVD of correction hiddens captures the "this is a correction"
   direction, which has no information about the specific target word.

2. **Contrastive cvec works**: `embed("vertical") - embed("business")` jumps
   the rank from 47K to 5K — a 9× improvement. The top tokens shift from
   formatting tokens to semantically related words.

3. **Scale is insensitive**: scales 1.0, 3.0, 10.0, 30.0 all produce nearly
   identical results. This suggests `llama_set_adapter_cvec` normalizes the
   vector, or the embedding-space vector doesn't perfectly align with the
   residual stream.

4. **Wrong semantic sense**: the top tokens ("Separation", "Revolutionary")
   suggest the contrastive vector amplifies the "upright/distinction" sense
   of "vertical", not the "business vertical" sense. The choice of
   contrastive default matters — `vertical - business` isolates a different
   sense than `vertical - horizontal` would.

5. **Probe format defeats cvec-only steering**: the model sees the
   fill-in-the-blank probe and generates question-formatting tokens (`**`,
   `1`, `Question`) instead of attempting to fill the blank. The prefix
   works because it provides answer context that overrides this
   interpretation. Cvec-only has no such override.

## Hypothesis: why SVD fails vs contrastive succeeds

The SVD of correction `_last_hidden` vectors captures the **correction signal**
direction — the model's representation of "this is a correction, update the
cortex." This is a meta-signal about the interaction type, not about the
content of the correction. The cvec shifts the output register (posture bias)
but carries zero information about which specific word the correction was
targeting.

A contrastive vector (`embed(target) - embed(default)`) directly encodes
**what makes the target token different from the default**. This is
token-specific signal. When applied as a cvec, it shifts the LM's residual
stream toward the target token's direction in embedding space, which
propagates through the unembedding matrix to boost the target token's logit.

## Implication for prefix+cvec unification

If the cvec can encode token-specific signal (via contrastive training), then
in principle a single cvec could achieve both domain shift AND exact-token
recall — unifying the two steering surfaces. The prefix would become
unnecessary.

## End-to-end probe results (runs #127-#128)

Two approaches were tested end-to-end:

### Through cortex proj_c @ warm_state (run #127)

Trained proj_c_shared from SVD of contrastive deltas, then articulated through
the normal cortex path (emit_all_cvecs → proj_c_shared @ warm_state). Result:
**zero effect** — pre and post-warm outputs identical. The 8D warm_state
updated by perceiving the correction doesn't produce a "vertical"-pointing cvec
when projected through the contrastive-trained projector. The cortex
indirection dilutes the token-specific signal.

### Direct to driver, bypassing cortex (run #128)

Applied the SVD principal component of contrastive deltas directly to the
driver via set_cvecs_per_layer, sweeping scales:

| scale | semantic | domain | answer |
|---|---|---|---|
| 0 (baseline) | 0 | 0 | "refers to a set of data or" |
| 0.03 | 0 | 0 | "refers to the analysis of" |
| 0.01 | 0 | 0 | "refers to a section that provides insights" |
| 1.0+ | 0 | 0 | "presents presents presents..." (garbage) |

The cvec shifts the output register at low scales ("set of data" → "analysis
of") but cannot force "vertical" to appear. At high scales: token repetition
garbage. No sweet spot.

## Conclusion

**Cvec is a posture bias even with contrastive training.** The rank
improvement from 47K to 5K is significant but insufficient — the target
needs to be rank 1 to appear in the output, and the cvec can't push it
that far. The unification path (cvec alone for both domain shift AND
exact-token recall) fails on this 1.2B model.

The prefix mechanism remains necessary for exact-token recall because it
provides direct KV-cache context that the LM attends to — a fundamentally
stronger signal than a residual-stream bias. The best composition remains
prefix (exact-token recall) + low-amplitude cvec (subtle register shift),
as documented in `2026-06-27_cvec_prefix_composition_tradeoffs.md`.

The contrastive cvec finding partially overturns the "posture bias, not
retrievable knowledge" verdict: cvec CAN carry token-specific signal (rank
47K→5K), but the signal is too weak to force the token to rank 1. It's a
weaker posture bias, not a knowledge retrieval mechanism.

## CFG logit blending result (run #133)

Implemented CFG-style logit blending with two Llama instances (clean +
steered) and contrastive cvec. At each decode step, blended logits as
`logits_clean + w * (logits_steer - logits_clean)`. Swept w from 0.5 to 10.0.

| w | semantic | domain | answer |
|---|---|---|---|
| 0.5 | 0 | 0 | "" (immediate stop) |
| 1.0 | 0 | 0 | "presents presents..." (repetition) |
| 2.0+ | 0 | 0 | multilingual garbage |
| 10.0 | 0 | 0 | more garbage |

**Root cause**: Cvec operates in residual stream space, CFG blend operates in
logit space. The unembedding matrix maps the cvec's residual-space direction
to a diffuse set of logit changes spread across many tokens — not concentrated
on the target token. Amplifying this diffuse delta amplifies noise, not signal.

This is the 5th cvec method tested for cvec-only exact-token recall. All
fail on the 1.2B model.

## Direct logit biasing result (runs #136-#137) — THE WINNER

A 6th, non-cvec mechanism was tested: direct logit biasing. After the forward
pass produces logits, add a constant bias to the target token's logit, then
argmax. The KV cache entry for the forced token is written from the CLEAN
residual stream — the bias is applied post-forward, not during.

| Probe | Target | Tokens | Threshold | Result |
|---|---|---|---|---|
| Disambiguation | " vertical" | [12825] (single) | bias ≥ 20.0 | "vertical layout" (exact=1, domain=1) |
| Consolidation | " marmalade" | [55678, 786, 1339] (3 subwords) | bias ≥ 20.0 | "marmalade" (exact=1, domain=1) |

Below threshold: natural LM output matching baseline. Above: forced target
token with coherent continuation — proving no KV cache contamination.

**This is the ONLY method that achieves exact-token recall without prefix on
the 1.2B model.** The key insight: logit biasing bypasses the residual stream
entirely, so the KV cache stays clean. All cvec methods fail because they
perturb the residual stream, which contaminates the KV cache entry for the
generated token.

The prefix mechanism is no longer the only path — logit biasing is a
fundamentally different, non-cvec approach that works.
