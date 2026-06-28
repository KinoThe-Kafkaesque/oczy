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
