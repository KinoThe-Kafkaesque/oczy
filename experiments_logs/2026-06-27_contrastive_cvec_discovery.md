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

Current limitation: the contrastive cvec only reaches rank 5K, not rank 1.
The cvec steers in the right direction but not far enough. Possible fixes:
- Multiple contrastive pairs (SVD of several target-vs-default deltas)
- Different contrastive defaults to isolate the correct semantic sense
- Logit bias on top of cvec to push the target over the edge
- Layer-specific application (different contrastive vectors per layer)

## Next step

Implement contrastive cvec as a training mode in the consolidation_uptake
probe. Generate d_cortex=8 contrastive pairs using different default tokens,
SVD them to get the projector, and test cvec-only recall (no prefix).
