# Decoder parity DEV v1 — September 13, 2026

**COMPLETE: the new versioned batch decoder qualifies on all 64 comparisons
with scalar decoding. Fixed continuation positions explain the observed batch
mismatches. This repairs execution consistency, not learned capability.**

[Protocol](../experiments/decoder-parity-dev-v1/README.md),
[complete outputs and independent audit](2026-09-13_decoder_parity_dev_v1.json),
[candidate implementation](../src/oczy/experiments/meta_cortex/generation_v2.py).
Manifest `c9e35d5ca8b9eee57fc8a6740afc9b3bcc27cecee6fcc25198c4448a322b15ef`.
The original organ remains hash-identical for earlier and ongoing experiments.

## Paired result

Each cell counts exact decoded-string agreement with the unchanged scalar
raw-argmax implementation, out of 16. No-bank cases use oracle-selected
instructions. Learned-bank cases use the saved DEV-v2 seed-0 joint bank with
no rule or example text. The same four clients and four words occur in both.

| Decoder | No bank | Learned bank |
|---|---:|---:|
| Original batch, inherited repetition penalty 1.1 | 9/16 | 12/16 |
| Original batch, neutral penalty 1.0 | 9/16 | 12/16 |
| Advancing positions, penalty 1.1 | **16/16** | **16/16** |
| Advancing positions, neutral penalty 1.0 | **16/16** | **16/16** |
| New neutral decoder, batches of four variable-length prompts | **16/16** | **16/16** |
| Native input-ID API, neutral settings | **16/16** | Not applicable |

The predeclared candidate criterion is met: all 32 single-item and all 32
four-item-batch responses equal scalar output. This is a bounded equality check
on known DEV cases, not token/logit equality, arbitrary batch equivalence or a
powered model-capability acceptance result.

## Mechanism and limits

The old batch function passes a prompt-length `position_ids` tensor to
`model.generate`. In the pinned Transformers 5.0.0 implementation,
`prepare_inputs_for_generation` slices that tensor for continuation;
`_update_model_kwargs_for_generation` extends the attention mask and cache
position, but does not extend the supplied tensor. Generated tokens therefore
reuse the final supplied position instead of receiving advancing positions.

The new helper omits explicit position IDs so they are reconstructed from the
expanding attention mask on each step. The two-by-two ablation isolates this
change: it restores all observed batch agreements at either repetition penalty.
Neutralizing repetition penalty alone restores none of the mismatching cases
in this battery. The helper additionally supplies an explicit neutral
`GenerationConfig`, matching scalar argmax and preventing inherited settings.

The underlying model artifact sets repetition_penalty=1.1, while the runtime
manifest declares 1.0. The old native/batch invocation does not enforce that
declared setting. In DEV v3, the native input-ID path can therefore disagree
with scalar even where positions are correct. Neutral native generation
matches scalar on all 16 registered cases here. The candidate makes effective
settings explicit rather than relying on a declarative manifest alone.

Matching a wrong answer is still parity. Scalar and the candidate score only
9/16 on the no-bank task targets and 8/16 with this learned bank. Original
batch accuracy is 6/16 and 7/16. Decoder repair does not establish correct
context selection or preservation, and cannot explain the ongoing learning
experiment's errors because that experiment uses scalar decoding throughout.

`calibration_runner.py` uses the original batch API, so future batched R20
calibration needs a separately versioned adoption and qualification. Earlier
campaign scores are preserved, not retrospectively altered or declared to have
an unmeasured error magnitude. The new helper is exercised by this diagnostic;
it has not been installed into the historical organ or old autonomous loop.

## Verification

All 176 task scores and 208 equality verdicts are independently recomputed.
Every registered prompt and numeric-state hash, source hash, runtime identity,
and unchanged pre/post model hash passes. Runtime source and model generation
configuration are retained in the archive for the mechanism audit.
One offline two-thread local CPU process takes 293.54 seconds. No optimizer
updates, meta-test, remote launch or historical instrument edits occur.
The associated curriculum/interface/guard/generation regression suite passes
73 checks. Live model qualification supplies the decisive decoder evidence.
