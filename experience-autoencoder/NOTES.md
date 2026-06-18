# Notes / Next Steps

This is the first minimal Experience Autoencoder prototype.

## What works

- `ExperienceEncoder` converts an episode dict into a fixed-size δz vector
  (currently 32 dimensions, pure Python + NumPy).
- Encoding uses a hand-built tokenizer, a small shared vocabulary, and a
  deterministic random sensing matrix for the residual part of the latent.
- The first few latent dimensions encode the outcome (`accepted` / `corrected`
  / `failed` / `unknown`).
- `ExperienceDecoder` uses Orthogonal Matching Pursuit (OMP) to recover a
  sparse set of source/token weights from δz and then builds:
  - `failure_class`
  - `corrected_behavior_hint`
  - `trigger_conditions`
  - `counterexamples`
- `ExperienceAutoencoder` exposes `encode`, `decode`, `update_identity`,
  `reconstruction_error`, and `compress`.
- `compress` produces fixed-size NumPy vectors that are smaller than the
  JSON-serialized raw episodes in the smoke tests.

## Limitations

1. The encoder is not actually trained on data. It is a pseudo-learned
   random projection, so generalization to very different vocabulary or
   episode structures is limited.
2. The sensing matrix and OMP recovery can confuse tokens that land in
   nearby random directions. Decoded trigger tokens therefore sometimes
   include plausible-but-not-central words from the episode.
3. Source separation (situation vs. model_answer vs. correction vs.
   revised_answer) is only implicit in the sensing matrix columns. A
   stronger decoder would explicitly model per-source weights.
4. The vocabulary is bounded to 256 tokens and stopwords are stripped. Very
   long or rare-word episodes will lose information.
5. `corrected_behavior_hint` currently pairs top correction tokens with the
   single highest-confidence revised token. It does not produce nuanced
   lexical-semantic substitutions.
6. The reconstruction error is a heuristic Jaccard-based overlap score, not a
   proper likelihood or semantic distance.
7. There is no replay/consolidation loop yet. This prototype only demonstrates
   the `episode → Δz → fields` transform.

## Next steps

1. Add a tiny trainable NumPy-only linear autoencoder trained on a small
   corpus of synthetic episodes so the projection is learned, not random.
2. Separate the latent into orthogonal subspaces: outcome, source-attention,
   token identity, and context scope so decoding is more disentangled.
3. Implement an anti-overgeneralization penalty: when a correction is scoped to
   a context, the decoder should emit negative trigger conditions or
   counterexamples that exclude unrelated contexts.
4. Build a replay buffer of raw episodes keyed by Δz so old episodes can be
   used for offline consolidation without prompt-time retrieval.
5. Add a small evaluation harness that measures transfer: after seeing one
   corrected episode, does the decoder produce the same correction for a
   paraphrased situation?
6. Upgrade the decoder to a small neural network once the project graduates
   from the NumPy-only constraint.
