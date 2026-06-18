# World-Model Critic v1 — Notes

## What is implemented

- `WorldModelCritic` is a lightweight, NumPy-free online predictor.
- It estimates, for a `(query, proposed_answer)` pair:
  - `accepted_prob` — probability the answer is accepted without change.
  - `correction_likelihood` — probability the user issues a correction.
  - `key_uncertainty` — Bernoulli variance, peaking when the prediction is near 0.5.
- Learning is online logistic regression over hand-built features:
  1. Normalized count of ambiguous tokens.
  2. Length ratio between answer and query (capped at 3.0).
  3. Empirical correction rate for similar prior queries (Jaccard overlap >= threshold).
- Each call to `record_outcome` stores the episode, recomputes the similarity-based
  feature with the new episode included, and takes one gradient step on the BCE loss.
- `prediction_error` measures |predicted correction probability - actual outcome|.

## Limitations

- **Hard-coded ambiguity list**: the model cannot discover new uncertainty cues
  from data; it only reweights the three pre-defined features.
- **Bag-of-words similarity**: Jaccard on normalized token sets is brittle.  It
  misses paraphrases, negation, and domain-specific terminology.
- **No true world model**: this is a single scalar outcome predictor, not a
  structured model of user intent, project state, or tool dynamics.
- **Unbounded memory**: every recorded outcome is retained; cost grows linearly
  with the number of interactions.
- **No consolidation or forgetting**: old raw traces are never summarized,
  compressed, or decayed, so catastrophic accumulation is possible.
- **Single-headed uncertainty**: `key_uncertainty` is derived only from the
  predicted probability.  It does not distinguish epistemic uncertainty from
  aleatoric uncertainty.
- **No scope control**: a correction for "profile" in one domain can bleed into
  other contexts if queries share tokens.
- **No validation against the proposed answer content**: the critic only looks at
  query/answer surface statistics, not whether the answer is actually correct.

## Next steps

1. **Learned similarity**: replace Jaccard with a small trainable embedding or
   autoencoder of query/answer pairs so the critic can recognize semantic
   similarity rather than literal token overlap.
2. **Fast weights / adapters**: add a tiny trainable adapter that is updated on
   every correction and modulates predictions, moving beyond three fixed
   features.
3. **Bounded memory & replay**: implement a surprise-gated buffer; replay
   high-error episodes occasionally and consolidate old traces into slow-moving
   priors.
4. **Structured outcomes**: distinguish correction types (semantic,
   formatting, factual, safety) and predict the *kind* of correction, not just
   its probability.
5. **Uncertainty ensembles**: maintain several Online Learners and report
   disagreement as epistemic uncertainty.
6. **Evaluation**: build the "Correction-to-Competence" benchmark suggested in
   `experiments.txt` (section 14) and measure uptake latency, transfer, scope
   control, and memory compression.

This prototype is intentionally the "dumbest thing that could work" for the
world-model-first idea: before replacing it, make sure it fails in the ways
listed above.
