# Experiment Log: Extended Learning Component Evaluation

**Date:** 2026-06-19
**Experiment:** Train all seven Oczy modules on a shared 30-item word-sense/trivia curriculum and evaluate each against task-specific metrics.
**Evaluator:** NCode
**Command:** `uv run python eval_extended.py`

---

## Hypothesis / Goal

The Plastic World Model Agent architecture proposes that experience should flow through a metabolism:

```text
experience -> fast change -> replay -> compression -> slow change -> forgetting raw trace
```

Most projects never test this end-to-end. This experiment asks: after an
extended learning curriculum, which of the seven organs actually learns
something, stays bounded in memory, and preserves the learned behavior?

## Method

1. **Curriculum:** 30 word-sense ambiguity corrections (e.g. "profile" -> business
   vertical, "branch" -> git branch) plus 12 unrelated trivia facts as controls.
2. **Training:** Each module is exposed to the curriculum through a thin wrapper
   that maps `answer()` / `correct()` calls to the module's native API.
3. **Metrics per module:**
   - **PlasticCortex**: correction accuracy vs. normal-text-only control; plasticity ratio.
   - **NeuralHippocampus**: surprise-gating precision, replay accuracy before consolidation,
     compression ratio, slow-update rate.
   - **WorldModelCritic**: correction likelihood on repeated/similar queries vs. unrelated queries;
     discrimination and calibration.
   - **IdentityHypernetwork**: adapter retrieval accuracy, identity shift magnitude.
   - **SkillImmuneCortex**: detector precision, recall, F1, merge ratio.
   - **ExperienceAutoencoder**: compression ratio, reconstruction error, final identity size.
4. **Baselines:** Oracle agent (upper bound) and Always-Wrong agent (lower bound).

## Results

| Rank | Module | Aggregate Score |
|---|---|---|
| 1 | NeuralHippocampus | 1.000 |
| 2 | PlasticCortex | 0.967 |
| 3 | Oracle (upper bound) | 0.844 |
| 4 | SkillImmuneCortex | 0.590 |
| 5 | WorldModelCritic | 0.399 |
| 6 | ExperienceAutoencoder | 0.203 |
| 7 | IdentityHypernetwork | 0.001 |
| 8 | Always-Wrong (lower bound) | 0.000 |

### Per-module details

- **NeuralHippocampus**: perfect surprise gating (high-surprise episodes stored,
  low-surprise trivia rejected), 100% raw-trace replay accuracy, strong trace
  compression, and every consolidated cluster became a slow update. This is the
  only module that exercises the full fast-write / replay / consolidate /
  decay loop.

- **PlasticCortex**: 96.7% correction accuracy with a normal-text exposure control
  near zero. The fast-weight correction gate clearly overrides slow priors.
  Still a word-association toy with no semantic generalization.

- **SkillImmuneCortex**: 0.98 F1 on trigger detection. Detectors fire on relevant
  queries and ignore unrelated trivia. Merging did not reduce detector count in
  this sparse curriculum, so compression reward was zero.

- **WorldModelCritic**: learns that corrected queries are risky, but the
  hard-coded ambiguity list and Jaccard sentence similarity fail to separate
  similar from unrelated queries. Discrimination is weak.

- **ExperienceAutoencoder**: total encoded identity is tiny (256 bytes), but the
  encoder is an untrained random projection. Reconstruction error is high.

- **IdentityHypernetwork**: fixed `CONCEPT_VOCABULARY` does not include the
  curriculum senses, so adapter retrieval fails completely. The latent shifts,
  but it cannot express the learned distinctions.

## Conclusion

1. **NeuralHippocampus is the most promising organ.** It is the only module that
   validates the architecture's central claim: raw memory can be compressed and
   forgotten while the learned structure survives.

2. **PlasticCortex proves the gate mechanism works**, but the toy model needs
   real embeddings or adapters for transfer and scope control.

3. **IdentityHypernetwork needs the most rework.** Replace the hard-coded concept
   list with an open vocabulary/tokens before it can learn.

4. **ExperienceAutoencoder** should prioritize a trained encoder; the latent size
   is right but the signal is not preserved.

5. **WorldModelCritic** needs learned sentence similarity and structured outcome
   prediction, not a single scalar correction probability.

## Next steps

- Wire NeuralHippocampus slow updates into PlasticCortex's inference path.
- Add a real embedding layer to IdentityHypernetwork and retest.
- Train the ExperienceAutoencoder encoder on synthetic episodes.
- Replace the WorldModelCritic's hard-coded features with a small learned
  similarity model.
- Re-run this evaluation after each organ upgrade.

## Artifacts

- `eval_extended.py` — full evaluation script
- `EVALUATION.md` — summarized report
- `correction-benchmark/` — canonical benchmark harness

