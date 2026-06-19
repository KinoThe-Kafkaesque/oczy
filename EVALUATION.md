# Extended Learning Evaluation

Ran `eval_extended.py` on 30 word-sense ambiguity corrections plus 12 trivia facts as controls.

## Aggregate ranking

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

## What each score means

- **NeuralHippocampus** (1.000): perfect surprise-gating, 100% replay accuracy before consolidation,
  strong trace compression, and every consolidated summary became a slow update.
  This is the only module that directly demonstrates the architecture's central
  thesis: raw traces can be compressed and forgotten while structured summaries remain.

- **PlasticCortex** (0.967): nearly perfect correction uptake. With `alpha_correction=8.0`
  a single correction overrides the slow prior, while normal text exposure does not.
  The 3% miss is on words that overlap multiple curriculum senses.

- **SkillImmuneCortex** (0.590): excellent detector precision/recall (0.98 F1) but merging does
  not reduce detector count in this sparse curriculum, so the compression reward is zero.

- **WorldModelCritic** (0.399): learns that corrected queries are risky, but the hard-coded
  ambiguity list and Jaccard similarity fail to separate similar from unrelated queries.
  Discrimination is weak.

- **ExperienceAutoencoder** (0.203): the Δz vector is tiny (256 bytes total identity), but
  reconstruction accuracy is low and the encoder is an untrained random projection.
  The idea is validated; the implementation is not.

- **IdentityHypernetwork** (0.001): the fixed `CONCEPT_VOCULARY` does not include the 30
  curriculum senses, so adapter retrieval fails completely. The latent shifts slightly,
  but it cannot express the curriculum.

## Takeaways

1. **Neural Hippocampus is the most promising organ** so far. It is the only module that
   really exercises the "forget raw memory after consolidation" part of the thesis.

2. **Plastic Cortex** proves the fast-weight gate mechanism works in isolation, but it
   is still a toy word-association model with no semantic generalization.

3. **Identity Hypernetwork** needs the most rework: replace the fixed concept list with
   an open embedding layer before it can learn anything real.

4. **Experience Autoencoder** should focus next on a trained encoder rather than a random
   projection; the latent size is right but the signal is not preserved.

5. **World-Model Critic** needs learned similarity and a structured outcome space rather
   than a single scalar correction probability.

## How to reproduce

```bash
uv run python eval_extended.py
```

All module tests still pass:

```bash
uv run pytest -q   # 39 passed
```
