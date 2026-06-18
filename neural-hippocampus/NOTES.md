# Neural Hippocampus – Prototype Notes

This folder contains the first minimal functional prototype of the Neural
Hippocampus described in `../experiments.txt` (sections 3 and 11). It is
intentionally tiny: it demonstrates the fast/slow memory loop in pure Python +
numpy without a real embedding model, vector database, or differentiable memory
module.

## What is implemented

- `SurpriseGatedMemory`
  - `write(episode)` computes novelty and surprise and only stores high-surprise
    traces.
  - `read_relevant(query, k=3)` returns the top-k episodes by cosine similarity
    over synthetic vectors.
  - `consolidate()` greedily clusters frequently replayed episodes into
    compressed slow-update summaries.
  - `decay_raw_traces(...)` prunes raw traces after their structure has been
    captured.

- `NeuralHippocampus`
  - `store(query, answer, correction, prediction_error)`
  - `reinforce(query)` -> replayed episodes
  - `consolidate()` -> slow-update summaries plus raw decay
  - `status()` -> episode count, slow-update count, approximate byte count

## Limitations

1. **Embeddings are synthetic.** We generate deterministic unit vectors from
   query strings using a hash. They preserve identity (the same query maps to
   the same vector) but have no real semantic geometry, so generalization to
   paraphrases is poor.
2. **Surprise is a heuristic.** It blends prediction error with novelty but does
   not learn a plasticity signal.
3. **Consolidation is shallow.** Clustering + averaging vectors is not the same
   as updating slow weights or adapters. There is no gradient descent here.
4. **No scope control.** The current system does not guard against
   overgeneralization, domain drift, or catastrophic forgetting.
5. **No external cortex.** `NeuralHippocampus.forward()` is intentionally
   unimplemented because a real cortex/adapter/core lives elsewhere in the stack.
6. **Byte count is approximate.** `status()` reports the pickled size of the
   in-memory trace dict; it is a proxy, not an exact on-disk footprint.

## Next steps

- Replace synthetic embeddings with a small fixed-size sentence encoder or a
  learned query embedding.
- Learn a real surprise/plasticity modulator instead of hand-combining novelty
  and prediction error.
- Replace greedy clustering with a differentiable `ExperienceAutoencoder` or
  adapter delta so consolidation actually changes slow weights.
- Wire hippocampus output to a fast-weight organ / plastic cortex so replay can
  influence inference.
- Add scope / anti-overgeneralization tests for users, domains, and tasks.
- Define the Correction-to-Competence benchmark task and measure
  `behavior_delta_per_byte_of_persistent_memory`.
