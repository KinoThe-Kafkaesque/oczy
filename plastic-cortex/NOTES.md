# Plastic Cortex — Prototype Notes

## What was implemented

A minimal end-to-end toy that demonstrates the architecture proposed in
`../experiments.txt` (sections 1, 2, and 11):

```text
input / correction
 
recurrent state (TokenRNN)
 
fast-weight scratchpad (FastWeightLayer)
 
scoring over slow priors → answer
```

### Components

1. **`plastic_cortex.state.TokenRNN`**
   - A tiny deterministic recurrent cell.
   - Each token is converted to a fixed-size input vector through a hash-based
     pseudo-embedding.
   - Hidden state is updated with an Elman-style step using randomly
     initialized, fixed weights.
   - Purpose: provide a persistent "session state" that changes because the
     cortex has seen tokens, not because it re-reads a chat log.

2. **`plastic_cortex.fast_weight.FastWeightLayer`**
   - A bounded associative scratchpad: one score vector per token over the
     current label set.
   - `update(token, correction=False, target=...)`:
     - normal text uses a small plasticity `alpha_normal`
     - explicit correction uses a much larger `alpha_correction`
   - Includes small lateral inhibition so strengthening one label weakens
     competing ones.

3. **`plastic_cortex.cortex.PlasticCortex`**
   - Combines slow priors (`baseline`), fast weights, and recurrent bias to
     score candidate labels.
   - Public API:
     - `answer(query) -> str`
     - `correct(correction_text, expected_answer) -> None`
     - `status() -> dict`
     - `reset_state()`
   - Toy domain: the word **"profile"** defaults to "user profile" and can be
     re-grounded to "business vertical" with a single correction.

## Why this mechanism was chosen

The brief asked for the *first* minimal prototype, not a production model.  Pure
Python keeps it dependency-free, inspectable, and fast to reason about.

The update rule is a Hebbian/associative version of the fast-weight idea:
normal tokens produce weak traces, corrections produce strong traces.  This
maps directly onto the "normal text → weak write; explicit correction → strong
write" design from `experiments.txt` section 2, and is the simplest possible
演示 that the higher-level concept ( corrections have a larger write gate)
can actually change future behavior.

## Limitations

- **No real language understanding.**  It is a word-association toy; it
  will fail on paraphrase, negation, or long-range context.
- **Hand-coded priors.**  The default "profile" → "user profile" bias is
  hard-wired, not learned.
- **No generalization control.**  A correction can overgeneralize because the
  fast weights are not scoped by domain or context.
- **No consolidation to slow weights.**  Fast weights are ephemeral; there is
  no replay loop that distills repeated corrections into permanent changes.
- **Deterministic, not trained.**  The recurrent cell is fixed; it demonstrates
  state dynamics but does not learn them.
- **No surprise/error signal.**  The correction gate is binary; the cortex
  does not estimate its own uncertainty or prediction error.

## Next steps

1. **Trained fast-weight organ.**  Replace the toy associative matrix with a
   small learnable TTT-like layer or a differentiable plasticity module.
2. **Scope control.**  Add a context/key to fast weights so a correction about
   "profile" in one domain does not leak into unrelated conversations.
3. **Surprise gating.**  Use a world-model critic to compute prediction error
   and modulate the write gate continuously, not just binary.
4. **Neural hippocampus.**  Add a bounded memory buffer of high-surprise
   episodes and a replay consolidator that can promote fast changes to slow
   priors.
5. **Benchmark.**  Build the Correction-to-Competence task described in
   `experiments.txt` section 14 and measure uptake latency and transfer.
