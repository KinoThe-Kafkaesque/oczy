# Experiment Log: Cortex-as-Centre Un-Inversion

**Date:** 2026-06-23
**Experiment:** Reframe the architecture so the cortex is the live mutation
surface and the LM is a perception/articulation organ on its edges. Replace
the toy label-string cortex with a tensorial `KVCortex` whose warm/cold
state is steerable into the LM's attention KV cache.
**Evaluator:** NCode

---

## Hypothesis / Goal

The thesis (`experiments.txt` section 12) calls for an agent where memory
becomes **changed dynamics**: activation → fast state → fast weights →
neural memory → slow weights → forgetting raw trace. Today's `OrganismAgent`
and `LMBackendAgent` are both inverted relative to this:

- `OrganismAgent`: cortex writes label strings ("user profile"); the LM is
  absent. The slow/fast weight distinction is decorative because
  `PlasticCortex.BASELINE` never mutates.
- `LMBackendAgent`: a frozen LLM drives the answer path while organ
  metabolism runs in parallel as bookkeeping. Experience never reaches the
  LM's forward dynamics.

This experiment sets up the cortex contract that un-inverts the
architecture: the cortex is a small-dim (`d_cortex = 128`) two-speed
neuromodulator that reads the LM's hidden state on the perception side and
projects its intent into the LM's attention KV cache on the articulation
side. The LM stays frozen; every mutation lives in the cortex.

## Method

1. **Author** `KVCortex` (`plastic-cortex/src/plastic_cortex/kv_cortex.py`) with
   the contract:
   - `warm_state: ndarray[d_cortex]` — session-mutable, dies on restart.
   - `cold_state: ndarray[d_cortex]` — boot-loaded, written only by
     `consolidate()`.
   - `observe(lm_hidden, correction_signal)` — warm update, sub-ms.
   - `project_intent(layer_idx)` — per-layer `(k, v)` cached delta
     for KV-slot injection.
   - `train_step(lm_hidden)` — Hebbian update on `proj_hidden`.
   - `consolidate(replays=...)` — slow EMA nudge + replay absorption
     into cold state.
   - `reset_warm_from_cold()` — cold boot semantics.
2. **Author** smoke tests covering shape contract, warm/cold separation,
   consolidation behaviour, neuromodulator effect, projector stability,
   pickle round-trip, status contract.
3. **Validate** the contract on its own, before binding to the LM. The LM
   binding is tracked as Goal 1 in `GOALS.md` and is next-session work.

## Results

### Cortex contract

```
File: plastic-cortex/src/plastic_cortex/kv_cortex.py
  d_cortex       = 128 (default)
  n_layers       = 28 (matches LFM2.5-1.2B)
  d_head         = 256
  per-turn cost  = sub-2 ms (one d_embd matmul + cached per-layer projections)
  cold persist   = pickle of (proj_hidden, proj_k, proj_v, cold_state)
```

### Smoke tests

```
uv run python plastic-cortex/tests/test_kv_cortex.py

ok: shapes
ok: warm_mutates_cold_does_not
ok: consolidate_moves_warm_into_cold
ok: consolidate_replay_absorption
ok: correction_signal_raises_plasticity
ok: reset_warm_from_cold
ok: hebbian_training_changes_projector
ok: pickle_round_trip
ok: status_contract

9/9 passed
```

The tests verify:

1. `observe()` mutates only warm_state, never cold_state.
2. `consolidate()` mutates cold_state (always via slow EMA nudge; replay
   absorption is gated by rehearse count threshold).
3. `correction_signal=1.0` produces larger warm drift than `0.0` — the
   neuromodulator is functional, not just a parameter.
4. Hebbian `train_step()` mutates `proj_hidden` while per-row L2
   renormalisation keeps norms in `[0.9, 1.1]` after 50 updates.
5. Pickle round-trips all state; RNG state is intentionally not serialised
   so loaded cortex is deterministic given its seed.
6. `status()` reports the canonical cross-organ fields (`serialized_bytes`,
   `record_count`, plus cortex-specific norms).

### What the contract does NOT yet do

The cortex has no LM binding. It can absorb synthetic hidden vectors and
emit `(k, v)` tuples, but no driver writes those into the LM's KV cache
yet. The loop closes mathematically but not behaviourally. This is the
work tracked in `GOALS.md` Goal 1.

## Conclusion / Next steps

The cortex contract is shape-correct, two-speed (warm/cold), and
neuromodulated (correction_signal raises plasticity). It is the first
organ in the repo whose mathematical shape genuinely matches the thesis:
small learnable state, fast in-session mutation, slow boot-persistent
identity, no text in its input or output.

**Next session:**
- Goal 1 — LM-side KV-write binding via `llama-cpp-python`'s `kv_self`
  surface.
- Once the binding is understood, write `CortexAgent` as the un-inverted
  organism: cortex is the mutation surface; organs consume tensor
  signals from cortex state; LM is perception + articulation only.
- Goal 2 (hidden-state extraction at layer L) can begin in parallel once
  Goal 1's binding surface is mapped.

## Artifacts

- `plastic-cortex/src/plastic_cortex/kv_cortex.py` — cortex contract
- `plastic-cortex/tests/test_kv_cortex.py` — 9 contract tests
- `GOALS.md` — architecture goals 1/2/3 with done-when criteria
- This log file
---

## Update — Goal 1 binding verified end-to-end

After authoring the cortex contract, Goal 1 (LM-side KV-write binding via
`llama-cpp-python`) was probed and prototyped. Findings:

### Surface probe

`llama-cpp-python` 0.3.31 exposes the cvec surface via
`llama_set_adapter_cvec(ctx_p, data_ptr, len, n_embd, il_start, il_end)`.

- `ctx_p` is reachable as `Llama._ctx.ctx` (raw int pointer).
- `data` is a flat `c_float *` pointer; for per-layer cvecs, length must be
  `n_embd * (il_end - il_start)` and the API distributes one slice per layer.
- Clearing is `llama_set_adapter_cvec(ctx_p, None, 0, n_embd, 0, n_layers)`.
- The cvec persists across `create_completion` calls (apply once, holds until
  cleared).
- The cvec REPLACES on each call — looped per-layer `set_cvec_layer(L)` calls
  keep only the last layer's steering.
- The LFM2.5-1.2B-Instruct GGUF reports `n_layers=16` via `llama_n_layer()`
  (the API counts attention-capable layers; RWKV/conv hybrid layers are
  not addressed).

### Driver shim

`oczy_lm/cvec_driver.py` formalises the surface as `LlamaCVecDriver`:

- `set_cvec_layer(L, vec)` — single-layer apply (replaces prior). Useful for
  targeted steering; not what cortex metabolism needs.
- `set_cvecs_per_layer(vectors, scale=1.0)` — batched single-call apply.
  Takes `n_layers` vectors of dim `n_embd`, concatenates, and invokes the
  adapter once with `il_start=0, il_end=n_layers`.
- `set_cvec_uniform(vec)` — same vector across every layer.
- `clear_cvec()` — NULL data over full range.
- `generate(prompt)` — wrapper around `Llama.create_completion`.
- `peek_embedding(prompt, last_token_only=True)` — final-layer prompt
  embedding via `Llama.create_embedding`. This is Goal 2 staging; layer-L
  intermediate extraction is not yet supported by the binding.

### Cortex contract revision

`KVCortex` was updated to match the real surface:

- `project_intent(layer_idx) -> (k, v)` (KV-slot tuples, dim `d_head`)
  was speculative. Replaced with `emit_cvec(layer_idx) -> ndarray[n_embd]`
  and `emit_all_cvecs() -> list[ndarray]`.
- `proj_k`/`proj_v` (shape `(n_layers, d_head, d_cortex)`) merged into a
  single `proj_c` (shape `(n_layers, d_embd, d_cortex)`).
- `KVCortexConfig.d_head` removed; `d_embd` is now both the input and
  per-layer output dim.
- Cortex tests rewritten (`plastic-cortex/tests/test_kv_cortex.py`):
  10/10 pass on the new contract including a cache-stability test that
  verifies `emit_cvec` returns the same memory until `observe()` is called.

### End-to-end test

`oczy_lm/tests/test_cvec_driver.py` proves the binding:

1. Baseline (no cvec): `"Hello, my name is"` → `" Alex, and I"`.
2. Cortex absorbs a synthetic hidden; emits per-layer cvecs at scale 30.
3. `driver.set_cvecs_per_layer(cortex.emit_all_cvecs(), scale=30)` shifts the
   LM output to something other than the baseline.
4. `driver.clear_cvec()` restores the baseline output exactly.

```
6/6 passed:
  ok: test_driver_reports_expected_shape
  ok: test_set_cvec_layer_shape_match
  ok: test_set_cvec_layer_rejects_wrong_dim
  ok: test_set_cvec_layer_rejects_bad_layer_idx
  ok: test_cvec_from_cortex_shifts_generation
  ok: test_peek_embedding_returns_n_embd_vector
```

### Regression

All pre-existing suites still pass:
- `plastic-cortex/tests/test_kv_cortex.py`: 10/10
- `experiments/organism_curriculum/validation.py`: 6 stages / 44 episodes, 0 errors
- `experiments/run_experiment.py --agent OrganismAgent`: same metrics as
  2026-06-22 (uptake 0.67, transfer 0.25, scope 0.17, forget 1.00).

### What is NOT done

- Goal 2 (layer-L hidden extraction): `peek_embedding` returns the
  final-layer output, which is the model's encoder-style summary. Layer-L
  intermediate residuals (the depth where semantic intent actually forms)
  are not exposed by the binding. Cortex metabolism currently works on
  the final-layer representation; that may be enough but is unproven.
- Goal 3 (organ upgrades to tensor inputs): the cortex binding is wired
  to the LM but the five metabolism organs still consume strings and
  dicts. The CortexAgent driver glue that string everything together
  hasn't been written yet.
- The cvec amplitude needed to dislodge greedy decoding is ~30x the
  cortex's default-init output. Once cortex weights train via Hebbian
  update on real hidden vectors, this scaling should become data-driven.

### Status

Goal 1: **complete**. Cortex→LM articulation coupling is real and testable.
Goal 2: deferred (final-layer embedding usable in staging).
Goal 3: next.

---

## Update — CortexAgent end-to-end (Goal 3 partial)

Goal 3's CortexAgent driver glue is built. The cortex is now the live
mutation surface of an un-inverted organism: it observes LM hiddens,
mutates warm_state, and its per-layer cvecs are applied to the frozen
LM via the `cvec` adapter on articulate(). The existing organ metabolism
is wired on the side, consuming cortex state vectors as surprise signals
where they used to consume string features.

### New file: `experiments/cortex_agent.py`

The CortexAgent class wires:
- `LlamaCVecDriver` (frozen LM + cvec adapter surface)
- `KVCortex` (live mutation surface)
- The five metabolism organs (existing implementations, routed off
  cortex state)

Lifecycle:

```
boot()                  cold_state loaded -> warm_state := cold_state.copy()
perceive(utterance)     LM final-layer embedding -> cortex.observe(...)
metabolize(utterance)   cortex.drift -> critic / hippocampus / immune /
                        autoencoder (autoencoder.train_step on perceptron
                        projector via the existing episode-shaped API)
articulate(prompt)      cortex.emit_all_cvecs() -> driver.set_cvecs_per_layer
                        (scale=30.0 default) -> driver.generate()
                        (clear_cvec on success)
consolidate()           hippocampus.consolidate + cortex.consolidate(replays)
                        (replays synthesised by re-embedding consolidated
                        queries via the LM -- placeholder until the
                        hippocampus natively stores tensor replays)
save(path)/load(path)   Pickle: cortex cold_state + proj_hidden + proj_c
                        + each organ. Warm state intentionally NOT
                        persisted (session-only by design).
turn(utterance)         Convenience: perceive + metabolize + articulate
                        in one call.
```

Correction-signal detection is lexical today (`_looks_like_correction`):
the cortex's neuromodulator fires `alpha_correction` plasticity when the
user's utterance contains markers like "no,", "wrong,", "actually,", etc.
Goal 3 follow-up replaces this with a learned detector based on the
WorldModelCritic's drift-vs-string-feature disagreement.

### Test: `experiments/tests/test_cortex_agent.py`

Six smoke tests, all passing against the real LFM2.5-1.2B-Instruct Q4_K_M:

```
ok: test_perceive_produces_warm_state
ok: test_correction_signal_drives_plasticity
ok: test_metabolize_routes_to_hippocampus_on_drift
ok: test_articulate_steered_differs_from_baseline
ok: test_consolidate_moves_cold_state
ok: test_save_load_round_trip_preserves_cold

6/6 passed
```

Verifies:

1. `perceive` produces a d_cortex warm vector; caches the (n_embd,)
   hidden vector for the next metabolize call.
2. A correction-marker utterance mutates the cortex more than the
   baseline drift threshold (0.05).
3. `metabolize` walks the cortex-drift through the hippocampus's
   surprise gate, so episodes land in the replay bank.
4. `articulate` with cortex steering produces different LM output
   than the same prompt without steering.
5. `consolidate` mutates `cold_state` -- the boot-persistent
   identity survives a real learning sequence.
6. `save` -> `load` round trip preserves `cold_state` to within
   rtol/atol 1e-6; loaded cortex's warm equals cold (boot
   semantics).

### Regression

All pre-existing suites pass unchanged:

- `plastic-cortex/tests/test_kv_cortex.py` 10/10
- `experiments/organism_curriculum/validation.py` 6 stages / 44 episodes OK
- `experiments/run_experiment.py --agent OrganismAgent` baseline
  unchanged (this suite does not exercise CortexAgent -- it still
  uses OrganismAgent against the toy PlasticCortex)

### What is NOT done

- **Goal 2 (layer-L hidden extraction)** remains staging-only: the
  cortex consumes `peek_embedding`'s final-layer mean-pooled embedding,
  not an intermediate residual at semantic depth. Whether this
  representation is rich enough for real attractor-basin carving is
  unknown -- the cvec steering test confirms behaviour SHIFT, not
  whether the SHIFT maps to the user's intent.
- **Organ tensor-input upgrade**: CortexAgent feeds the existing organs
  with cortex.drift as the surprise scalar and synthesises replay
  tensors from re-embedded hippocampus queries. The hippocampus itself
  still stores string-keyed episodes; the critic still has its feature
  vector overwritten by the cortex.drift hack rather than rebuilt to
  consume cortex state natively.
- **Correction-signal detection** is lexical; the WorldModelCritic's
  drift-vs-text disagreement should drive it instead.
- **CortexAgent curriculum bridge**: the existing 6-stage organism
  curriculum still exercises `OrganismAgent`, not `CortexAgent`. A
  CortexAgent-compatible curriculum run is a follow-up -- once the
  cortex stabilises on meaningful absorption (synthetic hiddens do that
  only weakly), the curriculum can be used to MEASURE whether the
  corrected sense lingers in cold_state across consolidation cycles.

### Status

Goal 3 CortexAgent glue: **complete** (articulate loop closes; identity
survives save/load; organs routed from cortex state).
Goal 3 organ tensor-input upgrades: **partial** (drift-as-scalar wired;
deep organ rewrite deferred until CortexAgent behaviour validates
that further organ investment is justified).

### Recommended next move

Run CortexAgent through a small, scripted conversation:'''

sed -i 's/^### Recommended next move.*$//' /home/nyanpasu/Desktop/code/kinoSoft/oczy/experiments_logs/2026-06-23_cortex_kv_contract.md
cat >> /home/nyanpasu/Desktop/code/kinoSoft/oczy/experiments_logs/2026-06-23_cortex_kv_contract.md << 'EOF'

### Recommended next move

Author a `cortex_conversation_demo.py` script that runs CortexAgent
through a 4-turn scripted conversation (query -> correction -> query ->
probe). Verify:

1. The cortex's `cold_state` visibly moves after `consolidate()`.
2. The LM's articulated output differs between baseline and post-correction
   on the same prompt.
3. The cold_state persists across save/load and still biases generation.

That demo is the qualitative behaviour check that tells us whether the
cortex is actually absorbing intent or just producing noise-shape
variation in the LM's cvec inputs. If the latter, Goal 2 (intermediate-
layer hidden) is the next priority; if the former, the organ upgrades
become the bottleneck to closing the full metabolism loop.

---

## Update — CortexAgent qualitative demo (cvec steering diagnosis)

`experiments/cortex_conversation_demo.py` was authored and run. The demo
proves the cortex state persists across consolidate + save/load, but at
the SAME scale that demonstrably shifts greedy decoding (30), the
post-correction articulate() output is CJK-token garbage
(`'选选选选选...'`). Steering at all tested scales:

```
baseline            : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 0.5           : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 1.0           : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 2.0           : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 4.0           : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 8.0           : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 16.0          : "  \nOptions:  \nA) A summary of the product's features  \nB"
scale 30.0          : '选选选选选选选选选选选选选选选选选选选选'   <-- crash
```

Cortex state after perceive: warm_norm=6.66, projected cvec norm=37.6.
Without scaling that's already large (residual stream at LFM2 is ~5-20
in norm), yet greedy decoding absorbs the perturbation silently.

### Diagnosis: cortex is producing noise-shape steering, not intent-shape

What the demo proves:
- Cvec persistence works (cold_state survives save/load).
- Cvec surface works at scale 30 (output CHANGES from baseline).

What the demo does NOT prove:
- The cortex is steering toward the correction's meaning. The post-
  correction output is Chinese token repetition — i.e. the random
  default proj_c at large amplitude pushes the residual off-manifold
  into degenerate decoding rather than toward "business vertical".

### Why this is the right diagnostic

The cortex.absorbed_hidden -> warm_state is well-bounded via tanh and
EMA. But proj_c is a 64 -> 2048 fixed-random projection -- its rows are
random vectors in d_embd space. Applying a random direction cvec at
scale 30 to a residual stream is equivalent to adding a high-norm
random vector, which of course breaks decoding (the residual is no
longer on-manifold).

The honest fix: **proj_c must be trained, not random**. Or the cortex
must steer through a SUBSPACE capable of carrying semantic intent
(e.g., a low-rank subspace matching the correction's hidden direction).
Until then, scaling either does nothing (greedy decoding absorbs the
perturbation silently below the threshold) or breaks things (above
the threshold).

### What this changes about Goal 3

Reachable claims (already true):
- Cortex is the live mutation surface.
- Adler signal persists across consolidate + save/load.
- The cvec API is mechanically end-to-end working.

Unreachable claims (deferred):
- Cortex steering produces semantically meaningful output.
- CortexAgent replaces LMBackendAgent for production use.

The cortex has to LEARN what direction means what. Two paths:

  (a) **Hebbian train proj_c on actual correction contexts.** Stop
  initialising proj_c with random direction; initialise it from a small
  batch of correction hiddens via SVD. Each correction's hidden vector
  then implicitly becomes one direction in the projector's range, and
  a correction's warm_state naturally retries via that direction. Match
  the experience-autoencoder's existing pattern.

  (b) **Per-correction cvec root mean, not projection.** Forget proj_c
  altogether for steering; set the cortex's output to a temporary
  mean of recent correction hiddens (scaled small). This drops per-layer
  expressivity but it gives semantically aligned cvecs for free. Try
  (b) first because it's a 5-line change.

Either path is conscious architecture work; we're past the "wire the
LM binding" stage and into "make the cortex meaningful."

### Recommended next move

Implement (b) as a `Mode=raw_hidden` mode on KVCortex: when set, the
cortex's `emit_cvec(layer_idx)` returns `warm_state * hidden_from_last_correction`
padded to n_embd -- a smoothed semantic anchor rather than a random
projection. Compare to baseline articulate to see if the LM steers
toward the corrected sense rather than crashing.

If (b) still produces CJK garbage at meaningful amplitudes, switch to
(a): SVD the proj_c over a small batch of seed correction hiddens.
