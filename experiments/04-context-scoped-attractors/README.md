# Experiment: Context-Scoped Semantic Attractors (two senses per token)

Research proposal: ../../research/04-context-scoped-attractors.md

## Status

- **Implementation:** `src/oczy/experiments/scope_selectivity_stressor.py` — implemented and tested.
- **Campaign 0d48130 (2026-07-11):** **POSITIVE** (scope selectivity) — `scope_selectivity_index=1.0`. Single-run, no cross-seed variance. Evidence: [`campaign log`](../../experiments_logs/2026-07-11_campaign_0d48130.md)


## Objective

Does replacing the cortex's single global `warm_state` with a **context-addressed slot store** let two senses of one token coexist — so correcting the technical sense in its context does not obliterate the common sense in another context — measured by a non-saturating Sense-Selectivity-Index on the 8 Stage-2 episodes?

## Setup

- **Data:** `src/oczy/experiments/organism_curriculum/stages/stage_2_scope.json` (8 episodes, each with one `retention` and one `scope` probe, both `match_mode="sense"`). Loaded via `organism_curriculum/dataset.py`; scored via `organism_curriculum/scoring.py` (`matches()`, `Episode.ambiguous_token()` at `dataset.py:96`).
- **Driver:** real `LlamaCVecDriver` (LFM2.5-1.2B-Instruct Q4_K_M) via the `_load_real_driver` pattern in `multi_fact_stressor.py`. The reference pattern is `CVecDriverConfig(n_ctx=128, n_threads=4, embedding=True)` (`run_curriculum.py:35-43`), but `n_ctx=128` is too small for Stage-2 (8 episodes × retention+scope probes per episode 16 turns of context = >512 tokens once chat-template + corrections are added). **Use `CVecDriverConfig(n_ctx=4096, n_threads=4, embedding=True)` for the real-driver real-stage runs; the `n_ctx=128` value only suits Stage-0/1 short turns and is wrong for 04 by construction.** Plus the shared deterministic `_MockDriver` (`n_embd=16`, `idx=sum(ord(c))%16`) for the mechanism control.
- **Cortex:** reference `KVCortex` (`plastic-cortex/.../kv_cortex.py`) at `KVCortexConfig(d_cortex=4)` to match the curriculum. New wrapper `ContextAddressedCortex` (artifact below) adds the slot store; the single-slot baseline is the unmodified `KVCortex`.
- **Steering:** cvecs through the existing `set_cvecs_per_layer` / `emit_uniform_cvec` path; context key from `driver.peek_embedding(prompt, last_token_only=False)` (final-layer mean-pooled, the only depth available today — `peek_layer` is Goal 2 / project `03`, not implemented).
- **Reused scaffolds:** `cortex_agent.py` (perceive/articulate lifecycle), `multi_fact_stressor.py` (`_MockDriver`, `_load_real_driver`, METRIC-/ASI- print contract), `organism_curriculum/{dataset,scoring}.py`. Prefer extending these over new infrastructure.

## Conditions / ablation matrix

Matched single-variable pairs (repo standard). All use `d_cortex=4`, same plasticity, same cvec scale.

| Condition | Cortex addressing | Context key source | Driver | Single variable vs |
|---|---|---|---|---|
| **A** `single_slot_real` (baseline/control) | global `warm_state` (current `KVCortex`) | n/a | real LFM2.5 | — |
| **B** `addressed_peek_real` | context-addressed slots | `peek_embedding` (final-layer pooled) | real LFM2.5 | A → isolates **addressing** |
| **C** `addressed_oracle_real` | context-addressed slots | oracle one-hot per request context | real LFM2.5 | B → isolates **key quality** |
| **D** `addressed_peek_mock` | context-addressed slots | mock hash embedding | mock | B → isolates **driver semantics** |

A↔B isolates the addressing mechanism. B↔C isolates whether the bottleneck is the context key (hands off to `03`). B↔D isolates whether keys carry meaning at all.

## Procedure

1. For each episode, **pre-teach probe**: answer the retention and scope probes with the cortex at cold/zero; record both answers (baseline = LM prior, expected ~common sense on both).
2. **Teach**: feed the `correction_utterance` + `corrected_response` through `perceive`→`observe` with `correction_signal=1.0`. In conditions B/C/D the write goes to the addressed slot keyed by the *teaching request* hidden; in A it updates the global `warm_state`.
3. **Post-teach probe**: re-answer the retention probe (context = teaching request) and the scope probe (different request). In B/C/D the read is keyed by each probe's request; gated to zero steering below `read_threshold`.
4. Score every answer with `scoring.matches(..., match_mode="sense")` using `Episode.ambiguous_token()` so the ambiguous token itself is discounted.
5. Aggregate SSI, retention_acc, scope_acc, obliteration_rate, slot_count, memory_bytes; print METRIC-/ASI- lines for the autoresearch harness.
6. Repeat across the 4 conditions; emit one matched-pair table.

## Metrics

- **`retention_acc`** = mean over 8 episodes of `[retention answer sense-matches corrected_label]`. (Did the taught technical sense stick in-context.)
- **`scope_acc`** = mean over 8 episodes of `[scope answer sense-matches the probe's `expected` common sense]`. (Did the original sense survive in the other context.)
- **`SSI` (Sense-Selectivity-Index, headline)** = mean over 8 episodes of `[retention_correct AND scope_correct]`. Per-episode conjunction. **Replaces** the binary Stage-2 scope-uptake and `code_qa_accuracy` (both saturate at 1.0 or sit at 0). SSI cannot be gamed by collapsing to one sense and currently sits at ~0 → full headroom.
- **`obliteration_rate`** = mean over episodes of `[scope answer sense-matches corrected_label]` (taught technical sense leaking into the common-sense context). Continuous discriminator: ~1.0 for a single global slot, target ≤ 0.25 addressed.
- **`slot_count`, `memory_bytes`** = allocated basins and bytes (`M·(d_key+d_cortex)·4`). Feeds north-star `behavior_delta_per_byte_of_persistent_memory` (`rl_pipeline_design.md:342`) and the `06` growth budget.

## Acceptance & kill criteria

**Tunable defaults** (referenced in the conditions and kill criteria below; override via CLI swaller flags in the sketch command):
- `slots_cap = 24` — slot-store capacity (≥3× expected distinct contexts; 8 episodes × ~3 contexts each 24, so the growth-kill trip at `slot_count > 16` fires before this is hit).
- `alloc_threshold = 0.6` — cosine below which a new slot is allocated rather than the nearest one updated. Picked above the LM-prior paraphrase cosine (~0.5 on d=2048 final-layer mean-pool) so paraphrases of the *same* context do not split.
- `read_threshold = 0.5` — cosine below which the read returns zeros (no steering). Deliberately one notch below `alloc_threshold` so a context that did not bank a slot sees no accidental steering.

- **ACCEPT:** B (or C) reaches **SSI ≥ 0.5** with `retention_acc ≥ 0.75` AND `scope_acc ≥ 0.75` AND `obliteration_rate ≤ 0.25`, while baseline A scores **SSI ≤ 0.125**.
- **KILL — mechanism:** C (oracle key) does not beat A on SSI by ≥ 0.25 → addressing/read itself is insufficient; abandon context-addressing.
- **KILL — key quality (escalate to `03`):** C passes but B ≤ A+0.125 → final-layer pooled key cannot separate senses; hand to `03-layer-l-hidden-extraction`.
- **KILL — growth:** `slot_count > 16` (2× distinct contexts) on the 8 episodes → allocation uncontrolled; defer to `06`.

## Controls

- **Matched pairs:** A↔B (only addressing changes), B↔C (only key source), B↔D (only driver). Every other config (d_cortex, plasticity, cvec scale, prompts) held fixed.
- **Mock-vs-real:** D mock is the mechanism-only control — `_MockDriver` keys are semantically empty (`idx=sum(ord(c))%16`), so D must NOT pass on semantics; it only confirms allocation/read fire and slots stay bounded.
- **Oracle-key upper bound:** C removes key noise to attribute any B failure to the key (→ `03`) vs the mechanism.
- **Pre/post contrast:** step 1 baseline answers confirm the common sense is the LM prior, so a passing scope probe reflects preservation, not luck.

## Expected failure modes

- **Key collision:** `"Log the runtime error."` vs `"Show the log."` don't separate at the final layer → B fails, C passes → motivates `03`.
- **Read-gate cliff:** no `read_threshold` cleanly steers in-context yet stays silent out-of-context (echoes the cvec scale cliff, GOALS.md / 2026-06-24); retention and scope can't both clear 0.75.
- **Answer-path bypass:** the LM ignores a weak gated cvec and answers from prior in both contexts → scope passes trivially, retention fails (SSI still low — caught by the conjunction).
- **Slot blow-up:** `alloc_threshold` too strict → a new slot per probe → growth kill fires.
- **Mock false positive:** hash keys accidentally separate on the 8 short strings → D "passes"; treated as a control artifact, not evidence.

## Artifacts to add

- `src/oczy/experiments/context_attractor_cortex.py` — `ContextAddressedCortex` wrapping `KVCortex`: slot arrays `(keys[M,d_key], deltas[M,d_cortex])`; `write_addressed(context_key, hidden, correction_signal)` (nearest-slot or allocate at `alloc_threshold`, correction-gated EMA reusing `proj_hidden`/`alpha_*`); `read_addressed(context_key, temperature, read_threshold)` (softmax read into `warm_state`, zero-gated below threshold); `consolidate_slots()` (cold fold + near-duplicate merge). Leaves the reference 9/9 contract untouched.
- `src/oczy/experiments/scope_selectivity_stressor.py` — loads `stage_2_scope.json`, drives conditions A–D, computes SSI / retention_acc / scope_acc / obliteration_rate / slot_count / memory_bytes, prints METRIC-/ASI- lines. Reuses `_MockDriver`/`_load_real_driver` from `multi_fact_stressor.py` and `scoring.matches`.
- New log `experiments_logs/2026-06-28_scope_selectivity_attractors.md` + SUMMARY.md entry.

Sketch reproduce command:

```
uv run python -m oczy.experiments.scope_selectivity_stressor \
  --driver real --condition addressed_peek \
  --d-cortex 4 --slots-cap 24 --alloc-threshold 0.6 --read-threshold 0.5
# matched baseline:
uv run python -m oczy.experiments.scope_selectivity_stressor \
  --driver real --condition single_slot --d-cortex 4
# oracle-key + mock controls:
uv run python -m oczy.experiments.scope_selectivity_stressor --driver real --condition addressed_oracle
uv run python -m oczy.experiments.scope_selectivity_stressor --driver mock --condition addressed_peek
```
