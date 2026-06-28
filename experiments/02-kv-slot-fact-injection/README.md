# Experiment: Reserved KV-slot fact injection vs the cvec ceiling

Research proposal: ../../research/02-kv-slot-fact-injection.md

## Objective

Does a text-derived KV chunk — prefilled once, snapshotted, and re-injected into the live sequence via the installed `llama_cpp` 0.3.31 per-sequence KV APIs — reproduce the live text-prefix's exact-token recall at **zero prefix re-encode per turn** on the LFM2.5 hybrid, and where does each steering surface (cvec vs KV slot) hit its information-capacity ceiling?

## Setup

- **Driver:** real LFM2.5-1.2B-Instruct Q4_K_M GGUF only. The mock `_MockDriver` (`n_embd=16`, hash embeddings) **cannot** be used — KV serialization is a real-binding feature and mock embeddings are semantically empty (`multi_fact_stressor` mock co_recall is always 0/0, run #85). Mock is retained only as a smoke-path that asserts the new code imports and the matrix wiring runs.
- **Reuse, do not reinvent:**
  - `src/oczy/lm/cvec_driver.py` — extend `LlamaCVecDriver` (already owns `_ctx_p`, `generate`, `logit_bias_generate`, `set_cvecs_per_layer`, `clear_cvec`). Add a `KVChunkDriver` subclass that calls `llama_get_memory`, `llama_decode`/`llama_batch_init`, `llama_state_seq_get_data`/`set_data`, `llama_state_seq_save_file`/`load_file`, `llama_memory_seq_cp`.
  - `src/oczy/experiments/multi_fact_stressor.py` — reuse FACTS (alpha→skylark, project-beta correction→rook, level-7→marmalade) and the three QUERIES (`:29-44`) as the probe set; reuse `_load_real_driver` and the `METRIC`/`ASI` print convention (`:575-613`).
  - Tokenization/logit-readout helpers already in `logit_bias_generate` (last-position slice of `_ctx.get_logits()`: `full[(n_last_batch-1)*n_vocab : n_last_batch*n_vocab]`, `cvec_driver.py:489`).
- **Configs:** `CVecDriverConfig(n_ctx=4096, n_threads=4, embedding=True)` (matches `multi_fact_stressor` real-driver path; note the bare `CVecDriverConfig` default is `n_ctx=512`, so `n_ctx=4096` is passed explicitly). cvec controls use the existing SVD and contrastive constructions; `articulate_scale` ≤ 0.01 where a prefix coexists (per composition log).

## Conditions / ablation matrix

Single variable = **injection surface**. Matched pairs hold prompt, query, target, and model fixed.

| # | Condition | Surface | Prefix re-encoded / turn | Expected exact-recall | Expected target rank |
|---|---|---|---|---|---|
| C0 | baseline-empty | none | 0 | no | ~47K |
| C1 | live-prefix | text prefix (`ReservedPosition`) | P (>0) | yes | rank 1 |
| C2 | kv-chunk | prefill-once + `seq_set_data`/`memory_seq_cp` | **0** | yes (H1) | rank 1 |
| C3 | kv-chunk-persisted | `seq_save_file` → `seq_load_file` (new process) | 0 | yes | rank 1 |
| C4 | cvec-svd | residual cvec, SVD of correction hiddens | 0 | no (ceiling) | ~46K |
| C5 | cvec-contrastive | residual cvec, `embed(target)−embed(default)` | 0 | no (ceiling) | ~5K |
| C6 | logit-bias | post-forward bias (needs known token id) | 0 | forced rank 1 | n/a |

Primary matched pair: **C1 vs C2** (only difference = KV reuse vs text re-encode). Secondary pair: **C4 vs C5** (only difference = cvec training). C6 is the known-good non-cvec reference (run #136-137).

## Procedure

1. Build `KVChunkDriver` over the real LFM2.5 driver; verify `llama_get_memory(_ctx_p)` returns non-null and `n_seq_max ≥ 2` (else fall back to twin-context).
2. For each FACT, form the prefix text (the fact statement) and the QUERY whose target is the fact's answer token(s).
3. **C0/C1:** run baseline and live-prefix `generate()`; record first-position logits and prompt-eval token count.
4. **C2:** `llama_decode` the prefix into scratch seq 1; `llama_state_seq_get_data` → bytes; on a fresh query, `seq_set_data`/`memory_seq_cp` into seq 0 then decode the query at `pos=len(prefix)`; record first-position logits and prefix-tokens-re-encoded (must be 0).
5. **C3:** `seq_save_file` the chunk; in a *new process*, load the model, `seq_load_file`, decode the query; record logits.
6. **C4/C5:** apply SVD and contrastive cvecs via `set_cvecs_per_layer`; record target-token rank (reproduce ~46K / ~5K anchors).
7. **C6:** `logit_bias_generate` at bias=20.0 as the rank-1 reference.
8. **Capacity ladder:** for k = 1…5 facts (reuse `--num-facts`), inject all k into one chunk (C2) vs one cvec (C4/C5), query each, count how many land at rank 1.
9. Emit `METRIC`/`ASI` lines for the autoresearch harness.

## Metrics

- **`max_logit_delta`** = max_t |logit_C2(t) − logit_C1(t)| at the first generated position (and **`logit_kl`** = KL(P_C2‖P_C1)). New continuous metric; the slot-equivalence test. Will not saturate (it is a distance, not a 0/1).
- **`prefix_tokens_reencoded_per_turn`** = count of prefix tokens passed through the transformer per query after chunk creation. C1 = P, C2/C3 = 0. Directly measurable; this *is* the "burns tokens every forward pass" cost GOALS.md names.
- **`target_token_rank`** = rank of the target token id in the last-position logits, per condition. **Replaces** the saturated exact `co_recall` (0/0 or 1/1, run #95) with a continuous rank over 65,536 — the slot-population logit test.
- **`exact_recall_at_rank1`** = fraction of facts whose target is natural argmax (no logit bias). Expected 0 for C4/C5, high for C1/C2/C3.
- **`capacity_facts_at_rank1(k)`** = number of facts at rank 1 as k grows; the headline capacity-bound curve. Separates cvec (flat ≈0) from kv-chunk (rising then competition-limited).
- **`injection_latency_ms`** = wall time of the chunk restore step.

## Acceptance & kill criteria

- **Accept H1** if C2 vs C1 `max_logit_delta` < 0.5 on ≥95% of facts AND identical top-1 token, with `prefix_tokens_reencoded_per_turn`=0.
- **Accept H2** if `capacity_facts_at_rank1(kv-chunk)` ≥ `capacity_facts_at_rank1(cvec)` + 2.
- **Kill the KV-chunk route** (pivot to binding-fork evaluation) if top-1 tokens differ on >20% of facts or median `max_logit_delta` > 0.5 — the hybrid-state-serialization failure mode. The 5–20% band between the ≥95% accept bar and the >20% kill bar is a deliberate grey zone: investigate (position-offset, tokenization mismatch, or quant drift) rather than accept-or-kill on a single run. The other-axis `{median max_logit_delta > 0.5}` kill is the symmetric complement of `{max_logit_delta < 0.5 on ≥95%}`-accept — same threshold (0.5), different aggregation (median vs p95) so it fires on systemic, not sporadic, divergence.
- **Kill the real-time claim only** (route survives) if `injection_latency_ms` > 5 ms but equivalence holds.
- **Overturn the cvec ceiling** (and rewrite the GOALS.md verdict) if any cvec condition reaches `exact_recall_at_rank1 > 0`.

## Controls

- **Matched pair C1↔C2:** identical prefix text, query, target; the only variable is text-re-encode vs KV-reuse. Isolates the slot mechanism.
- **Matched pair C4↔C5:** identical query/target; only cvec training differs. Reproduces the 46K/5K anchors from `contrastive_cvec_discovery` as a sanity check that the harness measures rank correctly.
- **Mock-vs-real:** mock asserts wiring/imports only; all behavioral numbers come from the real driver (mock embeddings are semantically empty).
- **C6 reference:** `logit_bias_generate` at bias=20.0 gives a known rank-1 forcing to bound the achievable ceiling.
- **C3 process-restart control:** isolates persistence from in-memory reuse.

## Expected failure modes

- LFM2.5 conv/recurrent state not captured by `state_seq_*` → C2/C3 diverge from C1 (KL large, top-1 changes); detected by `max_logit_delta`. (Binding flags `PARTIAL_ONLY`/`SWA_ONLY`, `llama_n_rs_seq` exist — real risk.)
- Position-offset bug → query decoded at wrong `pos` → garbage continuation despite a valid chunk.
- `n_seq_max=1` in the loaded context → `memory_seq_cp` unavailable → fall back to twin-context (two `Llama` instances).
- Capacity collapse at large k → retrieval competition inside one chunk (the run #82 length-4096 effect) — this is a *result*, the measured capacity bound, not a bug.

## Artifacts to add

- `src/oczy/experiments/kv_slot_injection.py` — new harness: `KVChunkDriver` (or a thin helper over `LlamaCVecDriver`) implementing prefill/snapshot/restore/persist, the C0–C6 matrix, the capacity ladder, and `METRIC`/`ASI` emission mirroring `multi_fact_stressor.py`.
- (optional) extend `src/oczy/lm/cvec_driver.py` with `KVChunkDriver` if the prefill helpers belong next to the ctypes boundary it already owns.

Sketch reproduce command:

```
uv run python -m oczy.experiments.kv_slot_injection --real --num-facts 3 \
    --conditions baseline,live-prefix,kv-chunk,kv-chunk-persisted,cvec-svd,cvec-contrastive,logit-bias \
    --report reports/kv_slot_injection.json
```
