# 02 — Beyond the cvec ceiling: reserved KV-slot fact injection

*A residual control vector changes the LM's posture; a reserved KV slot carries content. This project measures the boundary between the two and unblocks the practical slot without a binding fork.*

**Status:** REFUTED (2026-07-11) | **Thesis anchor:** experiments.txt §1 (Correction-Gated SSM Cortex), §7 (Energy / Attractor memory) | **Goal anchor:** GOALS.md Goal 1 (LM-side steering binding / reserved KV slot) | **Depends on / relates to:** 01-correction-to-competence-benchmark, 03-layer-l-hidden-extraction, 04-context-scoped-attractors, 05-metabolism-loop-closure

> **Outcome (2026-07-11):** REFUTED. Campaign 0d48130 (colab, commit `537260c`): `kv_slot_rank1_count=0.0` — the KV-chunk injection surface achieved zero rank-1 exact recalls across all probes, refuting H1 (slot equivalence). Logit biasing confirmed as the only working exact-token route (`logit_bias_rank1_count=3.0`, all `rank_logit_bias=1`). The KV-splice ≡ text-prefix parity from S1.3 holds, but the capacity bound (H2) is moot: the slot cannot recall. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

## Problem

The cortex emits steering through one of two surfaces in `src/oczy/lm/cvec_driver.py`, and both have hit a measured wall:

- **Residual cvec** (`llama_set_adapter_cvec` via `set_cvecs_per_layer` / `set_cvec_uniform`) shifts semantic domain/posture but **provably cannot force an arbitrary exact token**. The hard ceiling is documented in `experiments_logs/2026-06-27_contrastive_cvec_discovery.md`: target token `" vertical"` (id 12825) sits at rank 47,251/65,536 with no cvec, 46,322 with SVD-of-correction-hiddens cvec (unchanged), and only 5,212 with the best contrastive cvec — a 9× rank lift that **never reaches rank 1**. Five distinct cvec methods (SVD, contrastive, low-amplitude, per-position hooks, CFG logit blend) all fail exact-token recall on this 1.2B model.
- **Text prefix** (`ReservedPosition`, prepended inside `generate()` at `cvec_driver.py:398-422`) *does* force exact tokens (`prefix_steering_poc`: prefix_only exact=1/domain=1; cvec_only exact=0/domain=1), but it (a) re-pays the prefix's prompt-eval through the full transformer on every new `generate()` turn, (b) permanently occupies context-window positions, (c) interferes destructively with any cvec above `articulate_scale≈0.01` (cvec_plus_prefix collapses to exact=0/domain=0 at scale 0.03), and (d) has **no cold-persistence story** — it is carried across sessions only as a literal string, not a real KV slot.

The standing diagnosis (GOALS.md Goal 1) is that the *direct* path — writing an arbitrary `(k,v)` tensor at a chosen layer/position — is **blocked**: `Llama.kv_self` is opaque and the C++ internals are name-mangled. GOALS.md itself records the wish that `ReservedPosition` should one day "accept a prefilled KV cache chunk … so the reserved position is not burned as text tokens on every forward pass" — and treats that as future work pending binding support.

**The gap this project attacks:** the "blocked" claim is narrower than how it is used. Writing an *arbitrary cortex-emitted* `(k,v)` tensor is blocked. But a **text-derived KV chunk** — prefill the prefix once, snapshot its KV, and re-inject it into the live sequence — is *not* blocked on the installed binding (see Why now). Nobody has measured whether that route reproduces the text-prefix's behavior, what it costs, or where the information-capacity boundary between cvec and a KV slot actually sits.

## Hypothesis

- **H1 (slot equivalence + zero re-encode).** A KV chunk produced by prefilling the prefix once and re-injected via the per-sequence KV APIs reproduces the live text-prefix's next-token logits to within a small tolerance (max |Δlogit| < 0.5 at the first generated position) while re-encoding **0 prefix tokens per turn** after the first. Falsifier: divergence above tolerance, or visibly different top-1 tokens, on >20% of probes (a real risk because LFM2.5 is a conv/attention hybrid whose recurrent state may not round-trip — see Risks).
- **H2 (capacity bound is discriminating).** Held at rank-1 exact recall as a function of injected content size, the residual cvec's capacity is ≈0 facts (the proven ceiling) while the KV-chunk surface scales with chunk length up to a measurable retrieval-competition limit. The two surfaces therefore separate on a *continuous capacity curve* even though every current 0/1 recall metric saturates.

## Why now / what unblocks it

A direct probe of the installed binding (`.venv/bin/python -c "import llama_cpp"`, **llama_cpp 0.3.31**) shows the per-sequence KV machinery is already exposed as ctypes functions:

- `llama_state_seq_get_data(ctx, dst, size, seq_id)` / `llama_state_seq_set_data(ctx, src, size, seq_id)` — serialize/deserialize **one sequence's** KV cache to/from a byte buffer.
- `llama_state_seq_save_file(...)` / `llama_state_seq_load_file(...)` — persist a sequence's KV + its tokens to disk (the missing cold-persistence story).
- `llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1)` / `llama_memory_seq_rm` / `llama_memory_seq_keep` — copy/trim KV between sequences inside one context; `llama_get_memory(ctx)` returns the memory handle.
- `llama_decode`, `llama_batch_init`, `llama_batch_get_one` — low-level decode with explicit `(seq_id, pos)` so a prefix can be evaluated into a *scratch* sequence; the high-level `Llama` also exposes `eval` / `reset` / `save_state` / `load_state`.

This is exactly the "twin-eval KV prefill" / "prefilled KV cache chunk" route named in the project brief and wished for in GOALS.md — **available today, no binding fork required**. The driver already holds the raw context pointer (`LlamaCVecDriver._ctx_p`, `cvec_driver.py:143`), so wiring these calls is additive. What stays blocked (and is explicitly *out of scope* here) is forging an arbitrary `(k,v)` tensor that was never produced by a forward pass, because `state_seq_set_data` expects the opaque internal serialization format (magic/version), not raw float tensors. This project converts the blocked "arbitrary write" milestone into a tractable "text-derived prefill-and-reuse slot" milestone and measures its ceiling. Cross-links **03-layer-l-hidden-extraction** (reading the residual is the dual of writing it) and **04-context-scoped-attractors** (a scoped KV slot is the substrate for a context-scoped basin, thesis §7).

## Approach

- **Characterize the capacity bound (brief path a), thesis §7.** Treat cvec as the "basin tendency" surface and the KV slot as the "retrievable content" surface. Measure how many distinct facts each can carry at rank-1, producing the headline number `capacity_facts_at_rank1` per surface. cvec's bound is expected ≈0 (a *tendency*, never content); the KV slot's bound is the unknown this experiment reports.
- **Build the slot (brief path b: twin-eval KV prefill).** Add a `KVChunkDriver` that extends `LlamaCVecDriver`: (1) evaluate the prefix tokens once via low-level `llama_decode` into a scratch sequence; (2) snapshot with `llama_state_seq_get_data`; (3) at query time restore/copy the chunk into the live sequence (`llama_state_seq_set_data` or `llama_memory_seq_cp`) and decode only the query tokens at the correct position offset. Persist with `llama_state_seq_save_file` for the cross-session story.
- **Logits-shift slot-population test (brief path c), thesis §1.** A correction is a high-plasticity write; the test is the empty-vs-populated slot logit delta on the target token. This is the measurable behavioral test GOALS.md Goal-1 "done-when" asks for.
- **Honest scoping.** Keep the *arbitrary cortex (k,v) write* as a separately tracked, still-blocked milestone; document the two real future routes (binding fork exposing `llama_kv_cache` writes, or upstream API addition) rather than pretending they are this experiment.

## Success criteria

Discriminating, continuous metrics chosen specifically because `code_qa_accuracy` and exact `co_recall` saturate at 1.0 / collapse to 0/0 and no longer separate architectures.

- **Slot equivalence (H1):** on a held-out probe set, `max_logit_delta(kv_chunk, live_prefix)` < 0.5 at the first generated position on ≥95% of probes, AND `prefix_tokens_reencoded_per_turn` = 0 for the chunk path vs P (>0) for the live-prefix path. **Kill:** if top-1 tokens differ on >20% of probes, or `max_logit_delta` median > 0.5, declare the KV-chunk route non-equivalent on this hybrid model and pivot to the binding-fork evaluation.
- **Capacity curve (H2):** report `capacity_facts_at_rank1` for {cvec-SVD, cvec-contrastive, kv-chunk}. **Pass** if kv-chunk ≥ cvec+2 facts (a real separation). cvec is *expected* ≈0; if cvec ever reaches rank-1 the prior ceiling is overturned (also a publishable result).
- **Slot-population logit test:** `target_rank_empty − target_rank_populated` is large and monotone in chunk content (empty ≈ baseline 47K; populated → near rank 1). Continuous over 65,536, cannot saturate at 1.0.
- **Latency (GOALS Goal-1 done-when):** `injection_latency_ms` < 5 ms per chunk restore. **Kill** the "real-time" claim (not the route) if restore > 5 ms but equivalence holds.

## Risks & open questions

- **LFM2.5 is a conv/attention hybrid.** The binding exposes `LLAMA_VOCAB_TYPE_RWKV`, `llama_n_rs_seq` (recurrent-state-per-sequence), and `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` / `SWA_ONLY`; the driver also reports only n_layers=16 (attention-capable layers, per the 2026-06-23 log), consistent with un-addressed RWKV/conv hybrid layers. Sequence-state serialization may not fully capture recurrent/conv state, so the KV chunk could diverge from the live prefix. This is the top falsification risk and the reason the experiment must run on the **real** driver (the mock cannot test it).
- **Position-offset correctness.** The chunk must be restored at the same positions it was prefilled, then the query decoded at `pos = len(prefix)`. Off-by-one yields garbage — a concrete, debuggable failure.
- **`n_seq_max` / context geometry.** Multi-sequence copy needs `n_seq_max ≥ 2`; if the loaded config pins one sequence we fall back to twin-context (two `Llama` instances, as already done for CFG blend in run #133).
- **Arbitrary `(k,v)` write stays blocked.** This proposal does *not* claim to unblock cortex-emitted tensor writes; it unblocks the text-derived slot. The cortex→KV path (a learned `(k,v)` from `warm_state`) remains an open milestone.
- **Open:** does a persisted KV chunk (`save_file`/`load_file`) reload byte-identically across a process restart on LFM2.5, giving the prefix a true cold-persistence story?

## Prior evidence

- `experiments_logs/2026-06-27_contrastive_cvec_discovery.md` — cvec rank ceiling: 47,251 → 46,322 (SVD) → 5,212 (contrastive), never rank 1; runs #127/#128/#133/#136-137.
- `experiments_logs/2026-06-27_cvec_prefix_composition_tradeoffs.md` — six-surface decision matrix; logit biasing forces `" marmalade"` ([55678,786,1339]) at bias ≥ 20.0; composition run #139 (`cvec_scale=0.01 + bias=20.0 → "vertical slice of data"`); cvec+prefix interference above scale 0.01.
- `experiments_logs/2026-06-25_prefix_steering_poc.md` — prefix_only exact=1/domain=1 vs cvec_only exact=0/domain=1; interference sweep collapsing exact-uptake at scale 0.03.
- `GOALS.md` Goal 1 — blocked direct `(k,v)` write; "prefilled KV cache chunk" wish; done-when criteria (writable/overwritable slot, populated-vs-empty logit shift, <5 ms latency).
- `src/oczy/lm/cvec_driver.py` — `LlamaCVecDriver` (`_ctx_p` at :143, `generate` :398-422, `logit_bias_generate` :424-512, `peek_embedding` :514-572), `ReservedPosition`.
- `src/oczy/experiments/multi_fact_stressor.py` — FACTS skylark/rook/marmalade and three QUERIES (:29-44) reused as the capacity-ladder probe set; run #95 (real-driver exact co_recall 0/0, domain 1/1); run #82 (retrieval competition: recall collapses to 0.00 at length 4096).
- Session probe — `llama_cpp` **0.3.31** exposes `llama_state_seq_get_data/set_data`, `llama_state_seq_save_file/load_file`, `llama_memory_seq_cp/rm/keep`, `llama_get_memory`, `llama_decode`, `llama_batch_init`; high-level `Llama.save_state/load_state/eval/reset`.
