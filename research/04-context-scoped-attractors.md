# 04 — Context-Scoped Semantic Attractors

*Two senses of one token must coexist as distinct basins — correcting one sense must not obliterate the other.*

Status: PROPOSED | Thesis anchor: experiments.txt §7 (energy/attractor memory), §1 (correction-gated cortex), §4 (learned plasticity) | Goal anchor: Goal 3 (organ tensor upgrades; scope-control done-when: "correction of one sense does not obliterate the other because they live in different cortex state regions"), relates Goal 2 (layer-L peek) | Depends on / relates to: `03-layer-l-hidden-extraction`, `01-correction-to-competence-benchmark`, `06-bounded-growth-consolidation`, `05-metabolism-loop-closure`

## Problem

The organism cannot hold two meanings of one word at once. Stage 2 of the curriculum (`src/oczy/experiments/organism_curriculum/stages/stage_2_scope.json`, 8 episodes: `s2_log`, `s2_file`, `s2_key`, `s2_cell`, `s2_record`, `s2_branch`, `s2_model`, `s2_run`) teaches a *technical* sense of a token already known in a *common* sense, then tests both. Each episode carries two `match_mode="sense"` probes: a `retention` probe in the teaching context and a `scope` probe in a different context. For `s2_log` the correction is `"No, 'log' here means system error log."`; the retention probe `"Log the runtime error."` expects `system error log`, and the scope probe `"Show the log."` expects `captain's journal`.

The scope test fails 100%. On the real LFM2.5 driver, Stage 2 uptake stayed **0/8 across runs #73–#77**. The only variant that moved the needle (run #78, `policy_suppresses_fast_answer`) reached **0.62 stage-2 uptake** (and only **0.50** on the separate scope-discrimination metric) but **collapsed Stage 0 retention to 0.62 and Stage 1 transfer to 0.12** — it bought scope by destroying the rest. That trade-off is the whole problem: the cortex has one global state slot, so teaching sense B *moves the same vector* that encoded sense A.

The mechanism is explicit in the reference cortex. `KVCortex` holds `warm_state` and `cold_state` as **single `d_cortex` vectors** (`kv_cortex.py:107-108`), and `observe()` updates that one vector by a global EMA: `warm = (1-plasticity)*warm + plasticity*tanh(proj_hidden @ h)` (`kv_cortex.py:173-225`). There is no context address. Whatever the current request, the same `warm_state` projects into the same cvecs. The real-driver curriculum runs this at `KVCortexConfig(d_cortex=4)` (`run_curriculum.py:38-43`) — four scalars to hold every sense of every word. One basin, globally reshaped: exactly the overgeneralization the thesis warns about ("If the basin gets reshaped globally, 'profile' might become business vertical everywhere", `experiments.txt:427`).

The benchmark that should catch this is saturated: `code_qa_accuracy=1.0` across runs #75–#79. We need both a new cortex mechanism and a non-saturating scope metric (cross-link `01`).

## Hypothesis

- **H1 (mechanism).** If the cortex stores correction-deltas in **context-addressed slots** — a small associative store keyed by the request's hidden state, read by similarity at articulation time — then two senses of one token settle into two distinct basins, and correcting one sense leaves the other's retrieval intact. Measured: a single-slot baseline scores Sense-Selectivity-Index (SSI, both probes correct per episode) ≤ 0.125, while the context-addressed cortex scores SSI ≥ 0.5 on the same 8 episodes, with retention and scope accuracy *both* ≥ 0.75 (no run-#78 trade-off).
- **H2 (selectivity for free).** Because the *common* sense is the LM's natural prior, a context-addressed read that returns ~zero steering when no basin matches the request will preserve the common sense automatically. Measured: obliteration_rate (taught technical sense leaking into the common-sense context) drops from ~1.0 (single slot) to ≤ 0.25 (context-addressed), without a prefix that bakes the answer in.

## Why now / what unblocks it

- The cvec surface already does the *right kind* of thing for this test. cvec shifts semantic **DOMAIN/posture** reliably (`domain_co_recall 1/1`, run #95) even though it cannot force an arbitrary exact token. Stage-2 scoring is `match_mode="sense"` (token overlap minus stopwords and the ambiguous token, `scoring.py` / `dataset.Episode.ambiguous_token()` at `dataset.py:96`), i.e. a **domain-level** discrimination. So the failure is not the cvec ceiling — it is the *single global slot*. Context-addressing is the missing piece, and it is addressable in pure numpy on top of the existing `KVCortex`/`CortexAgent` plumbing.
- Thesis §7 is a literal spec for this: `energy E(h, context)`, correction lowers the desired interpretation's energy and raises the wrong one, "the basin must be scoped by context" (`experiments.txt:419-427`). Modern-Hopfield attention *is* the read rule. We are implementing the named design, not inventing one.
- Context keys can come from `peek_embedding` (final-layer mean-pooled, available today). If the two short requests `"Log the runtime error."` vs `"Show the log."` do not separate at the final layer, that failure directly motivates and is unblocked by `03-layer-l-hidden-extraction` (mid-layer `peek_layer`). The experiment is designed to *diagnose* which.

## Approach

Make warm/cold cortex state **context-addressed** instead of global, following thesis §7's energy/attractor framing and §1's correction gate.

- **Slot store.** Add an associative memory of `M` slots, each a `(key_m ∈ d_embd-or-projected, delta_m ∈ d_cortex)` pair, on top of `KVCortex` (a wrapper in `experiments/`, leaving the 9/9 reference contract untouched).
- **Correction-gated write (basin carving).** On `observe(hidden, correction_signal)`, compute the candidate delta `tanh(proj_hidden @ hidden)` (reuse `proj_hidden`) and the context key from the request hidden. Find the nearest slot; if max similarity < `alloc_threshold` (novel context) allocate a new slot, else EMA-update *only that slot* with the correction-gated plasticity. Writes are **local** — they cannot reshape a basin that the current context does not address (§7's scoping; §4's plasticity gate).
- **Similarity read (settling).** At articulation, compute the request key, softmax-attend over slot keys (temperature β), set `warm_state = Σ softmax(key·key_m/β) · delta_m`, then emit cvecs as today. **Gate the read**: if max similarity < `read_threshold`, return zeros → no steering → the LM falls into its natural (common-sense) basin (H2).
- **Bounded growth.** Cap slot count and merge near-duplicate keys, tying allocation to cross-link `06-bounded-growth-consolidation` and the north-star `behavior_delta_per_byte_of_persistent_memory` (`rl_pipeline_design.md:342`). `consolidate()` folds stable slots into cold storage (§ slow change / forgetting raw trace).

## Success criteria

Behavioral, on the 8 Stage-2 episodes, real LFM2.5 driver. Replaces the saturated `code_qa_accuracy` / old binary scope-uptake with a **joint** metric that cannot be gamed by collapsing to one sense.

- **PASS:** context-addressed cortex achieves **SSI ≥ 0.5** (≥ 4/8 episodes with retention AND scope *both* correct), with `retention_acc ≥ 0.75` AND `scope_acc ≥ 0.75`, while the matched single-slot baseline scores **SSI ≤ 0.125** (consistent with current 0/8). And **obliteration_rate ≤ 0.25** (H2).
- **Discriminating-by-construction:** SSI is the per-episode conjunction. An always-technical cortex fails every scope probe; an always-common cortex fails every retention probe; only genuine per-context selectivity scores. Current value is ~0 (briefs), so there is full headroom — it does not start at 1.0.
- **KILL (mechanism):** if the **oracle-key** context-addressed condition (clean orthogonal key per request) cannot beat the baseline SSI by ≥ 0.25, the read/write addressing itself is insufficient and context-addressing is not the lever — pivot away.
- **KILL (key quality, hand off to 03):** if oracle-key passes but `peek_embedding`-key SSI ≤ baseline+0.125, the final-layer pooled key cannot separate the two senses → escalate to `03` (mid-layer `peek_layer`) rather than claim success.
- **KILL (growth):** if allocated slots exceed 2× the number of distinct request contexts (~16), allocation is uncontrolled → fail (defer to `06`).

## Risks & open questions

- **Final-layer key collision.** `"Log the runtime error."` and `"Show the log."` are short and share `log`; final-layer mean-pooled embeddings may not separate them. This is the most likely failure and is *the* diagnostic that hands the problem to `03`.
- **cvec answer-path leakage.** The curriculum answers via the LM's own decoding. If the LM ignores a weak gated cvec and answers from its prior in *both* contexts, scope passes trivially but retention fails. The basin must steer hard enough in-context yet read ~zero out-of-context — the `read_threshold` is the knob; risk it has no clean setting (mirrors the cvec scale cliff, GOALS.md / 2026-06-24 sweep).
- **Mock has no semantics.** `_MockDriver` keys (`n_embd=16`, `idx=sum(ord(c))%16`) cannot carry meaning; mock is a *mechanism-only* control (does allocation/read fire correctly), never a semantic pass.
- **Open:** should slot keys be the raw `peek_embedding` or a learned `proj_key` projection? Should basins be per-token or global-by-context? Does `consolidate()` need a per-slot cold store (vs the single `cold_state` vector today, `kv_cortex.py:354-408`)?

## Prior evidence

- Stage-2 fails 0/8 on the real driver across runs #73–#77; run #78 reached **stage-2 uptake 0.62** (scope-discrimination 0.50) only by collapsing Stage 0→0.62 / Stage 1→0.12 (RL-pipeline brief, `2026-06-26_policy_head_ranking_loop.md`).
- The cortex's single global `warm_state` EMA: `kv_cortex.py:107-108` (state vectors), `kv_cortex.py:173-225` (`observe`), real-driver `d_cortex=4` (`run_curriculum.py:38-43`).
- cvec does domain not exact tokens: `domain_co_recall 1/1` exact `0/0` (run #95, `2026-06-27_domain_recall_metric.md`); five-method cvec exact-token ceiling (`2026-06-27_contrastive_cvec_discovery.md`).
- Saturated benchmark: `code_qa_accuracy=1.0` runs #75–#79 (SUMMARY.md).
- Scoring contract: `scoring.matches()` sense mode, `dataset.Episode.ambiguous_token()` (`dataset.py:96`).
- Thesis §7 energy/attractor + scope-by-context + overgeneralization danger (`experiments.txt:419-427`).
