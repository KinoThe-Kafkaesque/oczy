# 03 — Real Hidden-State Extraction at Layer L
*Feed the cortex the residual where semantic intent actually forms, not the final-layer mean-pool it metabolizes today.*

Status: REFUTED (2026-07-01, S1.4) | Thesis anchor: experiments.txt §1 (correction-gated SSM cortex), §3 (neural hippocampus) | Goal anchor: GOALS.md Goal 2 (hidden-state extraction at layer L) | Depends on / relates to: 04-context-scoped-attractors (consumer), 05-metabolism-loop-closure (consumer), 02-kv-slot-fact-injection (sibling Goal 1, blocked), 01-correction-to-competence-benchmark (eval substrate)

> **Outcome (2026-07-01):** REFUTED. S1.4 HF layer-L probe (`../experiments_logs/2026-07-01_s1_4_hf_layer_probe.md`) measured warm_sep_silhouette on two architectures: Qwen2.5-0.5B-Instruct (gap −0.083, threshold +0.10) and LFM2.5-1.2B-Instruct (gap +0.058, threshold +0.10). Mid-layer hiddens do NOT cluster by concept better than the final layer on either model. This confirms lane_03's refutation on a substrate that can see every layer — the mid-layer assumption is a model property, not a llama.cpp keyhole.
>
> **Campaign note (2026-07-11):** Campaign 0d48130 attempted to re-run Exp03 on colab but was **infrastructure-blocked** — repeated HF snapshot transfers failed before execution; no metrics or ASI scores emitted. This is not a scientific null or refutation; the authoritative pre-campaign verdict remains S1.4 above. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

## Problem

The cortex's only window into the LM is a depth-final, position-averaged summary, and that is almost certainly the wrong signal.

- `CortexAgent.perceive()` feeds `cortex.observe()` the output of `driver.peek_embedding(utterance, last_token_only=False)` (`src/oczy/experiments/cortex_agent.py:354`), which is the **final-layer, MEAN-pooled** embedding produced by `llama_cpp` `create_embedding` (`src/oczy/lm/cvec_driver.py:514-572`, pooling type MEAN). The docstring on that call (cortex_agent.py:350-353) explicitly marks it "Goal 2 staging."
- `peek_layer` does **not** exist. `cvec_driver.py:519-520` states: "Layer-L intermediate extraction is not supported here yet (binding limitation tracked under Goal 2)." GOALS.md "Goal 2 — Hidden-state extraction at layer L" is unimplemented.
- The Goal-2 rationale in GOALS.md is the gap: "Real cortex metabolism needs the residual at a layer that actually carries semantic intent — empirically layers in the middle to upper third," not the embedding/final summary the cortex sees now.
- The cortex subsystem brief records this as an unanswered open question: *"Is the final-layer mean-pooled embedding rich enough for meaningful steering, or does Goal 2 (layer-L peek) materially change the proj_hidden → warm_state signal?"* No experiment has measured it.
- We cannot answer it with the current metrics. Every recent run reports `code_qa_accuracy=1.0` (runs #79, #80, #82, #84, #85, #95, #101), and wherever a recall path succeeds it pins straight to ceiling — `domain_co_recall` 1/1 (real driver, run #95) and prefix-driven exact `co_recall` 1/1 (run #101); mock-driver exact `co_recall` is 0/0 (run #85) only because the hash embeddings carry no semantics. None of these discriminate architecture variants. A new, non-saturating **representational** metric is required to tell whether layer-L input changes the cortex at all.

Grounded layer facts (these correct the stale `n_layers=28` in CLAUDE.md KEY FACTS — and the matching "28-layer transformer" phrasing in GOALS.md's own Goal-2 rationale): the actual checkpoint config at `~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/<rev>/config.json` is `model_type=lfm2`, `hidden_size=2048`, `num_hidden_layers=16`, `vocab_size=65536`, with `layer_types` placing `full_attention` blocks at config indices **[2, 5, 8, 10, 12, 14]** (the other 10 blocks are `conv`). The GGUF driver's `llama_n_layer` also reports **16** (brief), matching HF — so "layer L" ranges over 16 blocks, and the 6 attention blocks are the most plausible carriers of routed semantic intent.

## Hypothesis

- **H1 (depth/position matter).** A cortex fed the residual at a mid/upper attention block (HF `hidden_states` index ~9 or ~13) yields `warm_state` vectors that are **more semantically separable** — paraphrases cluster, distinct concepts separate — than the same cortex fed layer-0 embeddings or the current final-layer mean-pool, by a cosine-silhouette margin ≥ 0.10.
- **H2 (structure is semantic, not a Hebbian artifact).** `proj_hidden` trained with `KVCortex.train_step` (Hebbian, `plastic-cortex/src/plastic_cortex/kv_cortex.py:499-524`) on **real** layer-L hiddens aligns its row-subspace with the top-PCA subspace of those hiddens at ≥ 2× chance **and** ≥ 2× a column-shuffled-hidden control — i.e., the learned projector reflects the real hidden manifold, not self-amplified noise.

**Falsifier.** If layer-L silhouette ≤ final-mean-pool silhouette, or if `warm_state` trajectories for layer-L vs layer-0 stay cosine ≥ 0.98 over an identical utterance stream (the input source is irrelevant), then Goal 2 buys the cortex nothing measurable and the production swap is killed.

## Why now / what unblocks it

- `transformers==5.12.1` is installed and ships an `lfm2` module (`Lfm2ForCausalLM` is importable), and the **full-precision HF checkpoint is already cached locally** (`models--LiquidAI--LFM2.5-1.2B-Instruct`). So the brief's named "twin eval with activation capture" path is a one-call `from_pretrained(..., output_hidden_states=True)` forward — no binding fork, no network.
- The validation needs **no steering/articulation**, so it sidesteps the blocked KV-slot path (Goal 1 / sibling 02) and the saturated benchmark entirely: it only exercises `cortex.observe` input → `warm_state` and `proj_hidden`, both pure-numpy (`kv_cortex.py:173-225, 499-524`).
- A binding-side fallback exists if the twin is too heavy: `llama_context_default_params` exposes `cb_eval` / `cb_eval_user_data` (the ggml backend eval callback — verified present in `llama_cpp` 0.3.31), the brief's "binding hook" path, runnable on the already-loaded Q4 context.

## Approach (ties to §1, §3)

- Implement `peek_layer(prompt, layer_idx, pooling)` on an HF twin driver returning a `d_embd=2048` float32 vector: tokenize → forward with `output_hidden_states=True` → select `hidden_states[layer_idx]` → pool (last-token default; mean optional). This is the cortex's real input window (§1).
- Drive the existing `KVCortex.observe()`/`train_step()` **unchanged**; vary **only** the peek-source layer (matched-pair single-variable, the repo standard).
- Measure (a) representational separability of `warm_state` and (b) structural alignment of `proj_hidden` — §1 (does the SSM cortex metabolize real intent?) and §3 (do the resulting hidden traces have hippocampus-worthy structure?).
- Parity-check the twin against the GGUF final embedding so we know it is the same network (control).
- **Defer** the production wiring — swapping `cortex_agent.py:354` `peek_embedding` → `peek_layer(L)` — behind passing acceptance. That swap is what unblocks 04 (context-scoped attractors need rich, separable cortex state) and 05 (metabolism loop needs real intent in, not a position-mean).

## Success criteria (discriminating, non-saturating)

These deliberately avoid `code_qa_accuracy`/`co_recall`, which already sit at 1.0 wherever they apply.

1. **`warm_sep_silhouette` (headline).** Cosine-silhouette of `warm_state` vectors over a labeled paraphrase/distinct battery. Pass: `silhouette(L_mid) − silhouette(L0) ≥ 0.10` **and** `silhouette(L_mid) − silhouette(final-mean-pool) ≥ 0.10`. Cannot saturate: within-group paraphrase variance is non-zero, so silhouette < 1 by construction.
2. **`warm_traj_cos_L_vs_L0`.** Mean per-step cosine between layer-L and layer-0 `warm_state` over an identical 16-utterance stream. Pass: ≤ 0.80 (trajectories visibly diverge — operationalizes GOALS.md done-when #2, today only a qualitative claim).
3. **`projh_align_ratio`.** PCA-subspace alignment of Hebbian-trained `proj_hidden`: real-trained ÷ shuffle-trained. Pass: ≥ 2.0 **and** real alignment ≥ 2× chance (`k/d_embd`). Operationalizes GOALS.md done-when #3 ("non-trivial structure, not random") with a number.
4. **`twin_gguf_final_cos` (parity control).** Cosine between `peek_layer(HF, final)` and `peek_embedding(GGUF, final mean-pool)`. Pass ≥ 0.85, else the twin is not trusted as the same model and the other numbers are void.

**Kill criteria.** (a) `silhouette(L_mid) ≤ silhouette(final-mean-pool)` → depth/position no better than the current input; do not swap the production path. (b) `warm_traj_cos_L_vs_L0 ≥ 0.98` → input source irrelevant. (c) `projh_align_ratio < 1.2` → `proj_hidden` "structure" is a Hebbian artifact, satisfying done-when #3 only trivially.

## Risks & open questions

- **Memory.** Full-precision LFM2-1.2B in fp32 ≈ 4.7 GB; the host ran the Q4 GGUF at ~1.6 GB RSS. Mitigation: load the twin in bf16/fp16 (~2.4 GB) and run the validation with **only** the HF twin (no GGUF needed). If still tight, use the `cb_eval` Path-B on the Q4 context.
- **Quantization/tokenization mismatch.** HF fp twin vs GGUF Q4 may diverge; if `twin_gguf_final_cos < 0.85` (different chat template/special tokens or quant drift), prefer the `cb_eval` extraction so layer-L hiddens come from the exact model the cortex steers.
- **conv vs attention blocks.** Which of the 16 blocks is richest for the cortex is unknown — swept across L0, a mid attention block, an upper attention block, and final. The 6 attention indices [2,5,8,10,12,14] are the prior, not a certainty.
- **tanh saturation.** `observe` applies `tanh(proj_hidden @ h)`; if real layer-L hiddens have larger norm than the embedding, the projection may saturate and collapse separability. Monitored via `warm_norm`; if saturated, note and scale.
- **`d_cortex` sensitivity.** Default `d_cortex=128`; stressors/curriculum use `d_cortex=4`. Separability may depend on capacity — swept as a secondary axis.
- **Out of scope.** Whether better `warm_state` separability translates into better steering/recall is **not** tested here — that belongs to 04 and 05, which consume this driver.

## Prior evidence

- `cortex_agent.py:350-354` — perceive() uses `peek_embedding(last_token_only=False)` (final-layer mean-pool) and labels it "Goal 2 staging."
- `cvec_driver.py:514-572` — `peek_embedding` is final-layer only via `create_embedding` (MEAN pooling); `peek_layer` unimplemented (docstring 519-520).
- GOALS.md "Goal 2" — done-when: `peek_layer(prompt, layer_idx)` returns d_embd; cortex on real layer-L hiddens shows visibly different `warm_state` trajectories vs layer-0; pickled `proj_hidden` trained on real hiddens shows non-trivial structure. (The rationale text still says "28-layer transformer"; the checkpoint is 16 layers — see Problem.)
- `kv_cortex.py:173-225` — `observe`: `warm = (1−plasticity)·warm + plasticity·tanh(proj_hidden @ h)`; `kv_cortex.py:499-524` — `train_step` Hebbian on `proj_hidden`, lr default 0.001, per-row L2 renorm.
- HF config (`models--LiquidAI--LFM2.5-1.2B-Instruct/.../config.json`) — `lfm2`, hidden_size 2048, num_hidden_layers 16, full_attention at config indices [2,5,8,10,12,14], vocab 65536; corrects CLAUDE.md/GOALS.md `n_layers=28`.
- `llama_cpp` 0.3.31 — `llama_context_default_params` exposes `cb_eval`/`cb_eval_user_data` (binding-hook fallback); GGUF `llama_n_layer` = 16 (brief), matching HF.

### Tracked fix-up (out of scope of this proposal but discovered while grounding it)

`KVCortexConfig.n_layers` defaults to **28** at `plastic-cortex/src/plastic_cortex/kv_cortex.py:57`, and the docstring at `:50` claims "Defaults match LFM2.5-1.2B-Instruct: d_embd 2048, n_layers 28." That is wrong — LFM2.5-1.2B has `num_hidden_layers=16` (HF + GGUF, verified above). The mismatch is silent because `proj_c` is `(n_layers, d_embd, d_cortex)`; the extra 12 rows go unprojected under the current per-layer cvec emission path, but anyone wiring a real L into `peek_layer` would index past the model's depth. **Action:** before implementing project 03, patch `kv_cortex.py:57` (and the `:50` docstring) to `n_layers=16`, or pass `KVCortexConfig(n_layers=16)` explicitly at every construction site (`run_curriculum.py:38-43`, real-driver stressors) until the default is fixed.
- Saturated-metric context: runs #79/#80/#82/#84/#85/#95/#101 all report `code_qa_accuracy=1.0`, with `domain_co_recall` 1/1 (run #95) and prefix exact `co_recall` 1/1 (run #101) at ceiling — motivation for a new representational metric.
