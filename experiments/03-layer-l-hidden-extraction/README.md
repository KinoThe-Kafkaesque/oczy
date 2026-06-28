# Experiment: Real Hidden-State Extraction at Layer L

Research proposal: ../../research/03-layer-l-hidden-extraction.md

## Objective

Does feeding `KVCortex.observe()` the **real residual at a mid/upper layer L** (instead of the current final-layer mean-pool) make the cortex's `warm_state` measurably more semantically separable, and make Hebbian-trained `proj_hidden` carry structure that reflects the real hidden manifold?

## Setup

- **Model / driver.** Primary: a new HF twin `LayerPeekDriver` loading the locally-cached full-precision `LiquidAI/LFM2.5-1.2B-Instruct` (`Lfm2ForCausalLM`, `transformers==5.12.1`, `output_hidden_states=True`, dtype bf16, CPU). It exposes `peek_layer(prompt, layer_idx, pooling)`. No GGUF / no cvec / no articulation is needed for this experiment.
- **Parity reference.** The existing `LlamaCVecDriver` (`src/oczy/lm/cvec_driver.py`) `peek_embedding(prompt, last_token_only=False)` (final-layer MEAN) provides the parity control and the "current behavior" baseline condition.
- **Cortex.** `KVCortex` (`plastic-cortex/src/plastic_cortex/kv_cortex.py`) used **unmodified**: `observe()` (line 173), `train_step()` (line 499), `reset_warm_from_cold()`/`reset_warm_to_zeros()` (lines 400/410). `KVCortexConfig(d_cortex=128, seed=0)` so `proj_hidden` is identical across all peek-source conditions — the only variable is the hidden `h`. (Only `observe`/`train_step`/`warm_state`/`proj_hidden` are exercised, so the config's `n_layers` default — which sizes the unused articulation projector `proj_c` — is irrelevant here.)
- **Layer index convention.** HF returns `hidden_states` of length `num_hidden_layers+1 = 17`; index 0 = embedding output, index k = output of block k−1. Attention blocks (config indices [2,5,8,10,12,14]) produce `hidden_states` indices [3,6,9,11,13,15]. We probe **L0 = index 0**, **L_mid = index 9** (output of attention block 8), **L_up = index 13** (output of attention block 12), **final = create_embedding mean-pool**.
- **Scaffolds reused (extend, do not reinvent).** `cvec_driver.py` (parity + config pattern), `kv_cortex.py` (cortex API), `src/oczy/experiments/organism_curriculum/dataset.py` (Episode senses as distinct-concept anchors), `src/oczy/experiments/needle_sweep.py` (the `METRIC ...` print convention, e.g. needle_sweep.py:304), `src/oczy/experiments/eval_suite.py` (run/scorecard pattern).
- **Battery.** A deterministic in-script battery: G=6 concept groups × P=4 paraphrases = 24 prompts (groups are distinct concepts; paraphrases share meaning). A separate ordered 16-utterance "conversation" stream for the trajectory metric. A 200-prompt corpus (battery ∪ curriculum episode requests) for `proj_hidden` PCA training (M ≥ d_cortex=128).

## Conditions / ablation matrix

Single variable = peek source layer (proj_hidden seed, cortex config, prompts all held fixed).

| ID | peek source | pooling | role |
|---|---|---|---|
| R | random Gaussian, norm-matched to L_mid | — | semantic floor (no model) |
| L0 | `hidden_states[0]` (embeddings) | last | depth floor |
| LMID | `hidden_states[9]` (attn block 8) | last | treatment |
| LUP | `hidden_states[13]` (attn block 12) | last | treatment |
| LFIN | final-layer mean-pool (current `peek_embedding`) | mean | current shipped baseline |
| LMID-mean | `hidden_states[9]` | mean | pooling matched-pair vs LMID |

Secondary axis (optional, one run): `d_cortex ∈ {4, 128}` to check capacity sensitivity (4 is the stressor/curriculum default).

## Procedure

1. Build `LayerPeekDriver` (bf16 HF twin); assert `peek_layer` returns shape `(2048,)` float32 and is deterministic on repeat (twin determinism check).
2. **Parity control.** Compute `cosine(peek_layer(final, pooling='mean'), LlamaCVecDriver.peek_embedding(prompt, last_token_only=False))` on 10 prompts → `twin_gguf_final_cos`. If < 0.85, halt and switch to the `cb_eval` Path-B extraction.
3. For each condition, instantiate a fresh `KVCortex(d_cortex=128, seed=0)`; `reset_warm_to_zeros()`; for each battery prompt call `observe(peek_source(prompt), correction_signal=0.0)` from the same zero start and record the resulting `warm_state` (so warm is a deterministic function of `h`). Compute `warm_sep_silhouette`.
4. **Trajectory.** For L0, LMID, LUP separately: fresh seeded cortex, `reset_warm_to_zeros()`, feed the 16-utterance stream sequentially through `observe()`, recording `warm_state` after each step. Compute per-step `cosine(warm_L[t], warm_0[t])` → `warm_traj_cos_L_vs_L0` (mean over t).
5. **proj_hidden structure.** For LMID: collect the 200 layer-L hiddens H. (a) `P0` = untrained seeded `proj_hidden`. (b) `P_real` = seeded `proj_hidden` after E=3 epochs of `train_step` over H. (c) `P_shuf` = same training over column-shuffled H. Compute `projh_pca_alignment` for each (fraction of row energy in the top-k=32 PCA subspace of H) and `projh_align_ratio = align(P_real)/align(P_shuf)`. Report participation ratio of singular values as a secondary descriptor.
6. Print `METRIC`/`ASI`-prefixed lines (needle_sweep.py convention) and write `experiments_logs/2026-06-28_layer_l_hidden_extraction.md`.

## Metrics

- **`warm_sep_silhouette` (per condition).** Cosine-distance silhouette of the 24 `warm_state` vectors labeled by concept group, computed in-script (mean intra-group vs nearest inter-group distance; no sklearn dependency). Replaces saturated `co_recall`/`code_qa_accuracy`; bounded < 1 by non-zero paraphrase variance.
- **`warm_traj_cos_L_vs_L0`.** Mean per-step cosine between layer-L and L0 warm trajectories. New metric; directly quantifies GOALS.md done-when #2 (today only qualitative "visibly different").
- **`projh_pca_alignment` + `projh_align_ratio`.** Mean over unit rows of `‖V_k^T row‖²` where V_k = top-32 right singular vectors of centered H; chance = 32/2048 ≈ 0.0156. Quantifies GOALS.md done-when #3 ("non-trivial structure, not random").
- **`twin_gguf_final_cos`.** Parity control cosine (must be ≥ 0.85).
- **`warm_norm`** (diagnostic) — to detect tanh saturation.

## Acceptance & kill criteria

- **Accept** if all hold: `peek_layer` returns `(2048,)` float32, deterministic; `twin_gguf_final_cos ≥ 0.85`; `silhouette(LMID) − silhouette(L0) ≥ 0.10` and `silhouette(LMID) − silhouette(LFIN) ≥ 0.10`; `warm_traj_cos_L_vs_L0 ≤ 0.80`; `projh_align_ratio ≥ 2.0` and `align(P_real) ≥ 2 × chance`.
- **Kill** (report and do NOT wire `peek_layer` into `cortex_agent.py:354`) if: `silhouette(LMID) ≤ silhouette(LFIN)`; or `warm_traj_cos_L_vs_L0 ≥ 0.98`; or `projh_align_ratio < 1.2`.
- **Floor check** (sanity, not pass/fail): condition R should give `silhouette ≈ 0` and `align ≈ chance`.

## Controls

- **Single-variable matched pairs.** Identical `proj_hidden` seed, `KVCortexConfig`, prompts, and zero-start across all peek-source conditions — only `h` differs.
- **Mock-vs-real floor.** Condition R (norm-matched Gaussian, no model) is the semantic floor at the same `d_embd`.
- **Pooling pair.** LMID (last) vs LMID-mean isolates pooling from depth.
- **Shuffle control.** `P_shuf` (column-shuffled H, cross-sample covariance destroyed) isolates "structure reflects real manifold" from "Hebbian self-amplification."
- **Parity control.** Twin-vs-GGUF final cosine confirms the twin is the same network.

## Expected failure modes

- **OOM** loading the fp twin → use bf16/fp16; run HF-only (no GGUF); else fall back to `cb_eval` Path-B on the Q4 context.
- **Low parity** (`twin_gguf_final_cos < 0.85`) from chat-template/special-token or quantization drift → switch to `cb_eval` so layer-L hiddens come from the exact steered model.
- **tanh saturation** at higher layer-L norm → `warm` rows collapse, silhouette degenerate; detect via `warm_norm`, scale `h` to unit norm before `observe` and re-run.
- **conv-block ambiguity** — `hidden_states` for conv blocks may carry less routed semantics than attention blocks; the sweep (L0/LMID/LUP/final) is designed to surface this.
- **Null result** — LMID ≈ LFIN: honest kill; the final mean-pool was already sufficient input for this toy cortex.

## Artifacts to add

- `src/oczy/lm/layer_peek_driver.py` — `LayerPeekDriver(config)`: loads `Lfm2ForCausalLM` (cached, bf16, `output_hidden_states=True`); `peek_layer(prompt, layer_idx, pooling='last'|'mean') -> np.float32[2048]`; `peek_embedding` parity wrapper; `n_layers`/`n_embd` introspection. Optional `cb_eval` Path-B behind a `--backend cb_eval` flag.
- `src/oczy/experiments/layer_l_probe.py` — builds the seeded cortex, runs the battery + trajectory + proj_hidden-structure across the condition matrix, prints `METRIC`/`ASI` lines, writes the log.
- `src/oczy/experiments/tests/test_layer_peek.py` — shape `(2048,)`, dtype, determinism, `layer_idx` range guard, parity-cosine smoke (skip if checkpoint absent).
- `experiments_logs/2026-06-28_layer_l_hidden_extraction.md` — run log + condition table.

Sketch reproduce command:

```
uv run python -m oczy.experiments.layer_l_probe \
  --layers 0,9,13,final --pooling last --d-cortex 128 --seed 0 --epochs 3
```
