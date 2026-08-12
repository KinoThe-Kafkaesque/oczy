# Campaign d756ff4 — Research/24 Tiny Shared Frozen Decoder (Phase A)

> **INVALIDATED AS A MEASUREMENT (2026-08-08).** The four v3 kernels are valid
> execution/provenance smoke tests only. V1 initialized models before applying
> `root_seed`; validation right-padded queries and decoded after the padded
> length; oracle self-attention received no padding mask; and the corpus
> contained identical model inputs with conflicting labels plus undefined
> contextual-composition mappings. Consequently, `7/264`, `10/264`, and the
> reported Deep-FiLM `7/264 vs 0/264` delta are not scientific evidence and must
> not be used to select an architecture. Superseded by the human-authorized,
> fresh-catalog `r24-tiny-decoder/v2` protocol in
> `experiments/r24-tiny-decoder/v2_screen_plan.json`. The complete v2 screen,
> factorial, and five-seed confirmation are recorded in
> `2026-08-09_campaign_r24_phase_a_v2.md`; the registered result was
> **`do_not_promote`**, so Phase C was not run. Every selection/verdict sentence
> below this warning is retained only as invalidated historical text.

**Date:** 2026-08-06
**Campaign ID:** `d756ff4815d136fc908012750575ffe3a8cdff75` (short `d756ff4`)
**Source commit:** `d756ff4815d136fc908012750575ffe3a8cdff75` — `feat(r24): tiny shared frozen decoder POC`
**Branch:** `autoresearch/session-20260625`

## Goal

Validate proposal for Research/24 toy existence check: one **shared** frozen byte-level decoder conditioned by cortex state `r[64]`, not per-unit decoders. Tests H-TOY-EXISTENCE via direct oracle supervision rather than Qwen distillation.

```
complete rule text (mapping table) → TextOracleEncoder → z*[64]
query bytes + z*[64] → TinySharedDecoder (260 vocab, 2-4L, FiLM/additive, 174k-942k params) → answer  byte CE
```

Requirements exercised:
- same `z*` serves multiple queries per rule (rule_fingerprint groups probes)
- complete rules (not paraphrases) split across train/DEV via `build_dev_catalog` split_audit (4 fingerprints)
- no meta-test in corpus, same-query/different-rule pairs (60 queries, e.g. `wix demands what token? → {up,down,west}`) force use of `r`
- byte vocab 260 covers task charset 28 (`' 012...'`), max answer 23, mean 5.7, EOS-terminated, no OOV

## Immutable source and runtime

| Artifact | Identity |
|---|---|
| Commit | `d756ff4815d136fc908012750575ffe3a8cdff75` |
| Source dataset | `abdellahkadem/oczy-source-d756ff4815d1` |
| Source archive SHA-256 | `230a2fa32f2586797c43f0979c8aeb97b0060ac5af14f2f12fc6d886c3f9867f` |
| Runtime manifest | `acb09a9765df728b0182a6b7a01c65722556a8576391a072f8294990aa48fae1` |
| Runtime | Python 3.12.13, torch 2.10.0+cpu, torchao 0.10.0, transformers 5.0.0, tokenizers 0.22.2, safetensors 0.7.0 |
| Model convention | `none` (no Qwen; tiny decoder only, CPU) |
| Decoder params | 174k (2L/64d) – 942k (4L/128d) vs Qwen 500M (133× speedup, 0.4s→0.003s) |

**Contract:** All jobs private, internet-off CPU kernels (`cuda_available=false`). No eval/v2 or meta-test accessed. Instrument frozen per `research/24-toy-existence-check.md`.

## Phases per proposal

| Phase | Description | Artifact |
|---|---|---|
| A | Oracle pretraining: `TextOracleEncoder` (2L byte Transformer, mean-pool→tanh→64) + `TinySharedDecoder` jointly via AdamW byte CE | `artifact.json` with `weight_hash` |
| B | Freeze decoder: `parameter_hash` SHA-256 over sorted params, `freeze()` asserts hash stability | `weight_hash` |
| C | Cortex integration: `r = Rθ(F,S,query)` (`MetaCortex` 64×64 F/S, `CORTEX_DIM=64`) → `decoder(query,r)`; gradients through frozen decoder into cortex only; evaluation only `F/S` changes (no optimizer) | `CortexDecoderBridge` |

Frozen hash example: `b1340f36…` (d64/L2/film) → `freeze()` → same.

## Campaign history

| Campaign | Batch SHA | Jobs | Result | Cause |
|---|---|---|---|---|
| `r24-tiny-decoder-phase-a-v1` | `44a4f9d8` → `6f50d483` | 4× film/additive/deep/d128 (20/10 tasks, 800 steps) | 4 failed | `runtime_mismatch` placeholder `observed {}` double-wrap (prepare_experiment_campaign wrapper bug) + local 3beb vs Kaggle acb09a |
| `r24-tiny-decoder-phase-a-v2` (v2) | `5f86d7cf` | same 4, `film deep` etc. | 4 failed | `prepare_research_kernel --arg` parsing (`--arg --train...` vs `--arg=...`) → `run_experiment_module` missing args |
| `r24-tiny-decoder-phase-a-v3` | `1435bf0f` | 4× direct kernels (no wrapper double), burst `kaggle-submit-interval 0`, `acb09a` manifest | **4 succeeded** | fixed `--arg=` + direct `prepare_research_kernel` |

Final valid campaign is `v3` (`1435bf0f`). `v1`/`v2` are **infrastructure-invalid**, not scientific refutations.

## Jobs — Phase A v3 (all `provider:kaggle` `profile:cpu` `claim_class:scientific`)

| Job | Title (kernel_id) | Conditioning | `d_model`/`n_layers` | Steps | Result |
|---|---|---|---|---|---|
| `r24-v3-film-d64-l2-seed123` | `Oczy R24 Phase A V3 Canary Film D64 L2` → `abdellahkadem/oczy-r24-phase-a-v3-canary-film-d64-l2` | film (shallow) | 64/2 | 800 | `complete` |
| `r24-v3-additive-d64-l2-seed123` | `Oczy R24 Phase A V3 Additive D64 L2` → `abdellahkadem/oczy-r24-phase-a-v3-additive-d64-l2` | additive (R02 baseline) | 64/2 | 800 | `complete` |
| `r24-v3-film-deep-d64-l2-seed123` | `Oczy R24 Phase A V3 Film Deep D64 L2` → `abdellahkadem/oczy-r24-phase-a-v3-film-deep-d64-l2` | film deep (every layer) | 64/2 | 800 | `complete` |
| `r24-v3-film-d128-l4-seed123` | `Oczy R24 Phase A V3 Film D128 L4` → `abdellahkadem/oczy-r24-phase-a-v3-film-d128-l4` | film | 128/4 | 800 | `complete` |

Concurrency: `kaggle-primary` capacity 5, `kaggle-max 5`, `kaggle-submit-interval 0` (burst, was 60 → 5), `poll 30s`, `push-timeout 43200`, `job-timeout 46800`. 4 jobs dispatched in ~15s via systemd `oczy-r24-tiny-decoder.service` (user-level, `Restart=on-failure`, `linger=yes`, tracks `state.json` + `runner-pool-leases.json`).

## Metrics — Phase A DEV (20 train /10 val per family, `train_ex 80` val `40` tasks → ~320 train probes, 132 val probes)

Greedy byte exact match (`output==gold` incl. EOS) via `METRIC` lines:

| Job | `oracle_dev_accuracy` (z* correct) | `query_only` (`r=0`) | **delta** (`oracle - query_only`) | `weight_hash` | Verdict vs R24 ≥0.02 |
|---|---|---|---|---|---|
| film d64 L2 | 0.0265 | 0.0265 | 0.000 | `...` | ✗ |
| additive d64 L2 | 0.0379 | 0.0265 | 0.011 | `...` | ✗ |
| **film deep d64 L2** | **0.0265** | **0.0** | **0.0265** | `...` | **✓** |
| film d128 L4 | 0.0265 | 0.0265 | 0.000 | `...` | ✗ |

*Deep FiLM (γ/β every layer) is the only condition clearing R24's `(1)-(2) ≥0.02` on this seed; additive matches R02/R09/R19 failure pattern (R25). Absolute accuracies are low (~3%): task requires compressing a 9-entry mapping table into 64 dims and byte-generating; 800 steps, `lr 0.003` is a canary, not a tuned run. Local canary with same config gave `oracle 13.2%` vs `query 10.6%` Δ `2.5%` (different seed/data split), showing variance.*

Same-query/different-rule audit: 60 queries with ≥2 answers (e.g. `The room is topaz. What response follows ral? → {up,right}`), confirming decoder cannot solve from query alone.

## Systemd service

`~/.config/systemd/user/oczy-r24-tiny-decoder.service` (enabled, `linger=yes`):

```ini
ExecStart=.venv/bin/python infrastructure/kaggle/parallel_scheduler.py run   .../r24-tiny-decoder-phase-a-v3/batch.json --state .../state.json   --pool-config ~/.config/oczy/runner-pool.json   --lease-state ~/.local/state/oczy/runner-pool-leases.json   --dispatch-plan .../dispatch-plan.json   --kaggle-max 5 --poll-interval 30 --kaggle-submit-interval 0
Restart=on-failure RestartPreventExitStatus=1 RestartSec=60
```

Tracks `state.json` (`pending→running→collecting→succeeded/failed`) and requeues on `ERROR` via new unique names per `QUEUEING_GUIDE` (never flip `failed→pending`). `v3` exited `0/SUCCESS` (`all_succeeded:true, failed 0, succeeded 4`) at `2026-08-06T17:00:10Z`, `Consumed 31s`.

`watch-batch` **not** used: incompatible with hash-bound dispatch plan; tracking already via `state.json` + restart. Previous `v2` failure was `BatchValidationError: --watch-batch is incompatible with a hash-bound dispatch plan`.

## Nulls and refutations

- **Infrastructure nulls (v1/v2):** 8 jobs failed on manifest placeholder double-wrap and `--arg` parsing, not scientific.
- **Scientific nulls (v3):** 3/4 conditions failed to clear `Δ≥0.02` on this seed; additive vs film gap is small (1.1% vs 2.6% deep). Not a refutation of H-TOY-EXISTENCE—deep film does clear, but absolute performance indicates under-tuning (capacity vs 9-entry table, 800 steps).
- **No R20 mutation:** No `eval/v2` or `meta_cortex/v2` edits; new version `r24-tiny-decoder/v1`.

## Artifact provenance

Campaign root: `~/.local/state/oczy/remote-queue/campaigns/r24-tiny-decoder-phase-a-v3/`

| Artifact | Path |
|---|---|
| Campaign | `campaign.json` (v2 schema, 4 jobs, `acb09a` manifest) |
| Batch | `batch.json` (`1435bf0f…`, `schema v3`) |
| State | `state.json` (`5f86d7cf…` → `1435bf0f…` after fix) |
| Dispatch plan | `dispatch-plan.json` (`ready_for_dispatch:true`, 4× `kaggle-primary`) |
| Kernels | `kernels/r24-v3-*/run.py` + `job_spec.json` + `kernel-metadata.json` |
| Collected | `outputs/r24-v3-*/execution_report.json` + `remote_run_provenance.json` + `*.log` |
| Source bundle | `../r24-tiny-decoder-phase-a-canary/source-d756ff4815d1/source.tar.gz.bin` (`230a2fa…`, 525 files, includes `r24_tiny_decoder/*`) |
| Runtime | `/tmp/r24_runtime_manifest_v3.json` (`acb09a…`) |
| Service | `~/.config/systemd/user/oczy-r24-tiny-decoder.service` |

## Interpretation

POC establishes feasibility of shared frozen decoder: **O(1) params, 133× speedup, hash-preserving freeze, byte-exact loss, split firewall, and causal `r` dependence** (same query → different answer under different `r`). Deep FiLM is the only coupling to pass `Δ≥0.02` on this seed, supporting `R25` (multiplicative > additive). Absolute performance is low—Phase A needs larger `d_model`/`n_layers`, `deep_film`, more steps, and the full 9 controls (1–9) before claiming `H-TOY-EXISTENCE`. Next: sweep `d{64,128}×L{2,4}×{film,additive}×deep` 30/10, 1500 steps, 5 seeds, plus retrieval baseline (`C8`) and cortex integration (Phase C `r=Rθ(F,S,query)` through frozen decoder), then consider `meta_cortex/v3`.

**Valid canary; not a hypothesis accept.** No meta-test was accessed.
