# Kaggle compute for offline cortex work

This directory provides private, reproducible Kaggle CPU/GPU jobs for the
offline developmental work around Research/20 and Experiment 09.

Use [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for every real research run. It
defines source/model pinning, phase-to-hardware routing, prepared generators,
submission/retrieval, provenance, meta-test gates, and the TPU admission rule.

The checked-in jobs are deliberately **infrastructure smoke tests**, not cortex
experiments. They run a synthetic learned-writer → fixed fast/slow state →
latent coupler → frozen differentiable organ path. The state remains 64×64 and
the organ interface uses Qwen2.5-0.5B-Instruct's real hidden width (896), but no
Qwen weights are loaded. The jobs do not use `eval/v2`, `meta_cortex/v1`,
retrieval, episode-conditioned code, or a real language model, and their losses
cannot be cited as evidence for H-META-CORTEX.

## What is wired

| Profile | Kaggle kernel | Requested machine | Intended later use |
|---|---|---|---|
| CPU | `abdellahkadem/oczy-cortex-cpu-smoke` | Kaggle CPU | instrument materialization, leakage/distribution audits, scoring, aggregation |
| T4 | `abdellahkadem/oczy-cortex-t4-smoke` | `NvidiaTeslaT4` | primary outer-loop and frozen-organ gradient workload |
| L4 compatibility | `abdellahkadem/oczy-cortex-l4-smoke` | `NvidiaL4X1` requested; P100 actually allocated | allocation-integrity sentinel; not a working target |
| P100 compatibility | `abdellahkadem/oczy-cortex-p100-smoke` | `NvidiaTeslaP100` | legacy-image compatibility sentinel; not a default target |
| Qwen/T4 model probe | `abdellahkadem/oczy-qwen-language-organ-t4-probe` | `NvidiaTeslaT4` plus pinned model source | frozen real-model and input-gradient verification |

All kernels are private and have internet disabled. The accepted CPU/T4 matrix
shares the exact v3 source and writes `cortex_smoke_report.json` to
`/kaggle/working`. The report records the hardware actually allocated, source
hash, throughput, held-out loss, finite-gradient checks, and before/after
hashes of every frozen parameter. See [`RESULTS.md`](RESULTS.md) for the remote
evidence and compatibility nulls.

## Local preflight

```bash
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle quota --csv

uv run python infrastructure/kaggle/run_cortex_smoke.py \
  --device cpu \
  --output /tmp/oczy-cortex-local-smoke.json
```

Authentication uses Kaggle's OAuth credentials at
`~/.kaggle/credentials.json`. Do not copy that file into this repository or a
kernel source directory.

## Submit the matrix

The explicit `--accelerator` value overrides metadata and makes the requested
GPU visible in the command history.

```bash
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels push \
  -p infrastructure/kaggle/t4-smoke \
  --accelerator NvidiaTeslaT4 \
  --timeout 900
```

Check and retrieve a run with:

```bash
kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels logs abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force
```

Repeat with the T4 kernel slug. A run counts as wired only when its JSON
artifact has `passed: true` and `device.name` matches the requested class.

The P100 and L4 profiles are retained only as compatibility sentinels. On
2026-07-09, the direct P100 request and the nominal L4 request both actually
received a Tesla P100 (compute capability 6.0). Kaggle's PyTorch 2.10.0+cu128
image supported only sm_70 and newer, so the first CUDA tensor operation
failed. Do not schedule cortex work on either path until a runtime artifact
proves a supported device or the pinned PyTorch build restores sm_60 support.

## Promotion to real offline phases

Do not turn the smoke loss into a new metric or optimize the real experiment
against it. Once `src/oczy/experiments/meta_cortex/` exists:

1. Pin a clean source commit or publish a private, hash-addressed source
   dataset; never make a remote run from an ambiguous dirty checkout.
2. Attach the official Kaggle Qwen2.5 0.5B model artifact as a version-pinned
   `model_source` so the language-organ weights are available with internet
   still disabled.
3. Use CPU jobs for Phase 0 instrument generation, scorer tests, split audits,
   distribution checks, and report aggregation.
4. Use GPU jobs for Phase 1 oracle-articulation validation and Phase 2
   developmental outer-loop training/checkpoint generation. Prefer one
   developmental seed per GPU; use DDP only after a single-seed profile shows
   that synchronized multi-GPU training is worthwhile. The smoke's
   `DataParallel` path is a hardware proof, not the production training design.
5. Keep the signed `meta_cortex/v1` meta-test inaccessible until the manifest,
   thresholds, and task counts have human sign-off. Run the one-shot meta-test
   from frozen checkpoints only; Kaggle compute does not relax that boundary.
6. Pull every JSON/checkpoint artifact, record its kernel version, source hash,
   accelerator, and exact command in `experiments_logs/` before interpreting
   it.

The [official Kaggle CLI kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
lists the accepted accelerator IDs. Availability is account- and
competition-dependent, so the artifact's detected device remains the source of
truth.

The source and kernel preparation commands used by the full workflow are:

```bash
uv run python infrastructure/kaggle/prepare_source_bundle.py --help
uv run python infrastructure/kaggle/prepare_research_kernel.py --help
```
