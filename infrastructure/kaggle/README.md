# Kaggle CPU compute for offline cortex work

This directory provides private, reproducible Kaggle **CPU-only** jobs for the
offline developmental work around Research/20 and Experiment 09.

Use [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for every real research run. It
defines source/model pinning, CPU profile generation, submission/retrieval,
provenance, meta-test gates, and the CPU-only contract.

The checked-in jobs are deliberately **infrastructure smoke tests**, not cortex
experiments. They run a synthetic learned-writer → fixed fast/slow state →
latent coupler → frozen differentiable organ path. The state remains 64×64 and
the organ interface uses Qwen2.5-0.5B-Instruct's real hidden width (896), but
no Qwen weights are loaded for the smoke task. The jobs do not use `eval/v2`,
`meta_cortex/v1`, retrieval, episode-conditioned code, or a real language
model, and their losses cannot be cited as evidence for H-META-CORTEX.

## What is wired

| Profile | Kaggle kernel | Requested machine | Intended use |
|---|---|---|---|
| CPU | `abdellahkadem/oczy-cortex-cpu-smoke` | Kaggle CPU | instrument materialization, leakage/distribution audits, scoring, aggregation |
| CPU | `abdellahkadem/oczy-qwen-cpu-probe` | Kaggle CPU | frozen real-model and input-gradient verification on CPU |

All kernels are private, have internet disabled, and set `enable_gpu: false`.
The smoke kernel writes `cortex_smoke_report.json` to `/kaggle/working`. The
report records the hardware actually allocated, source hash, throughput,
held-out loss, finite-gradient checks, and before/after hashes of every frozen
parameter. See [`RESULTS.md`](RESULTS.md) for the verified CPU smoke result and
the acceptance contract.

### CPU-only contract

Active runners and generated bootstrap code enforce a strict CPU-only
contract:

- Kernel metadata sets `enable_gpu: false` and `enable_tpu: false`.
- The generated bootstrap sets `CUDA_VISIBLE_DEVICES=""` before importing
  torch and verifies the env var is still empty at startup, with no CUDA
  or NVML runtime query performed.
- No active code calls `torch.cuda.*` for device selection, model placement, or
  profiling. Legacy report fields (`cuda_available`, `cuda_device_count`,
  `torch_cuda_version`) remain as constant `false`/`0`/`null` values for schema
  compatibility.

## Local preflight

```bash
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle quota --csv

uv run python infrastructure/kaggle/run_cortex_smoke.py \
  --device cpu \
  --output /tmp/oczy-cortex-local-smoke.json
```

The `--device` argument accepts only `cpu`. Authentication uses Kaggle's OAuth
credentials at `~/.kaggle/credentials.json`. Do not copy that file into this
repository or a kernel source directory.

## Submit and retrieve

CPU kernels do not use `--accelerator`. The metadata already specifies
`enable_gpu: false`.

```bash
# Cortex CPU smoke
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels logs abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force
```

```bash
# Qwen CPU probe
kaggle kernels push \
  -p infrastructure/kaggle/qwen-cpu-probe \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels logs abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-cpu-probe \
  --path reports/kaggle/qwen-cpu-probe \
  --force
```

A run counts as wired only when its JSON artifact has `passed: true`,
`cuda_available: false`, and `cuda_device_count: 0`. See
[`RESULTS.md`](RESULTS.md) for the full acceptance contract.

## Source bundling and kernel generation

The source and kernel preparation commands used by the full workflow are:

```bash
uv run python infrastructure/kaggle/prepare_source_bundle.py --help
uv run python infrastructure/kaggle/prepare_research_kernel.py --help
```

The kernel generator accepts `--profile cpu` (the only active profile) and
produces a private, internet-off kernel with a CPU-only bootstrap. The
bootstrap verifies a commit-addressed source archive, discovers the attached
version-pinned model, disables network-backed model resolution, records runtime
provenance, and then executes one Python module. See
[`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for the end-to-end workflow.

## Archived GPU material

Historical GPU verification (T4, P100, L4, and the T4-based Qwen model probe)
is preserved under [`archive/gpu/`](archive/gpu/). That material — including
kernel metadata, the 2×T4 throughput comparison, P100/L4 compatibility nulls,
and the T4 model probe results — is historical evidence only. GPU kernels and
metadata under the archive must not be resubmitted. See
[`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md) for the full historical
record.
