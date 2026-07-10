# Kaggle CPU compute verification — 2026-07-10

**Result class:** infrastructure verification, not a Research/20 experiment

**Active remote profile:** CPU only

## Active tasks

| Task | Kernel slug | Profile | Local status | Remote status |
|---|---|---|---|---|
| Cortex smoke | `abdellahkadem/oczy-cortex-cpu-smoke` | `cpu` | PASS | verified 2026-07-09 (v3) |
| Qwen CPU probe | `abdellahkadem/oczy-qwen-cpu-probe` | `cpu` | PASS | pending remote evidence |

The `cpu-smoke` task is the infrastructure plumbing verification: a synthetic
learned-writer → fixed fast/slow state → latent coupler → frozen differentiable
organ path with a 64×64 cortex state and a width-896 frozen-organ interface. No
Qwen weights are loaded, no `meta_cortex/v1` code runs, and the result is not
evidence for H-META-CORTEX.

The `qwen-cpu-probe` task verifies the pinned Qwen2.5-0.5B-Instruct model
artifact on CPU: frozen-parameter hashes, zero trainable parameters, finite
input-embedding gradient, and no parameter-fingerprint change. The kernel
metadata lives in [`qwen-cpu-probe/`](qwen-cpu-probe/) and uses
`enable_gpu: false`. **Remote acceptance for the `qwen-cpu-probe` slug is
pending** — do not claim it has passed until Main supplies remote evidence
(kernel status, pulled artifact, and provenance JSON).

## Acceptance contract

A remote run counts as accepted only when **all** of the following hold:

1. `kaggle kernels status <slug>` reports `complete`.
2. The pulled JSON artifact has `passed: true`.
3. `remote_run_provenance.json` (generated kernels) or the report JSON
   (smoke/probe kernels) records `cuda_available: false` and
   `cuda_device_count: 0`.
4. Source hash, model hashes (for probe jobs), and frozen-parameter hashes
   match the locally verified values.
5. The kernel metadata used `enable_gpu: false`, `enable_tpu: false`, and
   `enable_internet: false`.

Any run that reports CUDA availability, a non-empty `CUDA_VISIBLE_DEVICES`, or
a GPU device name is **BLOCKED** — it means the CPU-only contract was violated.

## Verified CPU smoke result (2026-07-09)

The CPU smoke kernel
[`oczy-cortex-cpu-smoke` v3](https://www.kaggle.com/code/abdellahkadem/oczy-cortex-cpu-smoke)
completed remotely on a Kaggle x86_64 CPU. The report recorded:

- 64×64 fast and slow state shapes;
- a 4×896 latent bank;
- 274,563 trainable smoke-path parameters and 3,473,792 frozen parameters;
- finite losses and gradients;
- held-out improvement of 42.11%;
- identical before/after frozen-state hash
  `8be7ef2addbe83ce0766a360f9417d9c37a4baf43f97826e0a4faff79048d331`;
- `passed: true`;
- `cuda_available: false`.

Pulled artifacts live locally under `reports/kaggle/` and are ignored by Git.

## Pinned Qwen model source

The official Qwen source
`qwen-lm/qwen2.5/transformers/0.5b-instruct/1` is version-pinned for all
model-bearing CPU jobs. The artifact is version 1, 999,604,126 bytes, with:

- model type `qwen2`;
- hidden size 896;
- 24 transformer layers and 14 attention heads;
- `config.json` SHA-256
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`;
- `model.safetensors` SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

The historical T4-based model probe is preserved as archived evidence in
[`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md). The active `qwen-cpu-probe`
task re-verifies the same model hashes on CPU; its remote result is pending.

## Historical GPU evidence

GPU verification results (T4, P100, L4) from 2026-07-09 are preserved under
[`archive/gpu/`](archive/gpu/). That material — including the 2×T4 throughput
comparison, P100/L4 compatibility nulls, and the T4 model probe — is historical
evidence only. GPU kernels and metadata under the archive must not be
resubmitted. See [`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md) for the
full historical record.

## CLI/account state

- Kaggle CLI version: **2.2.3** (upgraded from 2.1.2 via `uv tool`).
- OAuth credentials at `~/.kaggle/credentials.json` (mode `600`); no credential
  value is printed or copied.
- CPU-only jobs do not consume GPU or TPU quota.

## Exact active submission commands

```bash
# Cortex CPU smoke
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force

# Qwen CPU probe
kaggle kernels push \
  -p infrastructure/kaggle/qwen-cpu-probe \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-cpu-probe \
  --path reports/kaggle/qwen-cpu-probe \
  --force
```

No `--accelerator` flag is used. CPU kernels set `enable_gpu: false` in their
metadata; the generated bootstrap also sets `CUDA_VISIBLE_DEVICES=""` before
importing torch.
