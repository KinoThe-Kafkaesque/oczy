# Kaggle CPU compute verification — 2026-07-10

**Result class:** infrastructure verification, not a Research/20 experiment

**Active remote profile:** CPU only

## Active tasks

| Task | Kernel slug | Profile | Local status | Remote status |
|---|---|---|---|---|
| Cortex smoke | `abdellahkadem/oczy-cortex-cpu-smoke` | `cpu` | PASS | verified 2026-07-10 (v4) |
| Generated bootstrap probe | `abdellahkadem/oczy-cpu-bootstrap-probe` | `cpu` | PASS | verified 2026-07-10 (v4) |
| Qwen CPU probe | `abdellahkadem/oczy-qwen-cpu-probe` | `cpu` | PASS | verified 2026-07-10 (v1) |

The `cpu-smoke` task is the infrastructure plumbing verification: a synthetic
learned-writer → fixed fast/slow state → latent coupler → frozen differentiable
organ path with a 64×64 cortex state and a width-896 frozen-organ interface. No
Qwen weights are loaded, no `meta_cortex/v1` code runs, and the result is not
evidence for H-META-CORTEX.

The `qwen-cpu-probe` task verifies the pinned Qwen2.5-0.5B-Instruct model
artifact on CPU: frozen-parameter hashes, zero trainable parameters, finite
input-embedding gradient, and no parameter-fingerprint change. The kernel
metadata lives in [`qwen-cpu-probe/`](qwen-cpu-probe/) and uses
`enable_gpu: false`. Remote acceptance verified on 2026-07-10 (v1).

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

## Verified CPU smoke result (2026-07-10)

The CPU smoke kernel
[`oczy-cortex-cpu-smoke` v4](https://www.kaggle.com/code/abdellahkadem/oczy-cortex-cpu-smoke)
completed remotely on a Kaggle x86_64 CPU. The report recorded:

- 64×64 fast and slow state shapes;
- a 4×896 latent bank;
- 274,563 trainable smoke-path parameters and 3,473,792 frozen parameters;
- finite losses and gradients;
- held-out improvement of 42.11% (0.4211);
- elapsed: 2.4626 s; throughput: 1,039.6 episodes/s;
- identical before/after frozen-state hash
  `8be7ef2addbe83ce0766a360f9417d9c37a4baf43f97826e0a4faff79048d331`;
- `passed: true`;
- `cuda_available: false`;
- runner SHA `e262b17c8c0918e6464a79a889230bf68d8ed0083d578287cba625f1e0050c8c`;
- torch 2.10.0+cpu.

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
task confirmed the same model hashes on CPU (v1, 2026-07-10).

## Verified qwen-cpu-probe result (2026-07-10)

The Qwen CPU probe kernel
[`oczy-qwen-cpu-probe` v1](https://www.kaggle.com/code/abdellahkadem/oczy-qwen-cpu-probe)
completed remotely on a Kaggle x86_64 CPU. The report recorded:

- model: Qwen2.5-0.5B-Instruct, 494,032,768 parameters, float32;
- zero trainable model parameters;
- torch 2.10.0+cpu; transformers 5.0.0;
- model load time: 2.3473 s;
- frozen forward/backward: 0.7218 s;
- input-embedding gradient: present, finite, norm 357.6543;
- model-parameter gradients: absent;
- parameter fingerprint unchanged
  `0ff56033b6a93d267b868cd98a2990b61a958af56aa9268a88974102691d9f5d`;
- model hashes match reference (`config.json` and `model.safetensors` unchanged);
- `passed: true`;
- `cuda_available: false`;
- runner SHA `3c8cddceea2d5ddd5e083db8e42af0d83bfffcf68d9d0363596c8c241d715fe2`.

## Verified generated bootstrap probe result (2026-07-10)

The generated research-bootstrap kernel
[`oczy-cpu-bootstrap-probe` v4](https://www.kaggle.com/code/abdellahkadem/oczy-cpu-bootstrap-probe)
completed remotely on a Kaggle x86_64 CPU. This kernel exercises the full
`prepare_source_bundle.py` &#x2192; `prepare_research_kernel.py` pipeline: a
commit-addressed private source dataset, opaque archive extraction under
`/tmp/...` (not `/kaggle/working`), pinned Qwen model attachment, and the
generated `run.py` bootstrap with hardware and environment provenance checks.

Key evidence:

- clean committed source `9dfa484dc5ea0d48f06673ad27a6b64678ce7619`;
- source dataset `abdellahkadem/oczy-source-9dfa484dc5ea`;
- opaque archive `source.tar.gz.bin`, SHA-256
  `5a0d2990473ac35cf373be4af61f7c4066f113f73c5fc0874430cfd2ae7d77b1`,
  dirty=false;
- source extracted under a temporary directory (`/tmp/...`), not under
  `/kaggle/working`;
- pulled output contains exactly three files:
  `qwen_model_probe.json`, `remote_run_provenance.json`, and the kernel log;
- Qwen CPU probe within the bootstrap passed: float32 forward, finite
  input-embedding gradient norm 357.657470703125, no model-parameter gradients,
  parameter fingerprint unchanged;
- `passed: true`;
- `cuda_available: false`, `cuda_device_count: 0`;
- `CUDA_VISIBLE_DEVICES=""` verified before torch import;
- CPU profile with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `enable_gpu: false`, `enable_tpu: false`, `enable_internet: false`.

**Failure history and regression tests.** v1 failed with a raw JSON null
artifact (bootstrap serialization issue). v2 failed when Kaggle auto-extracted
the source tarball, bypassing the private-dataset extraction path. Both failures
are diagnosed, fixed, and covered by regression tests in the infrastructure
test suite. v3 ran successfully but left the extracted source in the pulled
output; v4 produces the compact three-file output documented above.

The generated bootstrap probes the same pinned Qwen2.5-0.5B-Instruct model
(hashes match the reference in the pinned model source section) and the same
CPU-only contract as the `cpu-smoke` and `qwen-cpu-probe` tasks. It is the
infrastructure proof that the full generated-job pipeline — source bundling,
opaque archiving, kernel generation, extraction, model attachment, bootstrap
execution, and provenance logging — works end to end on a remote CPU. It is
not evidence for the cortex hypothesis; it is a verified compute substrate.

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
