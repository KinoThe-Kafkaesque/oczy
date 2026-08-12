> **Archived historical evidence.** This file records GPU (T4/P100/L4) and
> CPU verification from 2026-07-09. The active remote profile is CPU-only; see
> [`../RESULTS.md`](../RESULTS.md) for the current acceptance contract. GPU
> kernels, metadata, and scheduling recommendations preserved here are not
> active and must not be resubmitted.

# Kaggle cortex compute verification — 2026-07-09

**Cortex smoke window:** 2026-07-09 21:00–21:09 +01:00 (Africa/Casablanca)

**Result class:** infrastructure verification, not a Research/20 experiment

**Runner schema:** `oczy/kaggle-cortex-smoke/v3`

**Runner SHA-256:** `6b193ee821f40dfac052fc2c7bab4dcf81f6d0d5df688ece4b079e4f6f8536ab`

## Verdict

Kaggle is usable for Oczy's offline cortex work. The authenticated CPU path and
a two-GPU T4 path both completed remotely, produced downloadable JSON
artifacts, kept all frozen parameters bit-identical, and used the same runner
source and workload. At an organ-interface width of 896—the hidden width of
the planned Qwen2.5-0.5B language organ—the 2×T4 run delivered **2.016×** the
CPU throughput for the measured training loop.

This verifies packaging, execution, artifact recovery, gradients through a
frozen differentiable organ, and multi-GPU reduction. It does not load Qwen,
does not run `meta_cortex/v1`, and is not evidence for H-META-CORTEX.
`torch.nn.DataParallel` is used only as a dependency-free two-card hardware
proof. Real developmental runs should prefer one independent seed per GPU or
DDP after profiling; the smoke parallelism is not a frozen architecture choice.

## Accepted remote runs

| Kernel/version | Actual hardware | Parallel mode | Loop time | Throughput | Held-out improvement | Frozen hash | Result |
|---|---|---:|---:|---:|---:|---|---|
| [`oczy-cortex-cpu-smoke` v3](https://www.kaggle.com/code/abdellahkadem/oczy-cortex-cpu-smoke) | Kaggle x86_64 CPU | single device | 2.1569 s | 1,186.88 episodes/s | 42.11% | unchanged | PASS |
| [`oczy-cortex-t4-smoke` v3](https://www.kaggle.com/code/abdellahkadem/oczy-cortex-t4-smoke) | 2× Tesla T4, 15.6 GB each, sm_75 | `torch.nn.DataParallel` on `cuda:0,1` | 1.0699 s | 2,392.75 episodes/s | 41.02% | unchanged | PASS |

The T4 elapsed time was 49.60% of CPU time. These timings cover only the
40-step workload after device synchronization; Kaggle queue and container
startup time are intentionally excluded.

Both artifacts reported:

- fixed fast and slow state shapes of 64×64;
- a 4×896 latent bank;
- 274,563 trainable smoke-path parameters and 3,473,792 frozen parameters;
- finite losses and gradients;
- identical before/after frozen-state hash
  `8be7ef2addbe83ce0766a360f9417d9c37a4baf43f97826e0a4faff79048d331`;
  and
- `passed: true`.

Pulled artifacts live locally under `reports/kaggle/` and are ignored by Git.
They can be refreshed with the commands in [`README.md`](README.md).

## Frozen Qwen model-source verification

At 21:45 +01:00, private kernel
[`oczy-qwen-language-organ-t4-probe` v1](https://www.kaggle.com/code/abdellahkadem/oczy-qwen-language-organ-t4-probe)
completed with `passed: true` using the version-pinned official source
`qwen-lm/qwen2.5/transformers/0.5b-instruct/1` and no internet access.

| Check | Remote result |
|---|---|
| Actual hardware | 2×Tesla T4 visible; probe executed on CUDA sm_75 |
| Framework | PyTorch 2.10.0+cu128; Transformers 5.0.0 |
| Artifact | 999,604,126 bytes; all files SHA-256 hashed |
| Model | 494,032,768 parameters; FP16; 0 trainable parameters |
| Load time | 3.1408 s |
| Frozen forward/backward | 0.8840 s for 8 tokens |
| Input/latent gradient | present, finite, norm 359.8534 |
| Model parameter gradients | absent |
| Parameter fingerprint | unchanged before/after |

The remote runner SHA-256
`efdb4f71117ccb2b8117543532603c3d91b73d0fc70d0c7d989d491f5fc5641b`
matched the checked-in source. The remote model hashes matched the locally
resolved artifact:

- `config.json`:
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`;
- `model.safetensors`:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

This closes the model-mount and frozen-gradient plumbing gate. It does not
test the learned cortex, latent coupler, task instrument, or behavior.

## Nulls and compatibility findings

| Requested profile | What actually happened | Adjudication |
|---|---|---|
| `NvidiaTeslaP100` | Kaggle allocated a Tesla P100 (sm_60), but the current PyTorch 2.10.0+cu128 image only included sm_70–sm_120 kernels. The first CUDA tensor operation failed with `no kernel image is available for execution on the device`. | BLOCKED by the current base image; do not use for Oczy PyTorch jobs. |
| `NvidiaL4X1` | CLI/server accepted the machine-shape request, but runtime allocated the same incompatible Tesla P100. Pulled metadata still said `NvidiaL4X1` while `enable_gpu` was false. | Requested accelerator is not proof of allocation. Treat L4 as unavailable on this account/path until a runtime artifact says otherwise. |

These nulls are why every production runner must record the actual CUDA device,
compute capability, device count, framework/CUDA versions, and source hash.

## CLI/account state

- Kaggle CLI upgraded from 2.1.2 to **2.2.3** using the existing `uv tool`
  installation.
- OAuth credentials at `~/.kaggle/credentials.json` were present with mode
  `600`; a live authenticated kernels query succeeded. No credential value was
  printed or copied.
- Before the test the account reported 30.00 GPU hours and 20.00 TPU hours.
- After all probes it reported **0.05 GPU hours used / 29.95 remaining** and
  20.00 TPU hours remaining, refreshing 2026-07-11T00:00:00.

## Exact accepted submission commands

```bash
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels push \
  -p infrastructure/kaggle/t4-smoke \
  --accelerator NvidiaTeslaT4 \
  --timeout 900
```

## Scheduling implication

- Keep small 64×64 cortex-only updates, instrument generation, split/leakage
  audits, scoring, and report aggregation on CPU. GPU launch/replication
  overhead can dominate tiny workloads.
- Use the 2×T4 path when the differentiable frozen-organ interface or large
  developmental batches dominate. The width-896 smoke crossed that boundary
  and produced the measured 2.016× loop throughput gain.
- Schedule independent developmental seeds per card first; only introduce DDP
  when a single seed is large enough to require or benefit from synchronized
  multi-GPU training.
- The official Qwen2.5-0.5B model source is now pinned and remotely verified.
  Before real Phase 1/2 work, package a clean committed Oczy source snapshot
  with the prepared generator. The current probes intentionally validate
  plumbing without pretending the absent `meta_cortex` implementation exists.
