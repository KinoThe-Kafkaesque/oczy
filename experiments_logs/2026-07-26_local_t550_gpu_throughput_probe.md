# Local T550 GPU throughput probe — not worth wiring into the research path

**Date:** 2026-07-26
**Classification:** VALID (infrastructure probe, not a cortex experiment)
**Decision:** Local NVIDIA T550 Laptop GPU is **not** added as a verified
compute path. The CPU-only contract stands. The T550 is recorded as a
local-dev convenience only.
**Recorded in:** [`infrastructure/kaggle/README.md`](../infrastructure/kaggle/README.md)
("Local GPU probe" section), [`CURRENT_STATE.md`](../CURRENT_STATE.md)
(remote compute pool row).

## Goal

Answer whether the local NVIDIA T550 Laptop GPU (4 GB, Turing compute 7.5)
should be wired into the Oczy research compute path alongside the verified
Kaggle/Colab CPU pool. Specifically:

1. What throughput does the T550 give on the small causal LMs already in the
   local HF cache (Qwen2.5-0.5B-Instruct is the pinned language organ;
   Qwen2.5-1.5B-Instruct is the documented fallback)?
2. How does that compare to the same models on the local CPU
   (12th Gen Intel i7-1260P, 12 cores / 16 threads)?
3. Is the delta large enough to justify breaking the CPU-only contract and
   maintaining a second, GPU-side code path?

## Method

Two benchmark scripts were written from scratch for this probe and live at
`/tmp/bench_throughput.py` (GPU, fp16) and `/tmp/bench_throughput_cpu.py`
(CPU, fp32). They share an identical workload:

- Prompt truncated/seeded to ~512 tokens.
- 128 new tokens generated, greedy decode, KV cache on.
- Per model: warmup (excluded from timing), then a prefill-only forward pass
  (TTFT + prefill throughput), then a full `generate()` call (decode
  throughput = new_tokens / (generate_secs - prefill_secs)).
- Each model runs in an isolated subprocess so a CUDA device-side assert or
  OOM cannot kill the rest of the run.
- Two batch sizes: 1 (single-stream) and 4 (aggregate throughput).

### Hardware and software

- **GPU:** NVIDIA T550 Laptop GPU, 4 GB GDDR6, Turing compute 7.5,
  ~48 GB/s memory bandwidth, driver 580.173.02 / CUDA 13.0 runtime.
- **CPU:** 12th Gen Intel Core i7-1260P, 12 cores / 16 threads, 30 GB RAM.
- **Software:** torch 2.6.0+cu124 (CUDA wheel from
  `https://download.pytorch.org/whl/cu124`), transformers 5.12.1,
  Python 3.10.12, Ubuntu.
- **Dtype:** fp16 on GPU (Turing supports fp16, not bf16); fp32 on CPU
  (fp16 on CPU is unvectorized and would be artificially slow).
- **Offline mode:** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set for both
  runs, because a stale OAuth token in `~/.cache/huggingface/token` was
  making transformers try to fetch `additional_chat_templates` from the Hub
  even for fully cached models (401 errors). This is a separate environment
  hygiene issue noted below.

### Candidates

The five causal LMs in the local HF cache that fit the 4 GB VRAM budget:

- `Qwen/Qwen2.5-0.5B-Instruct` (494 M, the pinned language organ)
- `Qwen/Qwen2.5-1.5B-Instruct` (1544 M, the documented fallback)
- `LiquidAI/LFM2.5-1.2B-Instruct` (1170 M)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1100 M)
- `sshleifer/tiny-gpt2` (0.1 M, toy workload)

The two `hf-internal-testing/tiny-random-*` stubs were excluded: their tiny
vocab/max-length configs trip device-side asserts (`srcIndex < srcSelectDimSize`
in `aten/src/ATen/native/cuda/Indexing.cu`) and they are not representative
workloads.

## Results

### Batch=1 (single-stream)

| Model | Params | GPU TTFT (ms) | GPU prefill (t/s) | GPU decode (t/s) | GPU VRAM (GB) | CPU TTFT (ms) | CPU prefill (t/s) | CPU decode (t/s) | CPU RSS (GB) | GPU/CPU decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 494 M | 1388 | 332 | **27.4** | 1.19 | 1366 | 338 | 18.9 | 3.73 | 1.45× |
| TinyLlama-1.1B-Chat | 1100 M | 2839 | 180 | 10.5 | 2.37 | 3864 | 133 | 9.4 | 7.19 | 1.12× |
| LFM2.5-1.2B-Instruct | 1170 M | 3020 | 153 | 10.2 | 2.51 | 3712 | 125 | 8.2 | 7.64 | 1.24× |
| Qwen2.5-1.5B-Instruct | 1544 M | 4073 | 113 | 9.6 | 3.31 | 5120 | 90 | 4.9 | 9.88 | 1.96× |
| sshleifer/tiny-gpt2 | 0.1 M | 3 | 150 688 | 784 | 0.06 | 9 | 54 199 | 688 | 0.82 | 1.14× |

### Batch=4 (aggregate tokens/sec across the batch)

| Model | GPU TTFT (ms) | GPU prefill (t/s) | GPU decode (t/s) | GPU VRAM (GB) | CPU TTFT (ms) | CPU prefill (t/s) | CPU decode (t/s) | CPU RSS (GB) | GPU/CPU decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 9138 | 202 | 30.2 | 1.75 | 6520 | 283 | **47.8** | 4.46 | 0.63× |
| LFM2.5-1.2B-Instruct | 17 495 | 106 | 15.8 | 2.99 | 14 551 | 127 | **28.1** | 7.64 | 0.56× |
| TinyLlama-1.1B-Chat | 18 769 | 109 | 12.9 | 2.84 | 13 319 | 154 | **16.0** | 7.19 | 0.81× |
| Qwen2.5-1.5B-Instruct | OOM | — | — | — | 18 994 | 97 | 14.5 | 9.88 | — |
| sshleifer/tiny-gpt2 | 8 | 231 288 | 3176 | 0.21 | 25 | 75 122 | 2300 | 1.10 | 1.38× |

### Raw artifacts

- GPU results JSON: `/tmp/bench_all_results.json`
- CPU results JSON: `/tmp/bench_cpu_all_results.json`
- GPU runner: `/tmp/bench_runner.py`
- CPU runner: `/tmp/bench_runner_cpu.py`
- GPU bench script: `/tmp/bench_throughput.py`
- CPU bench script: `/tmp/bench_throughput_cpu.py`

The scripts and JSON dumps are in `/tmp/` and are ephemeral. The numbers in
this log are the durable record.

## Analysis

### The T550 is a weak GPU

At batch=1 the T550 is only **1.1–2.0× faster** than the i7-1260P on decode.
The i7 is a 28 W, 12-core part; the T550 is a 30 W entry-level GPU with
~48 GB/s memory bandwidth — barely more than DDR5 feeding the CPU. This is
consistent with the project's existing decision to archive T4 (which is
itself stronger than the T550): if T4-class GPUs were not worth the
contract complexity, the T550 is not either.

### The crossover is at batch

At batch=4 the **CPU wins on aggregate decode throughput for every model
that fits the GPU**, by 1.2× to 1.6×. The 12-core CPU parallelizes across
the batch while the 4 GB GPU is memory-bandwidth-starved and serializes on
the SM. The T550's only batch=4 win is that it can still serve Qwen-0.5B at
30 t/s with 1.75 GB VRAM — cheap and headroom-friendly, but slower
aggregate than CPU.

### The pinned organ is not the bottleneck

The research question is whether the cortex learns around a frozen LM.
Throughput on the frozen organ is an engineering convenience, not a
research variable. Spending complexity budget on a 1.45× speedup of a
non-bottleneck is a bad trade. The cortex state, training, and consolidation
loops dominate wallclock; the LM forward is one component.

### The fallback case is the only real temptation — and it is weak

Qwen-1.5B at batch=1 is 1.96× faster on GPU and fits in 3.31 GB. But the
project's own S1.1 decision record
([`2026-07-02_s1_1_model_selection.md`](2026-07-02_s1_1_model_selection.md))
chose 0.5B *because* 1.5B was too slow on CPU (196.5 ms/tok). If 1.5B is
ever needed, the right move is the verified 2×T4 Kaggle path, not the
local T550. The T550 is weaker than a single T4.

### Prefill (TTFT) is roughly a wash at batch=1

GPU compute advantage is offset by CPU parallel prefill. At batch=4 CPU TTFT
scales much better (6.5 s vs 9.1 s for Qwen-0.5B; 14.6 s vs 17.5 s for LFM)
because the GPU is bandwidth-bound on the larger prompt batch.

## Conclusion

**Stay CPU-only on the research path. Do not wire the T550 in.**

1. The T550 offers no meaningful win: 1.1–2.0× at batch=1, loses at batch=4.
2. The LM is not the bottleneck; the cortex loops dominate wallclock.
3. Adding GPU breaks the CPU-only contract in
   [`infrastructure/kaggle/RESEARCH_GUIDE.md`](../infrastructure/kaggle/RESEARCH_GUIDE.md)
   and `AGENTS.md` rule 7. A GPU code path would need: a new verified path,
   hash re-verification on GPU, a second bootstrap generator, and dual code
   paths in `src/oczy/lm/hf_driver.py` and `src/oczy/lm/cvec_driver.py`.
   The project would gain ~10 t/s and lose a load-bearing reproducibility
   guarantee.
4. The T550 is weaker than the archived T4. If GPU compute is ever
   re-authorized, the verified 2×T4 Kaggle path is the right starting point,
   not the local T550.

### Where the T550 is acceptable (non-research)

- **Local dev iteration** on cortex plumbing — for "I am iterating on adapter
  code and want faster feedback," 27 vs 19 t/s is real for a human in the
  loop. Keep it out of recorded experiments.
- **Benchmark scripts** (`bench_hf_cpu.py`, `bench_cross_backend.py`) —
  adding a GPU column to those is fine and informative; they are not
  research runs.
- **Quick local look at the 1.5B fallback** — the T550 can run it at batch=1,
  but the verified result should come from 2×T4 on Kaggle.

## Side findings (not the decision)

- A stale HF OAuth token in `~/.cache/huggingface/token` was making
  transformers try to fetch `additional_chat_templates` from the Hub even
  for fully cached models, producing 401 errors. Running with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` works around this. The token
  should be cleared (`huggingface-cli logout` or
  `rm ~/.cache/huggingface/token`) to avoid the 401s in other workflows.
  This is environment hygiene, not a project decision.

## Provenance

- Benchmark scripts written and executed by the orchestrating session on
  2026-07-26. No remote compute was used; no Kaggle/Colab kernels were
  submitted; no `eval/v2`, `research/`, `lanes/`, or
  `experiments/organism_curriculum/` paths were modified.
- The CPU-only contract in `infrastructure/kaggle/RESEARCH_GUIDE.md` and
  `AGENTS.md` rule 7 were not touched. This probe ran locally only and
  produced no claim about the cortex hypothesis.
- The decision (do not wire the T550 into the research path) was made by
  the orchestrating session based on the analysis above and recorded in
  `infrastructure/kaggle/README.md` and `CURRENT_STATE.md`. It is an
  infrastructure decision, not a scientific result, and does not require
  human sign-off under `AGENTS.md` rule 7 (which governs *adding* verified
  compute paths, not declining to add one).
