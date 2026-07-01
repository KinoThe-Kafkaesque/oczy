# S1.1 — HF substrate model selection (Sprint 1)

**Date:** 2026-07-02
**Decision:** `Qwen/Qwen2.5-0.5B-Instruct` (fallback: `Qwen/Qwen2.5-1.5B-Instruct`)
**Recorded in code:** `src/oczy/lm/hf_model_choice.py`

## Method

- KV-cache structure check (`check_kv_cache.py`): load each candidate,
  forward pass, assert plain decoder-only architecture (no conv/SSM hybrid
  layers, no sliding-window-only attention) and a (k, v) tensor pair per
  layer in `past_key_values`.
- CPU benchmark (`bench_hf_cpu.py --models ... --gen-tokens 32 --runs 3`,
  float32, 4 threads, greedy): encode latency, TTFT, ms/token, peak RSS.
- Qualitative sanity via `HFDriver` (3 prompts: word-sense with context,
  fact recall, one-word instruction).

## Results

| Model | Params | KV check | Enc ms | TTFT ms | ms/tok | Peak RSS |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 494M | PASS | 1.7 | 149.5 | **82.8** | 2.67 GB |
| Qwen2.5-1.5B-Instruct | 1.54B | PASS | 3.3 | 488.8 | 196.5 | 6.53 GB |
| TinyLlama-1.1B-Chat | 1.10B | PASS | 0.9 | 425.8 | 437.1 | 4.85 GB |

Notes:
- TinyLlama emitted EOS after 1 token on the plain (non-chat-template)
  benchmark prompt in all runs — a formatting fragility that would fight the
  curriculum's raw-prompt probes; combined with the worst ms/tok, eliminated.
- Qwen2.5-1.5B is within the ≤300 ms/tok requirement but costs 2.4× the
  latency and 2.4× the RSS of 0.5B. Multi-seed curriculum runs (S0.5
  requirement) multiply that cost.
- Qualitative (0.5B via HFDriver, greedy): fact recall correct ("The capital
  of France is" → "Paris. …"), one-word instruction followed ("cold. …"),
  in-context sense adoption partial ("'profile' refers to a specific…") —
  adequate; steering that adoption is precisely what the cortex work is for.

## Decision rationale

Speed dominates: Sprint 1's experiments (S1.3 rank tables, S1.4 per-layer
sweeps) and every future multi-seed curriculum run scale linearly with
ms/token. 0.5B passes every structural requirement at 2.4× the speed and
0.4× the memory of the next option. If Phase-B experiments show
0.5B-specific quality ceilings, `HF_MODEL_FALLBACK_ID` (1.5B, already cached
and verified) is a config-line change.

## Provenance

Partial credit: the S1.1 fanout agent (died at time cap mid-download) wrote
`check_kv_cache.py` and the bench candidate list; benchmark execution and
the decision were completed by the orchestrating session.
