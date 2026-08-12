# Research/24 Phase-A v2 improvement screen and fresh confirmation

**Date:** 2026-08-09  
**Protocol:** `r24-tiny-decoder/v2`  
**Authorization:** `conversation-019fd6c5-14ae-712f-b8ce-d622526d84fb-go-ahead-test-all` (Phase A only)  
**Decision:** **DO NOT PROMOTE** a frozen decoder to Phase C.

## Evidence boundary

The old `r24-tiny-decoder/v1` numbers are invalidated as measurements. V1 seeded
after model construction, decoded variable-length right-padded queries from the
padded rather than actual query boundary, omitted the oracle transformer's
padding mask, and trained on conflicting specificity plus undefined contextual
composition rows. Its `7/264`, `10/264`, and apparent Deep-FiLM delta are only
execution-smoke history and did not enter v2 selection.

V2 was a new, human-authorized Phase-A protocol. It seeded construction and
independent RNG streams, used exact query-length buckets, masked oracle text,
fail-closed on rendered-input conflicts, saved and reload-verified weights,
recorded rule-clustered controls, and treated correct-vs-swapped state as the
causal gate. Phase A only qualifies an organ; it does not test the three-event
meta-cortex hypothesis.

## Execution validity

All kernels were private, internet-off, CPU-only and used runtime manifest
`acb09a9765df728b0182a6b7a01c65722556a8576391a072f8294990aa48fae1` (Python 3.12.13, torch 2.10.0+cpu).
Every collected execution report matched its clean source commit, archive hash,
job name, expected runtime manifest, and frozen reload hash.

| Stage | Clean commit | Source archive SHA-256 | Jobs | Result |
|---|---|---|---:|---|
| 22-case one-factor screen | `daa536342c2b698068b0f17f635575c4095d1fcb` | `61c36c521c43920664689b71772633b1bc43ebf07b7c407a010f4460bfc7cb8c` | 22 | 22 COMPLETE, 0 invalid |
| closed 2^3 factorial completion | `b7e5ddaab4439aa3991f9f003db77e68d1f2577c` | `53b557f60298f17b6141b62d063943f6001fe8f270a5c83ece7d54029e0c2668` | 4 | 4 COMPLETE, 0 invalid |
| fresh base-vs-finalist confirmation | `33983c0e19b69b1f0b96fc43e8973e4bcbed5552` | `bf1a122a051f4ec7d2a55bdb079410cae3925c8a6241e641e1386f1545dc5973` | 10 | 10 COMPLETE, 0 invalid |

The screen canary and repeated base job produced bit-identical complete
`artifact.json` content and identical initial/final decoder hashes on the same
Kaggle runtime. User systemd services dispatched at most five jobs concurrently
with `--kaggle-submit-interval 0`; all three stages exited `0/SUCCESS`.

## One-factor screen (tuning catalog only)

Frozen validation: 219 rows clustered in 30 held-out rules, SHA-256
`6626ad31c8577c4b827c47a440df1e5d8538d24433a4f7dba26d58b806fdaabd`.
The table reports exact oracle-state and swapped-state correct counts.

| Case | Oracle | Swapped | Net | Paired oracle-only / swapped-only | Equal-rule macro delta |
|---|---:|---:|---:|---:|---:|
| base Deep FiLM / mean / encoder LR×1 | 24/219 | 17/219 | +7 | 9 / 2 | +0.0333 |
| deep additive | 28/219 | 15/219 | +13 | 18 / 5 | +0.0677 |
| CLS pooling | 26/219 | 11/219 | +15 | 17 / 2 | +0.0740 |
| encoder LR×0.3 | 27/219 | 13/219 | +14 | 15 / 1 | +0.0685 |
| 1500 steps | 27/219 | 19/219 | +8 | — | — |
| warmup 40 | 28/219 | 20/219 | +8 | — | — |
| N=1/family | 30/219 | 30/219 | 0 | — | state-insensitive; rejected |
| d128/L4 | 22/219 | 22/219 | 0 | — | state-insensitive; rejected |

All 22 registered paths ran: mean/CLS/attention/line-attention pooling; nested
N=1/5/20/40 data; none/additive/FiLM/prefix and shallow/deep conditioning;
d64/L2 vs d128/L4 capacity; LR, weight decay, cosine, warmup, encoder LR,
counterfactual loss, and longer-step variants. Data N=40 raised token accuracy
but reduced exact/state-causal performance at the equal update budget; more
capacity, prefix conditioning, high encoder LR, and counterfactual ranking did
not improve the endpoint. Rule-transformation exact accuracy was zero for the
three credible one-factor ingredients; most signal was finite-state, so the
screen did not show broad family behavior.

## Closed factorial and mechanical selection

After the screen, the four missing A×C×L cells were frozen and run together,
where A=deep additive, C=CLS, and L=encoder LR×0.3. Eligibility required at
least base raw exact (24), oracle-minus-swapped ≥5 rows, paired oracle-only >
swapped-only, and equal-rule macro delta ≥0.02. Eligible arms were ranked by
raw oracle exact, then paired net, fewer swapped-only rows, and lower
complexity.

| Missing cell | Oracle | Swapped | Net | Paired oracle-only / swapped-only | Macro delta | Eligible |
|---|---:|---:|---:|---:|---:|---|
| A+C | 22/219 | 10/219 | +12 | 15 / 3 | +0.0609 | no: raw exact |
| A+L | 23/219 | 15/219 | +8 | 11 / 3 | +0.0413 | no: raw exact |
| C+L | 25/219 | 5/219 | +20 | 21 / 1 | +0.1034 | yes |
| A+C+L | 21/219 | 10/219 | +11 | 15 / 4 | +0.0542 | no: raw exact |

The pre-registered rule selected the single-factor **deep additive** arm
(28/219 oracle, 15/219 swapped), not the C+L arm, because exact oracle
performance was the primary rank after causal eligibility. No outcome-adaptive
combination was promoted.

## Fresh five-seed confirmation

Confirmation used unseen catalog seed `25001`, train hash
`eec87f23c895e08dd8fd694ec29d0dc295888660ab466930783f90d2cb57ea96`, and validation hash `0863b8c67af9fc175b19005461f15477c08616d5c35410ed3cd197758d9463ad`.
Each seed had the same 223 validation rows / 30 held-out rules for paired base
and finalist jobs. The five init/batch/dropout/control tuples were frozen before
execution; no seed was replaced.

| Seed | Base oracle | Base swapped | Base delta | Additive oracle | Additive swapped | Additive delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18/223 | 18/223 | 0/223 | 18/223 | 15/223 | 3/223 |
| 1 | 29/223 | 30/223 | -1/223 | 28/223 | 25/223 | 3/223 |
| 2 | 27/223 | 19/223 | 8/223 | 18/223 | 14/223 | 4/223 |
| 3 | 30/223 | 23/223 | 7/223 | 26/223 | 19/223 | 7/223 |
| 4 | 16/223 | 18/223 | -2/223 | 18/223 | 13/223 | 5/223 |
| **aggregate** | **120/1115** | **108/1115** | **12/1115 = 0.01076** | **108/1115** | **86/1115** | **22/1115 = 0.01973** |

Registered confirmation gates:

| Gate | Observed | Result |
|---|---:|---|
| Additive swapped delta positive in 5/5 seeds | 5/5 | pass |
| Mean additive swapped delta ≥0.02 | 0.019731 | **fail** |
| Mean additive oracle exact no worse than paired base | -0.010762 | **fail** |
| Mean additive swapped delta no worse than paired base | +0.008969 | pass |

The failure is not just rounding around the causal threshold: the finalist
lost 12 raw oracle exact answers versus base across the paired seeds. Its
aggregate oracle counts were contextual remap 8/370, finite state 98/345, and
rule transformation 2/400 (base: 6/370, 109/345, 5/400). The observed state
signal is sparse and almost entirely finite-state, not evidence across held-out
families/users.

## Decision and next action

The frozen decision is `do_not_promote`. The deep-additive tuning win did not
replicate as a usable, no-worse oracle organ on the fresh catalog. Phase C was
not run: it is outside this Phase-A authorization, and its required confirmed
organ gate failed. Therefore there is no H-TOY-EXISTENCE accept/refute claim and
no meta-cortex result.

The next truthful action is redesign rather than another adaptive retry:
construct an identifiable exactly-three-example string-transformation catalog,
raise the Phase-A oracle articulation ceiling (especially contextual remap and
rule transformation), pre-register a fresh organ confirmation, and separately
obtain human authorization before any C1–C9 Phase-C execution.

Durable machine-readable closure:
`experiments_logs/2026-08-09_r24_phase_a_v2_confirmation.json`.
Remote campaign roots:

- `~/.local/state/oczy/remote-queue/campaigns/r24-v2-screen-daa536342c2b-a1/`
- `~/.local/state/oczy/remote-queue/campaigns/r24-v2-factorial-b7e5ddaab443-a1/`
- `~/.local/state/oczy/remote-queue/campaigns/r24-v2-confirmation-33983c0e19b6-a1/`
