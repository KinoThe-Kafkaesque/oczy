# R18/R19 de-block proposal — frontier-teacher co-learning (HUMAN SIGN-OFF REQUEST)

**Date:** 2026-07-26
**Filed by:** autonomous agent session
**Teacher decision (2026-07-26, human):** DeepSeek via **OpenRouter** —
`deepseek/deepseek-v4-flash-0731`, provider pinned to **DeepSeek** (no
cross-provider fallback). Details and implications in §4a.
**Status:** **PROPOSAL — no authorization is claimed.** This document does
NOT modify the pre-registered specs `research/18-consolidation-as-distillation.md`
or `research/19-lm-as-language-organ.md`, does NOT modify eval/v2, thresholds,
or gates, and does NOT authorize any run. Per the project's standing
agreements, changing a pre-registered approach requires explicit human
decision. This file exists so that decision can be made on one page.

## 1. Problem

R18 (consolidation-as-distillation) is BLOCKED at the teacher validity gate:
every seed measured `teacher_dev_delta=0.17647058823529413 < 0.2`. Mechanism
diagnosis (commit `33169cc`, 2026-07-12) ruled out prompt-contract defects:
vanilla=0, raw_prefix=0.1765, chat_template=0 — none reach the gate. Teacher
ceiling n=17. The diagnosis is **teacher expressivity / prompt-task ceiling**:
the teacher is the frozen organ itself (Qwen2.5-0.5B-Instruct) with a
per-fact prefix, and 0.5B cannot express the target behavior strongly enough
to teach. Identical reruns are retired. R19's DEV calibration is underway but
R19's signed evaluation remains gated on human approval, and R18's blocked
verdict leaves the "consolidation" leg of the thesis (experience → ...
→ compression → slow change) without an executable line.

## 2. Why now

Karten et al., "Continual Harness" (arXiv:2605.09998, May 2026) report a
reset-free **model + harness co-learning loop** that drives sustained
milestone progress in open-source models *despite* the same weakness we hit:
"the open-source models we evaluated (up to 31B) are not yet capable enough
to act as both teacher and trainee." Their fix is structural, not model-size:

1. **Warm-up:** SFT the student on frontier Continual-Harness trajectories,
   then an **offline GRPO** pass on a per-step process reward. Neither
   produces meaningful gains alone; both are prerequisites.
2. **Online loop:** K-step **DAgger** rollouts of the student **inside the
   live-refining harness**, reset-free (iteration k+1 starts from iteration
   k's saved state).
3. **Scoring:** a **pairwise process reward model (PRM)** scores transitions
   on a sliding window.
4. **Teacher relabel:** a **frontier teacher** (their config: Gemini-3.1-pro)
   relabels only the **low-reward windows**.
5. **Update:** **soft SFT** on the relabeled shard → θ_{k+1}.

Key point: the teacher's only job is relabeling weak windows — it never has
to express the whole task from a 0.5B prefix. This is precisely the role the
R18 gate asked the 0.5B organ to fill and it failed.

## 3. Proposed Oczy adaptation

| Paper stage | Oczy counterpart |
|---|---|
| Warm-up SFT on frontier trajectories | Frontier teacher (e.g., Gemini/Claude/Qwen-Max API) generates correction-treatment trajectories for eval/v2 dev facts |
| Offline GRPO on per-step PRM | Pre-training the process reward before the online loop (offline, CPU/2×T4 per `infrastructure/kaggle/RESEARCH_GUIDE.md`) |
| Pairwise PRM | Candidate in-house PRM: `WorldModelCritic` from R07 (acceptance head + TD(0) value head); must beat string-feature baseline (`critic_auc_delta` was 0.0 — flag) |
| Frontier-teacher relabel of low-R windows | **DECIDED 2026-07-26:** OpenRouter `deepseek/deepseek-v4-flash-0731`, provider = DeepSeek, relabels only low-PRM windows of the student's own rollouts |
| DAgger + soft SFT, reset-free | Student = frozen organ + LoRA (R18's registered student shape, rank ≤ 8); persistence across iterations |
| Live-refining harness | R23.5 text-level harness serialization bundle (see `research/23.5-...` addendum) |

Strictly one variable at a time: the FIRST change is **only** the teacher
(0.5B-prefix → frontier API) with R18's unchanged student, metric, and gate.
Only if that passes the gate do we add the PRM/DAgger/warm-up stages.

## 4. What must be human-decided

- [x] **Teacher access & model choice — DECIDED 2026-07-26:** OpenRouter,
  `deepseek/deepseek-v4-flash-0731`, provider pinned to DeepSeek (see §4a).
- [x] **Teacher spend ceiling + key provisioning — CLOSED 2026-08-06:** key =
  Prime Agent's `~/.prime/agent/auth.json` `openrouter.key` (env/untracked,
  never committed); hard spend ceiling = **USD $5** (see §4b estimate; ~25x
  above aggressive worst case).
- [x] **Spec amendment — SIGNED 2026-08-06:** registered as Amendment A1 in
  `research/18-consolidation-as-distillation.md` (dev-gate teacher
  substitution only; student/metric/split/gate unchanged; eval/v2 intact).
- [ ] **Spec amendment scope:** allow a registered amendment to R18 (teacher
  substitution + optional warm-up) and/or R19, with the same eval/v2 and the
  same `teacher_dev_delta ≥ 0.2` gate. Specs are otherwise frozen.
- [ ] **Remote-compute scope:** the online co-learning loop involves a
  frontier API (network) interacting with local rollouts; confirm this fits
  the approved CPU/2×T4 remote-compute policy, or scope it to a local
  workstation run.
- [ ] **Capability-floor caveat (accept or reject):** even with a frontier
  teacher, the 0.5B student may sit below the expression floor, making the
  distilled behavior weak (R18 final DEV students scored ~0–0.12). If so,
  this line may need a stronger organ (see S4.3) — approve the sequencing.

## 4a. Teacher decision (2026-07-26, human) — recorded

**Approved choice:** OpenRouter, model `deepseek/deepseek-v4-flash-0731`,
provider pinned to **DeepSeek** (no auto-fallback to other providers).

Operational implications to honor at wiring time:

- **Endpoint/client:** OpenRouter exposes an OpenAI-compatible
  `https://openrouter.ai/api/v1` API; the repo's `benchmarks/pi/proxy_server.py`
  pattern already consumes OpenAI-compatible chat completions, so the teacher
  adapter should follow that shape.
- **Provider pinning:** in the OpenRouter request body, pin routing to
  DeepSeek only, e.g. `"provider": {"only": ["DeepSeek"]}` (verify the exact
  provider-name string at wiring time); do **not** use order-with-fallbacks,
  because cross-provider fallback would make teacher responses
  non-reproducible across runs — the same reproducibility requirement as
  `infrastructure/kaggle/model_manifests/` version pinning.
- **Key handling:** the `OPENROUTER_API_KEY` must live in environment /
  untracked local config (repo convention: `.env` is absent; nothing secret is
  committed — `.devin/config.local.json` is permissions-only). Cost goes to
  the OpenRouter account.
- **Teacher validity gate is invariant to teacher identity:** `teacher_dev_delta`
  is dev accuracy of the teacher on the **same** stage-0 dev facts as R18
  measured (`teacher_dev_delta ≥ 0.2`). The frontier teacher does not lower the
  bar; it just changes *who* must clear it. Excellent property — the recorded
  baseline stays comparable.
- **Cheap first check before the full loop — DONE (2026-08-06):** the teacher
  substitution ran on R18's stage-0 dev split via
  `scripts/teacher_validity_check.py`: **`teacher_dev_delta = 0.5294`** (17
  dev probes; with-correction 9/17, without 0/17) vs the registered 0.2 gate
  and the failed 0.5B baseline (0.1765) — **gate cleared**, ~1,675 tokens.
  Full record: `experiments_logs/2026-08-06_stage0_openrouter_teacher_gate.md`.
  Three wiring caveats were fixed from that run: (1) DeepSeek V4-Flash's
  proprietary reasoning pass must be suppressed (`reasoning.enabled=false`)
  or `content` comes back empty; (2) empty answers must count as misses in
  the check (the eval's substring fallback would otherwise silently score
  them as hits); (3) teacher output length is **unbound** (`max_tokens`
  omitted => provider default) — the original 32-token cap truncated
  answers and would break later relabeling. If the full R18 gate run later fails, the problem is
  upstream of teacher capability and §4 decisions should be revisited before
  any further spend.
- **Capability note:** DeepSeek-V4-Flash is far above the 0.5B floor, so the
  *teacher* gate-failure mechanism is expected to be resolved. The *student*
  (0.5B + LoRA) floor question is separate and remains — see the
  capability-floor caveat in §4 item 4.
- **Spend ceiling:** teacher relabeling touches only low-PRM windows, so spend
  should be modest; please still set an explicit USD ceiling when approving
  §4 item 1's remaining scope (the model choice is now fixed, the spend
  ceiling is not).

## 4b. Cost estimate (measured, 2026-08-06)

Grounded in live runs: DeepSeek-V4-Flash @ OpenRouter lists $0.09/M in,
$0.18/M out (~$0.10/M blended); a teacher probe request measures ~44+7
tokens (~5.2e-6 USD). Measured 17-probe stage-0 gate check = 1,672 tokens.

| Scope | API calls | Tokens | Cost |
|---|---|---|---|
| R18 gate-only, stage-0, 5 seeds (wired `--teacher openrouter`) | 85 | ~4.3k | ~$0.0005 |
| Gate-only, all 6 eval stages, 5 seeds | 470 | ~24k | ~$0.0025 |
| Warm-up trajectories (40) | 40 | ~68k | ~$0.007 |
| Online co-learning relabeling (60–400 relabels) | 60–400 | ~0.1–0.7M | ~$0.01–0.07 |
| Aggressive full line worst case | — | ~1–2M | ~$0.10–0.20 |

**Recommended hard spend ceiling: $5** — ~25x above the aggressive worst
case, ~10,000x above the immediate scope; even a single 65k-token runaway
completion costs ~$0.012. The real cost of the run is local CPU hours
(student distillation/holdout), not tokens.

## 5. What stays frozen regardless

- eval/v2 (metrics, thresholds, baselines, episodes, scoring) — untouched.
- The `teacher_dev_delta ≥ 0.2` gate as the R18 admission criterion — unchanged.
- Retrieval remains the mandatory baseline in every table; harness/co-learning
  results never get labeled as metabolism (AGENTS.md §5).
- No episode-ID-conditioned code; mechanism-level fixes only (AGENTS.md §2).
- Any approved run follows `infrastructure/kaggle/RESEARCH_GUIDE.md`
  (clean commit-addressed source, version-pinned instrument, provenance).

## 6. If approved

I (or a session you direct) will draft the registered amendment text for R18
and R19 with the exact conditions above, for your review, before any run.

## References

- Karten et al. 2026, arXiv:2605.09998 (source of this proposal).
- `notes/2026-07-26_continual_harness_applicability.md` (full mapping).
- `research/18-consolidation-as-distillation.md`,
  `research/19-lm-as-language-organ.md` (unchanged).
