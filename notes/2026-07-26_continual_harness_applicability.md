# Continual Harness (arXiv:2605.09998) — how it applies to Oczy

**Date:** 2026-07-26 (added by autonomous agent session)
**Source:** Karten, Zhang, Upaa, Feng, Shi, Jin, Li, Vodrahalli — "Continual
Harness: Online Adaptation for Self-Improving Foundation Agents",
arXiv:2605.09998v1, 11 May 2026.
**Status:** analysis note only. No experiment was run; no eval/v2 or approved
threshold was changed. R23.5/R24 remain drafts not authorized for execution.

## One-paragraph summary

A reset-free framework where the acting model also plays "Refiner": every F
steps it reads the recent trajectory, detects failure signatures, and applies
CRUD edits to harness state H = (system prompt, sub-agents, skills, memory).
On Pokémon Red/Emerald this recovers a majority of the gap between a
minimalist baseline and a hand-engineered expert harness, with
capability-dependent gains and a hard capability floor. A co-learning loop
(DAgger rollouts through the live-refining harness, pairwise process reward,
frontier-teacher relabeling, soft SFT) updates an open-source student's
weights without resetting the environment.

## Concept map: paper term → Oczy counterpart

| Paper | Oczy |
|---|---|
| Harness state H = (prompt, sub-agents, skills, memory), CRUD-edited in place | External/text-level memory; the repo's own autoresearch loop + agent harness (rlm prompt notes/skills/subagents/memories, `refine.run()`) |
| Refiner (same model as actor) reads trajectory window, every F steps | A "metabolism" pass over experiment logs / lane METRIC lines (failure signatures) |
| Inner act loop + outer refine loop, reset-free, compounding within one run | Organism act loop + consolidation/metabolism loop; argument against episode-reset autoresearch |
| Co-learning: DAgger + pairwise PRM + frontier teacher relabel + soft SFT | R07 WorldModelCritic = in-house PRM; R18 consolidation-as-distillation; R19 direct gradient training |
| Capability floor (Flash-Lite fails to bootstrap; CH < baseline) | The R18 teacher-ceiling failure (0.5B teacher, dev 0.1765 < 0.2 gate); warning for R20/R21 organ choice |
| Bootstrap frozen vs updating | R23.5 serialize→restore→continue test |
| Oracle-relative skill scoring (path cost vs Dijkstra, Fig. 8) | Oczy oracle comparators; S4.4 dashboard instrumentation |
| Cost-vs-completion Pareto plane (Fig. 6), seed medians + per-seed traces | S4.4 headline dashboard: behavior_delta_per_byte vs cost, capability as third axis |

## The key framing contribution

The paper occupies the **lossy-text** point on Oczy's compression curve
(retrieval/metabolism ↔ lossless/lossy, per the 2026-07-26 thesis reframe):
more compact than raw context / persistent KV, far larger than a compact
latent, and — unlike either — proven to rescue long-horizon behavior at
scale. It is therefore not a competitor to Oczy's bet; it is the strongest
**baseline the compact latent must beat on tokens/bytes at equal behavioral
recovery**, and the cleanest existence proof that the reset-free refinement
loop itself works.

## What it changes in the plan (references only — no execution)

1. **R23.5** (draft): add text-level harness serialization as a Method
   condition (see the addendum in `research/23.5-...`). The latent must beat
   it on size.
2. **R18/R19** (blocked at teacher gate): the co-learning recipe — warm up
   with SFT + offline GRPO on per-step PRM, then online DAgger + soft SFT
   with a **frontier teacher for relabeling only** — is the documented fix
   for a teacher expressivity ceiling. Filed as a human sign-off proposal in
   `notes/2026-07-26_r18-r19_deblock_proposal.md`. Specs themselves are
   untouched.
3. **S4.3** (second model): capability-floor result argues the second model
   should be *stronger*, not another 0.5B.
4. **S4.4** (dashboard): adopt oracle-relative scoring + cost-vs-completion
   Pareto reporting.

## Constraints honored

No eval/v2 modification. No threshold change. No pre-registered spec edited
(R23.5/R24 were still DRAFT). No remote experiment run. Retrieval stays the
mandatory baseline; harness-level results are the bar to clear, never counted
as metabolism.
