# Remediation Audit — What Could Have Been Done Better

**Date:** 2026-07-01
**Session:** `f645e4af` (claude-fable-5 review of repo experiments)
**Prompt:** "check this repo experiments and all and tell me who could have been
done better to achieve its objective"
**Objective (from `experiments.txt`):** an agent where *memory becomes changed
dynamics, not retrieved content* — corrections metabolize into fast weights,
consolidate into slow weights, raw traces are forgotten. North-star metric:
`behavior_delta_per_byte` on held-out probes.

## TL;DR

Twelve days of work produced genuinely good infrastructure and some honest
science, but the objective was not achieved — and the biggest reasons are
strategic, not technical: the wrong inference substrate was chosen and never
abandoned, the eval was co-evolved with the system it measured, and the working
mechanisms that emerged (text prefix + logit bias + scope-slot retrieval) are
*retrieval with extra steps* — the exact thing the thesis defines itself
against. This audit drove the five-sprint remediation plan in `SPRINT.md`.

## Finding 1 — The substrate choice sabotaged Goals 1 and 2 from day one

The thesis requires two things from the LM: writing into the KV cache (Goal 1)
and reading mid-layer hidden states (Goal 2). `llama-cpp-python` provides
neither — `GOALS.md` admits Goal 1 is "blocked on the installed binding," and
`peek_embedding()` can only see the final layer. The project then spent days
engineering around its own inference engine: residual cvecs as a KV-slot
substitute (which the 06-25/06-27 experiments proved *cannot* carry content,
only posture), 348 KB `llama_state_seq_get_data` snapshots per 15-token fact,
and text prefixes burned as tokens on every forward pass.

**Better:** run the same 1.2B-class model through HuggingFace/PyTorch, where
`past_key_values` is a plain tensor you can write and `output_hidden_states=True`
gives every layer for free. The repo even contained `bench_hf_cpu.py` — the
escape hatch existed and wasn't taken. Choosing a hybrid conv+attention model
(LFM2.5) made it worse: even KV snapshots include conv1d recurrence state. For
research about intervening in model internals, pick the most transparent
substrate, not the fastest one.

## Finding 2 — Breadth-first organ building before any loop closed

Five organs (hippocampus, critic, identity, immune, autoencoder) plus a DSI
index, an ingestion pipeline, and a Pi tool-use benchmark were all built while
the core metabolism loop — correction → cortex drift → changed LM behavior —
was still unproven. The code audit shows most organs never grew past heuristics:

- **SkillImmuneCortex**: pure keyword substring matching, zero learned
  parameters; the `Skill` compilation path is never invoked.
- **WorldModelCritic**: 3 of 4 logistic weights hardcoded, hand-coded
  ambiguous-word list, MLP head initialized but never trained.
- **IdentityHypernetwork**: hardcoded 14-token vocab that scored 0.001–0.006 for
  over a week, partly because its own eval probe compared incomparable strings.
- **ExperienceAutoencoder**: an "autoencoder" with no decoder; bag-of-words in,
  random projection out (until late Hebbian patches).

`EVALUATION.md` (June 19) already showed which organs were dead weight. The
thesis's own words — "the missing art is the metabolism" — pointed at closing
**one** minimal loop (cortex + hippocampus, one LM, one behavioral metric)
before adding anything. Instead effort spread across seven components whose
outputs mostly feed a *string-label reranker* (`organism.py:_rank_answer`), not
the cortex. Goal 3 ("organs consume tensors") was still unmet on July 1.

## Finding 3 — The eval was co-evolved with the system it measured

The most consequential methodological problem, visible directly in the git log:

- **Metrics were added until thresholds passed.** Lane 01's acceptance criterion
  was literally "desaturation_count ≥ 3" — a count of how many sub-metrics show
  spread — and the fix was "added 3 NEW derived sub-metrics… lifted 1.0 → 4.0.
  MET." That's improving the ruler, recorded as progress.
- **Thresholds were hit by tricks that defeat the thesis.** Lane 06's byte
  target was met by the A0b "seed-regenerable" autoencoder — dropping 236 KB →
  6.6 KB by regenerating a *random* matrix from a seed, which by construction
  means learned updates to it can't persist. The compression metric passed; the
  thesis (compress *learned* experience) was abandoned in that same move.
- **Baselines were designed to lose.** Lane 07's headline gap of 1.0 came from a
  lexical baseline that "stops at 0 by construction," with AUC=1.0 on an
  8-example corpus for both feature sets.
- **Test-set leakage in the Stage 5 "fix."** The `_SCOPE_TEACHING` commit adds
  teaching utterances keyed to the specific failing episode IDs, and — worse —
  "scope probes in failing episodes get `prefix_targets=[probe.expected]`," i.e.
  the expected answer is fed into the logit-bias path for exactly the probes that
  were failing. Stage 5 going 0.00 → 1.00 this way is not learning; it's the
  harness handing the agent the answer key.
- **The "13.5x breakthrough" (`044cb51`) was unproven as a capability gain.**
  Four things changed at once with no ablation, endpoint-only reporting (no
  K-trajectory), no specificity check, and a mechanism where the metric is
  directly coupled to vector magnitude — accumulating `warm_state` makes cvecs
  *louder*, which raises logits whether or not the direction is semantically
  right. It may be real, but the evidence couldn't distinguish steering from
  shouting. (Adjudicated and retracted in S2.4 — see
  `2026-07-01_s2_4_breakthrough_ablation.md`.)

**Better:** freeze and version the eval; let the autoresearch loop optimize the
system but never the metrics, thresholds, or per-episode data paths; require any
"fix" to be episode-ID-agnostic; require single-variable ablations plus a
K-trajectory before the word "breakthrough" enters a commit message.

## Finding 4 — Basic experimental hygiene was consistently missing

- **Silent integration failures invalidated whole weeks.** The scope-slot
  reranker was broken by three compounding bugs (last-token pooling collapsing
  all embeddings to cosine 1.0, a 16-slot cap, a 0.85 retrieval threshold that
  real embeddings at ~0.3–0.65 could never reach) from June 26–30. Every
  curriculum number in that window was invalid, and follow-on experiments — the
  1M-parameter expansion "null result," the June 29 reranker A/B — were run *on
  top of the broken component* and mis-concluded. One hour of empirically
  checking the cosine-similarity distribution of real curriculum embeddings
  would have caught it; so would one end-to-end behavioral assertion ("teach X,
  probe X, assert the reranker fired").
- **No statistics anywhere.** Single seed, single run, point estimates on stages
  with 6–10 episodes, where one flipped episode moves a metric ±17%. Claims like
  "SSI 0.5 → 0.625" are one episode.
- **No held-out or external eval until the final day.** The same 44-episode
  curriculum served as both training and test for 12 days. When the external QA
  benchmark finally ran (July 1), the cross-domain split came out *worse* than
  vanilla — precisely the overfitting signature you'd expect.
- **Post-hoc metric shopping.** Lane 03's hypothesis was refuted, then
  mean-pool → last-token → max-pool variants were tried until the number
  improved (+0.035, still under threshold). The refutation was honestly kept,
  but the pattern is p-hacking-shaped; pre-register the fallback analysis.

## Finding 5 — The architecture inversion was diagnosed but never cured

`GOALS.md` opens by saying the codebase is "inverted: the LM is load-bearing on
the answer path while the cortex writes label strings." That was still true on
July 1. In `OrganismAgent`, the answer is a top-scoring label string picked by a
reranker; in `CortexAgent`, the mechanisms that actually recall facts are the
**text prefix** and **post-forward logit bias** — retrieval-shaped mechanisms —
while the cortex's cvec contributes framing only, and the "metabolism" metrics
measure a state vector whose drift the answer path barely consumes. The system
that works is a small retrieval/rerank stack; the system the thesis describes
remains mostly instrumentation around it. Deciding — explicitly, mid-project —
either "we accept retrieval and measure it honestly" or "we refuse retrieval and
force the cortex to carry content" never happened.

## What was done well (worth keeping)

For calibration: the pre-registered lane specs with falsifiable criteria, the
honest retention of Lane 03's refutation, the June 21 review that caught the
metric-direction bug (Oracle ranked below a real agent) and the hippocampus's
self-referential 1.000 score, the cvec-ceiling investigation (June 25–27)
culminating in the genuinely valuable finding that residual cvecs cannot carry
content on a 1.2B model while logit bias reaches rank-1 — these are real
research practice. The `research/README.md`'s "the eval went blind" framing
shows the right instincts; the failure was that the autoresearch loop's
incentives then overrode those instincts.

## The five changes, compressed

1. **Switch to a PyTorch/HF substrate** — dissolves Goals 1 & 2, which gated
   everything.
2. **Close one loop before building seven organs** — cortex + one memory, one
   behavioral metric, end to end.
3. **Separate eval from search** — frozen, versioned eval; the optimizing loop
   may never touch metrics, thresholds, baselines, or episode-conditioned code.
4. **Statistics and held-out data from day 1** — multiple seeds, CIs, external
   probes, a vanilla-LM baseline; make `behavior_delta_per_byte` on *held-out*
   probes the single headline number.
5. **Magnitude-controlled steering claims** — every drift/logit result must
   report specificity (non-target logits) and be robust to normalizing the
   steering-vector norm, or it's measuring loudness, not learning.

## Outcome

This audit was converted into `SPRINT.md` (five one-week sprints with checkbox
tasks and definitions of done) and executed in the same session: Sprint 0 (eval
integrity freeze) and Sprint 1 (substrate migration to Qwen2.5-0.5B + HFDriver)
are complete; Sprint 2's S2.3/S2.4 (the magnitude-controlled drift metric and
the 13.5x re-adjudication) are done — the 13.5x claim is formally retracted.
Three of the four headline claims the repo carried into July (Stage-2 scope,
mid-layer hiddens, 13.5x drift) fell under pre-registered, leak-free conditions.
See `SPRINT.md` for current status and the per-sprint logs for evidence.
