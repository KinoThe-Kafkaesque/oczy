# Literature landscape: in-context adaptation serialization

**Date:** 2026-07-26
**Source:** External review of the Oczy research program
(`chat-export-1785143922754.json`, model: Qwen3.8-Max-Preview). The review
performed web searches for successful research in the direction of the
reframed thesis (see `notes/2026-07-26_in-context_serialization_thesis_reframe.md`).
This note organizes the findings by approach, with the relevance and catch for
Oczy noted for each.

## The gap Oczy sits in

Nobody has published a benchmark showing that a compact learned latent (not raw
KV cache, not full context re-injection, not weight updates) can preserve the
behavioral adaptation of in-context learning in an LLM across sessions. The
closest approaches each solve part of the problem but miss at least one axis:

| Question | Answer | Evidence |
|---|---|---|
| Can in-context learning be serialized? | **Yes** | Context distillation (Snell 2022) |
| Can it be done without weight updates? | **Partially** | KV cache persistence, ILCP (unproven in LLMs) |
| Can it be done compactly? | **Yes in telecom, unproven in LLMs** | ILCP (128-byte payload, peer-reviewed in 6G) |
| Can it persist across sessions? | **Yes, but bulky** | Persistent KV cache |
| Can a frozen LLM be controlled via injected latent? | **Architecturally yes, empirically unproven** | ILCP V1 wiring exists, no benchmarks |
| Does retrieval alone solve the product problem? | **Yes** | Memento (top-1 GAIA) |
| Can the adaptation be serialized at *text level* (prompt/skills/memory edits), reset-free, mid-episode? | **Yes — proven at scale** | Continual Harness (Karten et al., arXiv:2605.09998) recovers a majority of a hand-engineered-harness gap on Pokémon Red/Emerald |
| Can it then be compressed into a *compact latent*? | **Unproven — this is the open gap** | Nobody (ILCP wire-in exists, no benchmark) |

R23.5 / R24 (see `research/`) — compress in-context adaptation into a latent,
re-inject into a frozen model, measure behavioral recovery — is exactly the
missing benchmark, with the text-level harness serialization now serving as
the proven lossy-text anchor to beat (added 2026-07-26, see §7).

## 1. Context distillation — the most proven "save button"

**Snell, Klein & Zhong (2022), UC Berkeley.** arXiv:2209.15189.

The method:
1. Give the model `[instructions] + [task-input]` → it produces
   `[scratchpad] + [final answer]`.
2. Fine-tune the **same** model to produce `[final answer]` from `[task-input]`
   alone, without the instructions or scratchpad.
3. The context has been **internalized into weights**. The performance gain
   persists indefinitely.

Results:
- Effectively internalizes 3 types of signal: abstract instructions,
  step-by-step reasoning, and concrete training examples.
- Outperforms direct gradient descent by 9% on SPIDER Text-to-SQL.
- Can iteratively overwrite old instructions with new ones.
- Can internalize more examples than the context window allows by doing it in
  batches.

**Upadhayaya et al. (2024), Georgia Tech.** arXiv:2409.01930. Replicated with
LoRA adapters on OPT models (125M–2.7B):
- Context distillation achieves comparable in-domain accuracy to in-context
  learning and better out-of-domain generalization.
- Works with as few as 2–16 context examples.
- Smaller models benefit disproportionately — CD narrows the gap between small
  and large models.

**Catch for Oczy:** Context distillation updates the model's weights (or LoRA
adapters). It violates the "frozen organ" constraint. The organ is not frozen —
it is being fine-tuned. But it proves the fundamental thesis: in-context
learning can be serialized into persistent state.

## 2. TTT-E2E — the most impressive, but does not persist

**Tandon, Sun et al. (2025/2026), NVIDIA/Stanford.** arXiv:2512.23675.

Test-Time Training End-to-End. The method:
1. Use a standard Transformer with sliding-window attention (not full
   attention).
2. At test time, continue training the model via next-token prediction on the
   context it is reading — compressing context into weights on the fly.
3. At training time, meta-learn the initialization so the model is prepared
   for test-time training.

Results (3B models, 164B tokens):
- Matches full-attention quality across context lengths, while Mamba 2 and
  Gated DeltaNet do not.
- Constant inference latency regardless of context length — 2.7× faster than
  full attention at 128K, 35× faster at 2M.
- No scaling wall observed across extensive experiments.

The NVIDIA blog frames it exactly the way the reframed thesis does:
> "Humans compress a massive amount of experience into their brains, which
> preserves the important information while leaving out many details. For
> language models, we know that training with next-token prediction also
> compresses a massive amount of data into their weights. So what if we just
> continue training the language model at test time?"

**Catch for Oczy:** TTT-E2E's weight updates are ephemeral. They last for the
session but do not persist across sessions. The model resets. This is exactly
the gap the reframe identified: "the context given should survive sessions."
TTT-E2E solves the compression problem but not the persistence problem.

## 3. Persistent KV cache — the direct save, but bulky

Multiple works persist the KV cache across sessions:

- **Shkolnikov (2026).** Persists each agent's KV cache to disk in 4-bit
  quantized format and reloads it directly into the attention layer,
  eliminating redundant recomputation across sessions.
- **LMCache.** Turns KV cache from a temporary state into reusable, persistently
  stored knowledge that can be reused across multiple serving sessions.
- **MemArt.** Stores conversational turns as reusable KV cache blocks and
  retrieves relevant memories by computing attention scores in latent space.
- **Durable Agentic AI Sessions.** Treats the KV cache of an active agent
  session as durable state — neither short-lived like a single prompt nor
  permanently stored like database state.

**Catch for Oczy:** KV cache persistence works, but it is large (proportional
to context length × number of layers × hidden dimension), model-locked (tied
to the specific model's architecture and version), and not compressed (it is
the raw computational state, not a learned summary). It is lossless but
expensive. The Oczy thesis is about whether you can do better: a compact
learned representation that captures the behavioral effect of the context
without storing the full computational state.

## 4. ILCP — the closest to the Oczy architecture

**Banerjee & Awan, Nokia Munich (2026), accepted at AI4NextG @ ICML 2026.**

Inductive Latent Context Persistence. This is almost line-for-line the Oczy
architecture, originally developed for 6G telecom handover and then mapped to
LLM agents:

| 6G handover | LLM agent mapping |
|---|---|
| Source base station's GRU state | Sender LLM's pooled hidden state |
| β-VAE compresses to 32-dim latent | β-VAE compresses 4096-dim pooled vector to latent z |
| 128-byte payload over 3GPP Xn interface | TransportPayload (in-process in V1) |
| Target base station's gated MLP projects latent | Gated MLP projects z into K memory tokens in LM embedding space |
| Concat with new observations | `torch.cat([memory_tokens, question_embeds])` via `inputs_embeds` |

The telecom results are peer-reviewed and strong:
- 0.0% ping-pong handover rate vs 6.5% for the no-transfer baseline.
- Peak accuracy improvement of +13.3 percentage points, average +5.1 pp.
- Runs on a single GTX 1080.

The LLM agent V1 exists as working code (Qwen2.5-7B-Instruct, β-VAE
compressor, gated MLP projector, greedy decode from soft prefix) but has no
published LLM-side benchmarks yet. The author is explicit about this: "No
agent-side numerical receipts in V1... Every numerical claim comes from the 6G
paper."

**This is the gap Oczy could fill.** The architecture is validated in telecom.
The LLM wiring exists. But nobody has measured whether the compressed latent
actually preserves the behavioral adaptation of in-context learning in an LLM,
across sessions, on real tasks.

## 5. Memento — the retrieval answer (not metabolism)

**Zhou et al. (2025), UCL/Huawei.** arXiv:2508.16153.

Memory-augmented MDP with episodic memory + neural case-selection policy. The
LLM is frozen. Past experiences are stored externally. Policy improvement comes
from memory reading (retrieval).

Results: Top-1 on GAIA validation (87.88% Pass@3), outperforms training-based
methods on DeepResearcher. Case-based memory adds 4.7–9.6% absolute on
out-of-distribution tasks.

**Catch for Oczy:** This is retrieval, not metabolism. The LLM does not change.
The memory is external. By the Oczy framework, this is exactly the thing that
"is not the thesis." But it works extremely well. This is the strongest
evidence that the product question ("does the user get correct, durable
behavior?") can be answered affirmatively without solving the scientific
question ("does the system internalize?").

## 6. VISTA — context proprioception (adjacent)

**Xu et al. (2026), CUHK/Tencent.** arXiv:2606.30005.

Not about persistence per se, but about making the LLM aware of its own context
state so it can manage what to keep, archive, and recover. Training-free,
model-agnostic. Lifts Gemini-3-Flash from 22.7% to 50.7% on LOCA-Bench.

Relevant because it addresses the "what to save" question that the Oczy
compressor implicitly answers.

## 7. Continual Harness — the harness-level "save button" (added 2026-07-26)

**Karten et al. (2026), Princeton/ARISE/Google DeepMind.** arXiv:2605.09998.

Online, reset-free self-improvement of the *harness* rather than the weights
or the cache. A "Refiner" (the same model that acts) reads the recent
trajectory every F steps and applies CRUD edits to harness state
H = (system prompt, sub-agents, skills, memory). The paper is the proven
successor to the GPP (Gemini Plays Pokémon) human-in-the-loop harness work.

Why it is the missing middle of Oczy's compression curve:

| | Losslessness | Evidence status |
|---|---|---|
| Raw context / persistent KV | lossless, bulky | works, un-compact |
| **Continual Harness state (p, G, K, M)** | **lossy at text level** | **proven at scale** — recovers a majority of the gap between a minimalist baseline and a hand-engineered expert harness on Pokémon Red/Emerald |
| Compact learned latent (Oczy's bet) | lossy, cheapest | unproven — the open gap |

**Results of direct use to Oczy:**
- **Compounding within one run:** refinement information accumulates
  monotonically inside a single reset-free episode; reset-based methods
  (GEPA-style prompt optimization) restart this accumulation after each
  update. This backs the metabolism-loop design over episode-reset loops.
- **Bootstrap transfer:** a harness refined in a prior run accelerates the
  next even when the task state resets (`bootstrap frozen`), and continued
  refinement adds on top (`bootstrap updating`). Directly relevant to
  R23.5's serialize→restore→continue test and to any "save button" claim.
- **Capability floor:** harness gains are capability-dependent and *vanish
  below a floor* (Flash-Lite underperforms the minimalist baseline). With
  Qwen-0.5B as Oczy's organ, weak harness/latent results may reflect the
  organ floor, not the serialization idea. See S4.3 / R20 organ-ceiling
  notes.
- **Concentrated recurrent refinement:** edits concentrate on a few
  load-bearing components, and prompts cycle growth → simplification rather
  than monotonically growing. Supports research/06 bounded-growth
  consolidation as *simplification*, not accumulation.
- **Oracle-relative measurement (Fig. 8):** refined navigation skills were
  scored by path-cost deficit versus a Dijkstra oracle, independent of
  end-task efficiency — the same instrumentation Oczy keeps in its oracle
  comparators.

**Catch for Oczy:** this is *text-level* serialization, not latent. It does
not beat the frozen-organ/compact-state question — it is the strongest
available *baseline* that Oczy's compact latent must outperform on
tokens/bytes at equal behavioral recovery. It also uses frontier models
(Gemini) as refiner/teacher; the paper explicitly notes open-source models up
to 31B cannot yet serve as both teacher and trainee — matching Oczy's R18
teacher-gate failure with a 0.5B teacher.

Concise analysis mapping the paper onto Oczy: see
`notes/2026-07-26_continual_harness_applicability.md` and the R23.5 addendum.

## Adjacent literature the project had not engaged with

The external review also flagged that the project reinvents several well-studied
ideas without citing or building on them:

| Oczy concept | Adjacent literature |
|---|---|
| Frozen LM + learned external state | Fast Weight Programmers (Schmidhuber 1992; Ba et al. 2016) |
| Latent steering of a frozen model | Representation Engineering (Zou et al. 2023), Activation Steering (Turner et al. 2023) |
| Meta-learned online update | MAML (Finn et al. 2017), In-Context Learning as implicit meta-learning (Xie et al. 2021; Akyürek et al. 2023) |
| Test-time state change without backprop | Test-Time Training (Sun et al. 2020; Gandelsman et al. 2022) |
| Fixed-shape persistent memory | Memory Transformers, Memorizing Transformers (Wu et al. 2022) |
| What frozen transformers can/can't compute | Merrill & Sabharwal (2023) on transformer expressiveness |

This matters not for credit but for **avoiding dead ends that others have
already mapped.** The fast weight programmer literature, in particular,
directly addresses the Oczy thesis: can a separate learned "fast weight" system
modulate a "slow weight" base model? The answer from that literature is *yes,
but the fast weights need to interact multiplicatively with the computation,
not additively through a bottleneck.* If the Oczy coupler is additive, that may
explain the repeated failures of R02, R09, and R19.

## Provenance

This literature review was compiled during the external review on 2026-07-26.
Section 7 (Continual Harness) was added on 2026-07-26 by an autonomous agent
session after reviewing arXiv:2605.09998; the summary should be independently
verified before citation in any pre-registration.
The web searches were performed by Qwen3.8-Max-Preview. The summaries are
paraphrased from the chat transcript at `chat-export-1785143922754.json`. The
arXiv IDs and key claims should be independently verified before being cited
in any pre-registration or publication. This note is a starting point for
literature engagement, not a vetted bibliography.
