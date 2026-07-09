# Prefix-Based Closed-Set Generation for Curriculum Answer Labels

**Status:** Research project  
**Context:** `OrganismAgent` real-LM mode (`use_cortex_lm_answer=True`)  
**Rationale:** The chat-tuned LFM2.5-1.2B-Instruct model naturally produces verbose, sentence-length responses to curriculum requests (e.g., "The ship's log is missing" → "I'll search the system error logs"). Forcing terse label-like output requires closed-set generation constraints.

## Candidate approaches

1. **`prefix_targets` at answer time**
   - Pass the known corrected label as `prefix_targets=[label]` to `CortexAgent.articulate()`.
   - Requires knowing the candidate label before generation, which is the cross-domain lookup problem itself.
   - Could be combined with a top-k candidate sweep: generate with each candidate as prefix and pick the highest-likelihood one.

2. **Logit-bias closed-set forcing**
   - At generation step 1, add bias only to tokens belonging to any learned label token id.
   - Combine with a stop token after a short label phrase.
   - Risk: multi-token labels need sequential subword biasing, as shown in `kv_slot_injection.py` experiments.

3. **Few-shot prompt formatting**
   - Prepend examples: "Answer using only one of: [label list]."
   - Still not guaranteed; the model may ignore the instruction.

## Open questions

- What is the right candidate label set size (`k`) for a top-k sweep?
- How does prefix forcing affect the learned cortex cvec/slot signal when the KV cache starts from an arbitrary forced token?
- Does this interact cleanly with the context-addressed slot store, or does each candidate need its own slot retrieval?

## Recommended trigger

Revisit this project once `scope_selectivity_index` exceeds 0.80 on Stage-2 and the slot-store routing is stable. At that point, the bottleneck will shift from *retrieving the correct sense* to *expressing it in one token/phrase*.

## Related files

- `src/oczy/experiments/organism.py`
- `src/oczy/experiments/cortex_agent.py`
- `src/oczy/experiments/kv_slot_injection.py`
- `src/oczy/experiments/scope_selectivity_stressor.py`
