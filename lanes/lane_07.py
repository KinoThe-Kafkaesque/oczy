"""Lane 07: Conversation World-Model RL Phase 0.

marker_free_uptake_gap = uptake(world-model critic) - uptake(token-overlap critic)
on marker-stripped corrections. Source: research/07-conversation-world-model-rl.md.

Baseline: a competitive token-overlap Jaccard NN classifier (not a
constructed-to-fail lexical gate). Each marker-free test correction is flagged
when its max Jaccard similarity against the marker-bearing teaching phrases
exceeds 0.25 -- the same threshold the world-model critic's
``_similar_correction_rate`` uses (critic.py:442-460). With whitespace-lowercase
tokenization, "the sky is blue" overlaps "no actually the sky is blue" with
Jaccard 4/6 >= 0.25, so the baseline CAN fire on marker-stripped corrections.

Lift: the world-model critic carries an online Jaccard-similarity feature
(``_similar_correction_rate``, critic.py:442-460) and a learnable prior-correction
weight. After a teaching pass of marker-bearing corrections for the SAME facts
(record_outcome each), the critic generalizes to the marker-stripped versions via
token overlap. TD(0) value head sees no test-turn updates; the string-logistic
similarity head carries the lift -- the "semantic trace left in the surrounding
tokens" the spec calls out (07-conversation-world-model-rl.md:13). Real-LM
peek_embedding is Goal-2-staging only and unavailable at baseline; the Jaccard
count-based stub is the spec's named "the X is Y" fallback. Config matches
CortexAgent (cortex_agent.py:219-226); untrained MLP path (use_hidden=True) is
bypassed via lm_hidden=None since 0.01-init randn weights saturate near 0.5.

Honesty note: because the baseline now uses the same token-overlap signal the
critic exploits, the gap may collapse to 0.0 or even go negative -- that is the
correct, non-gameable result. ``measure()`` returns a dict reporting the gap
alongside ``lane_07_corpus_n`` (the test corpus size) and the raw uptake rates
for transparency.
"""

from __future__ import annotations

from lanes._common import MARKER_BEARING_CORRECTIONS, lane_measure


def name() -> str:
    return "lane_07_marker_free_uptake_gap"


# 4 marker-stripped corrections: factual content with NO lexical markers
# ("actually,", "wrong, ", "correction:", "no, ").
_MARKER_FREE_CORRECTIONS = (
    "the sky is blue",
    "paris is the capital of france",
    "two plus two is four",
    "jupiter is the largest planet",
)

# Jaccard threshold matching the critic's _similar_correction_rate (critic.py:442-460).
_JACCARD_THRESHOLD = 0.25


def _token_overlap_flag(correction: str, teach_set: list[set[str]]) -> bool:
    """Flag ``correction`` if max Jaccard overlap with any teaching phrase >= 0.25.

    Competitive baseline: the same token-overlap signal the world-model critic
    exploits, so it can fire on marker-stripped corrections (unlike the old
    constructed-to-fail lexical substring gate).
    """
    tokens = set(correction.lower().split())
    if not tokens:
        return False
    for teach_tokens in teach_set:
        if not teach_tokens:
            continue
        jaccard = len(tokens & teach_tokens) / len(tokens | teach_tokens)
        if jaccard >= _JACCARD_THRESHOLD:
            return True
    return False


@lane_measure
def measure() -> dict[str, float]:
    from world_model_critic import WorldModelCritic

    # Tokenize the teaching set once.
    teach_set = [set(p.lower().split()) for p in MARKER_BEARING_CORRECTIONS]

    wm = WorldModelCritic(
        {  # CortexAgent wiring (cortex_agent.py:219-226).
            "use_hidden": True,
            "use_value_head": True,
            "mlp_hidden_units": 16,
            "value_learning_rate": 0.05,
            "mlp_learning_rate": 0.1,
        }
    )

    # Teaching pass: record_outcome appends to self.records and runs the
    # online logistic update (critic.py:224-303). Token overlap drives
    # _similar_correction_rate (x3) to 1.0 on the matched fact; weights[3]
    # converges near 1.5 after 4 cycles -> threshold > 0.5 fires on the
    # marker-stripped version. TD(0) value head sees no test-turn updates.
    for teach_query in MARKER_BEARING_CORRECTIONS:
        wm.record_outcome(
            query=teach_query,
            proposed_answer="",
            correction=teach_query,  # non-empty -> label=1
        )

    wm_flagged = 0
    tfidf_flagged = 0
    for correction in _MARKER_FREE_CORRECTIONS:
        if _token_overlap_flag(correction, teach_set):
            tfidf_flagged += 1
        # lm_hidden=None: the untrained MLP would saturate at sigmoid(0)=0.5;
        # the string-logistic similarity head carries the lift.
        result = wm.predict_acceptance(
            query=correction, proposed_answer="", lm_hidden=None,
        )
        if float(result.get("correction_likelihood", 0.0)) > 0.5:
            wm_flagged += 1

    n = len(_MARKER_FREE_CORRECTIONS)
    return {
        "lane_07_marker_free_uptake_gap": wm_flagged / n - tfidf_flagged / n,
        "lane_07_corpus_n": float(n),
        "lane_07_wm_uptake": wm_flagged / n,
        "lane_07_baseline_uptake": tfidf_flagged / n,
    }
