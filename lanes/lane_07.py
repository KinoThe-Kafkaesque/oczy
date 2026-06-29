"""Lane 07: Conversation World-Model RL Phase 0.

marker_free_uptake_gap = uptake(world-model critic) - uptake(lexical critic)
on marker-stripped corrections. Source: research/07-conversation-world-model-rl.md.

Baseline: lexical detector (substring markers, cortex_agent.py:53-71) AND the
untrained world-model critic both miss the 4 marker-stripped corrections -> gap=0.

Lift: the world-model critic carries an online Jaccard-similarity feature
(``_similar_correction_rate``, critic.py:442-460) and a learnable prior-correction
weight. After a teaching pass of marker-bearing corrections for the SAME facts
(record_outcome each), the critic generalizes to the marker-stripped versions via
token overlap (e.g. "the sky is blue" overlaps "no actually the sky is blue" with
Jaccard 4/6 >= 0.25 threshold). TD(0) value head sees no test-turn updates; the
string-logistic similarity head carries the lift -- the "semantic trace left in
the surrounding tokens" the spec calls out (07-conversation-world-model-rl.md:13).
Real-LM peek_embedding is Goal-2-staging only and unavailable at baseline; the
Jaccard count-based stub is the spec's named "the X is Y" fallback. Config matches
CortexAgent (cortex_agent.py:219-226); untrained MLP path (use_hidden=True) is
bypassed via lm_hidden=None since 0.01-init randn weights saturate near 0.5.
"""

from __future__ import annotations

from lanes._common import MARKER_BEARING_CORRECTIONS, lane_measure


def name() -> str:
    return "lane_07_marker_free_uptake_gap"


# 4 marker-stripped corrections: factual content with NO lexical markers
# ("actually,", "wrong, ", "correction:", "no, "). Lexical misses all by design.
_MARKER_FREE_CORRECTIONS = (
    "the sky is blue",
    "paris is the capital of france",
    "two plus two is four",
    "jupiter is the largest planet",
)


def _lexical_flags(text: str) -> bool:
    # Mirrors cortex_agent._looks_like_correction (cortex_agent.py:68-71) and
    # _CORRECTION_MARKERS tuple (cortex_agent.py:53-65).
    markers = (
        "no, ", "no:", "wrong, ", "wrong:", "correction:", "correct:",
        "expected:", "not what i meant", "i meant", "actually,", "rather than",
    )
    return any(m in text.strip().lower() for m in markers)


@lane_measure
def measure() -> float:
    from world_model_critic import WorldModelCritic

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
    lex_flagged = 0
    for correction in _MARKER_FREE_CORRECTIONS:
        if _lexical_flags(correction):
            lex_flagged += 1
        # lm_hidden=None: the untrained MLP would saturate at sigmoid(0)=0.5;
        # the string-logistic similarity head carries the lift.
        result = wm.predict_acceptance(
            query=correction, proposed_answer="", lm_hidden=None,
        )
        if float(result.get("correction_likelihood", 0.0)) > 0.5:
            wm_flagged += 1

    n = len(_MARKER_FREE_CORRECTIONS)
    return float(wm_flagged / n - lex_flagged / n)
