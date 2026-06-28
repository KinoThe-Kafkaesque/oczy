"""Lane 07: Conversation World-Model RL Phase 0.

Measures marker_free_uptake_gap = uptake(world-model critic) - uptake(lexical critic)
on marker-stripped corrections. Source: research/07-conversation-world-model-rl.md.

Lexical critic mirrors ``_looks_like_correction`` (cortex_agent.py:68-71) substring
match against ``_CORRECTION_MARKERS`` (cortex_agent.py:53-65: "no, ", "wrong, ",
"correction:", "actually,", "rather than", ...). World-model critic is
``WorldModelCritic`` at default config (string-features mode, use_hidden=False),
flagged when ``predict_acceptance(correction)["correction_likelihood"] > 0.5``
following the cortex_agent.py:463-467 calling convention (query=text, proposed_answer="").

At baseline the world-model is untrained and its default bias (-0.2) places short
non-ambiguous corrections just under 0.5 by construction, so the expected gap is ~0.0
(the lexical detector misses all marker-free corrections by design, and the
world-model's string-features priors also fail to flag them).
"""

from __future__ import annotations


def name() -> str:
    return "lane_07_marker_free_uptake_gap"


# Fixed marker-stripped corrections: correct content with NO lexical correction
# markers ("actually,", "wrong, ", "correction:", "no, ", ...). All are paraphrased
# corrections the lexical detector should miss by construction.
_MARKER_FREE_CORRECTIONS = (
    "the sky is blue",
    "paris is the capital of france",
    "two plus two is four",
    "jupiter is the largest planet",
)


def _lexical_flags(text: str) -> bool:
    # Mirrors cortex_agent._looks_like_correction (cortex_agent.py:68-71) and
    # the _CORRECTION_MARKERS tuple (cortex_agent.py:53-65).
    markers = (
        "no, ",
        "no:",
        "wrong, ",
        "wrong:",
        "correction:",
        "correct:",
        "expected:",
        "not what i meant",
        "i meant",
        "actually,",
        "rather than",
    )
    lowered = text.strip().lower()
    return any(marker in lowered for marker in markers)


def measure() -> float:
    try:
        try:
            from world_model_critic import WorldModelCritic
        except Exception:
            # Lane not yet implemented at baseline (world-model module unavailable).
            return 0.0

        wm = WorldModelCritic()  # default config; use_hidden=False

        wm_flagged = 0
        lex_flagged = 0
        for correction in _MARKER_FREE_CORRECTIONS:
            if _lexical_flags(correction):
                lex_flagged += 1
            result = wm.predict_acceptance(query=correction, proposed_answer="")
            if float(result.get("correction_likelihood", 0.0)) > 0.5:
                wm_flagged += 1

        n = len(_MARKER_FREE_CORRECTIONS)
        wm_uptake = wm_flagged / n
        lex_uptake = lex_flagged / n
        return float(wm_uptake - lex_uptake)
    except Exception:
        return float("nan")