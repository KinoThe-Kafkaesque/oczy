"""Conversation world model probe (Experiment 07).

Measures whether the WorldModelCritic, reading real LM hidden states, predicts
corrections better than its string-only path (A1 vs B0), and whether it
recovers marker-stripped corrections that the lexical gate misses (C0 vs C1).

Mode:
  - ``--driver mock``: semantic-null control; AUC delta should collapse to 0.
  - ``--driver real``: loads LFM2.5-1.2B-Instruct GGUF and measures the gap.

Primary metric: ``critic_auc_delta`` = AUC(hidden path) - AUC(string-only path).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

_MARKER_BEARING_CORRECTIONS = (
    "no actually the sky is blue",
    "wrong paris is the capital of france",
    "correction two plus two is four",
    "actually jupiter is the largest planet",
)

_MARKER_FREE_CORRECTIONS = (
    "the sky is blue",
    "paris is the capital of france",
    "two plus two is four",
    "jupiter is the largest planet",
)

_ACCEPTANCES = (
    "water boils at 100 degrees celsius",
    "the earth orbits the sun",
    "a square has four equal sides",
    "iron is a dense metal",
)


def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based ROC AUC (Mann-Whitney U)."""
    pos = [s for s, label in zip(scores, labels) if label == 1]
    neg = [s for s, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    n = 0
    for p in pos:
        for ng in neg:
            n += 1
            if p > ng:
                wins += 1.0
            elif p == ng:
                wins += 0.5
    return wins / n


def _lexical_flags(text: str) -> bool:
    markers = (
        "no, ", "no:", "wrong, ", "wrong:", "correction:", "correct:",
        "expected:", "not what i meant", "i meant", "actually,", "rather than",
    )
    lowered = text.strip().lower()
    return any(m in lowered for m in markers)


def _mock_embedding(text: str, n_embd: int = 16) -> np.ndarray:
    """Deterministic hash embedding matching multi_fact_stressor _MockDriver."""
    idx = sum(ord(c) for c in text) % n_embd
    h = np.zeros(n_embd, dtype=np.float32)
    h[idx] = 1.0
    h[(idx + 1) % n_embd] = float(len(text)) * 0.05
    return h


def _build_critic() -> Any:
    from world_model_critic import WorldModelCritic

    return WorldModelCritic(
        {
            "use_hidden": True,
            "use_value_head": True,
            "mlp_hidden_units": 16,
            "value_learning_rate": 0.05,
            "mlp_learning_rate": 0.1,
        }
    )


def _train_and_score(embeddings: list[np.ndarray] | None) -> tuple[float, float]:
    """Train critic on 8-example corpus and return (auc_real_or_emb, auc_string)."""
    queries = list(_MARKER_BEARING_CORRECTIONS) + list(_ACCEPTANCES)
    labels = [1] * len(_MARKER_BEARING_CORRECTIONS) + [0] * len(_ACCEPTANCES)

    wm = _build_critic()
    for q, emb, lbl in zip(queries, embeddings or [None] * len(queries), labels):
        kwargs = {
            "query": q,
            "proposed_answer": "",
            "correction": q if lbl == 1 else None,
        }
        if emb is not None:
            kwargs["lm_hidden"] = emb
        wm.record_outcome(**kwargs)

    scores_hidden: list[float] = []
    scores_string: list[float] = []
    for q, emb in zip(queries, embeddings or [None] * len(queries)):
        r_hidden = wm.predict_acceptance(
            query=q, proposed_answer="", lm_hidden=emb
        )
        r_string = wm.predict_acceptance(
            query=q, proposed_answer="", lm_hidden=None
        )
        scores_hidden.append(float(r_hidden.get("correction_likelihood", 0.0)))
        scores_string.append(float(r_string.get("correction_likelihood", 0.0)))

    return _auc(scores_hidden, labels), _auc(scores_string, labels)


def _marker_free_uptake() -> float:
    """Teach on marker-bearing, test on marker-stripped; return gap."""
    wm = _build_critic()
    for teach_query in _MARKER_BEARING_CORRECTIONS:
        wm.record_outcome(
            query=teach_query,
            proposed_answer="",
            correction=teach_query,
        )

    wm_flagged = 0
    lex_flagged = 0
    for correction in _MARKER_FREE_CORRECTIONS:
        if _lexical_flags(correction):
            lex_flagged += 1
        result = wm.predict_acceptance(
            query=correction, proposed_answer="", lm_hidden=None
        )
        if float(result.get("correction_likelihood", 0.0)) > 0.5:
            wm_flagged += 1

    n = len(_MARKER_FREE_CORRECTIONS)
    return float(wm_flagged / n - lex_flagged / n)


# ---------------------------------------------------------------------------
# Driver-specific runners
# ---------------------------------------------------------------------------


def _run_real_driver() -> dict[str, float] | None:
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver

    try:
        driver = LlamaCVecDriver.load(
            CVecDriverConfig(n_ctx=256, n_threads=4, embedding=True)
        )
    except Exception:
        return None
    if driver.n_embd == 0:
        return None

    queries = list(_MARKER_BEARING_CORRECTIONS) + list(_ACCEPTANCES)
    embeddings = []
    for q in queries:
        emb = driver.peek_embedding(q, last_token_only=False)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[0] != driver.n_embd:
            return None
        embeddings.append(emb)

    auc_hidden, auc_string = _train_and_score(embeddings)
    return {
        "critic_auc_delta": max(0.0, auc_hidden - auc_string),
        "accept_pred_auc_hidden": auc_hidden,
        "accept_pred_auc_string": auc_string,
        "marker_free_uptake_gap": _marker_free_uptake(),
    }


def _run_mock_driver() -> dict[str, float]:
    n_embd = 16
    queries = list(_MARKER_BEARING_CORRECTIONS) + list(_ACCEPTANCES)
    embeddings = [_mock_embedding(q, n_embd) for q in queries]
    auc_hidden, auc_string = _train_and_score(embeddings)
    return {
        "critic_auc_delta": max(0.0, auc_hidden - auc_string),
        "accept_pred_auc_hidden": auc_hidden,
        "accept_pred_auc_string": auc_string,
        "marker_free_uptake_gap": _marker_free_uptake(),
    }


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conversation world model probe"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
    )
    args = parser.parse_args(argv)

    if args.driver == "real":
        try:
            results = _run_real_driver()
        except Exception:
            results = None
        if results is None:
            print("ASI real_driver=failed")
            print("METRIC critic_auc_delta=nan")
            return 0
    else:
        results = _run_mock_driver()

    print(f"METRIC marker_free_uptake_gap={results['marker_free_uptake_gap']}")
    print(f"ASI critic_auc_delta={results['critic_auc_delta']}")
    print(f"ASI accept_pred_auc_hidden={results['accept_pred_auc_hidden']}")
    print(f"ASI accept_pred_auc_string={results['accept_pred_auc_string']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
