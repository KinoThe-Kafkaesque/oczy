"""Lane 03: Real Hidden-State Extraction at Layer L -- warm_sep_silhouette.

Measures cosine silhouette (mean intra-concept cosine minus mean
inter-concept cosine) of ``warm_state`` vectors over a labeled paraphrase
battery, using the current FINAL-LAYER mean-pool embedding path
(``peek_embedding(last_token_only=False)``) that ``CortexAgent.perceive``
feeds the cortex today. This is the *foundation baseline* against which
layer-L extraction (research/03-layer-l-hidden-extraction.md H1) is to be
compared. Real-driver only: the LFM2.5-1.2B-Instruct residual must be
exercised; the mock hash-embeddings carry no semantics. On GGUF missing,
driver load failure, or dim mismatch, returns ``float('nan')``. Never
raises; deterministic (fixed seed + fixed phrase order).
"""

from __future__ import annotations


_CONCEPTS: dict[str, list[str]] = {
    "paris": [
        "The capital of France is Paris.",
        "France's capital city is Paris.",
        "Paris is the capital of France.",
    ],
    "water": [
        "Water boils at 100 degrees Celsius.",
        "The boiling point of water is 100C.",
        "At sea level, water boils at 100 degrees.",
    ],
    "gravity": [
        "Gravity pulls objects toward Earth.",
        "Things fall because of gravity.",
        "Earth's gravity attracts masses downward.",
    ],
}


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def name() -> str:
    return "lane_03_warm_sep_silhouette"


def measure() -> float:
    try:
        import numpy as np

        from src.oczy.experiments.multi_fact_stressor import _resolve_gguf_path
        from src.oczy.lm.cvec_driver import LlamaCVecDriver
        from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig
        from llama_cpp import Llama

        resolved = _resolve_gguf_path()
        if resolved is None:
            return float("nan")

        llm = Llama(
            model_path=str(resolved),
            n_ctx=512,
            n_threads=4,
            embedding=True,
            verbose=False,
        )
        driver = LlamaCVecDriver(llm)
        if driver.n_embd == 0:
            return float("nan")

        # Fit a fresh cortex per concept, observe each paraphrase's
        # final-layer mean-pooled embedding, capture warm_state after each.
        # One warm_state vector per (concept, paraphrase): 3 per cluster.
        warm_by_concept: dict[str, list] = {}
        for concept, phrases in _CONCEPTS.items():
            cortex = KVCortex(
                KVCortexConfig(
                    d_embd=driver.n_embd,
                    n_layers=16,  # tracked fix-up: LFM2.5-1.2B has 16 layers, not 28.
                )
            )
            states = []
            for phrase in phrases:
                emb = driver.peek_embedding(phrase, last_token_only=False)
                if emb.shape[0] != driver.n_embd:
                    return float("nan")
                state = cortex.observe(emb, correction_signal=0.0)
                states.append(state)
            warm_by_concept[concept] = states

        # Cosine silhouette = mean intra-concept cosine - mean inter-concept cosine.
        concepts = list(warm_by_concept)
        intra: list[float] = []
        inter: list[float] = []
        for i, ci in enumerate(concepts):
            si = warm_by_concept[ci]
            for a in range(len(si)):
                for b in range(a + 1, len(si)):
                    intra.append(_cosine(si[a], si[b]))
            for cj in concepts[i + 1:]:
                sj = warm_by_concept[cj]
                for a in range(len(si)):
                    for b in range(len(sj)):
                        inter.append(_cosine(si[a], sj[b]))

        if not intra or not inter:
            return float("nan")
        return float(np.mean(intra) - np.mean(inter))
    except Exception:
        return float("nan")