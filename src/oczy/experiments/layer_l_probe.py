"""Layer-L hidden extraction probe (Experiment 03).

Measures whether feeding ``KVCortex.observe()`` real residuals at mid/upper
layers produces more semantically separable ``warm_state`` vectors than the
current final-layer mean-pool.  Follows the Experiment 03 spec and reuses the
silhouette algorithm from ``lanes/lane_03.py``.

Modes:
  - ``--driver mock``: fast, deterministic, semantics-free floor using
    norm-matched random vectors.
  - ``--driver real``: loads ``LiquidAI/LFM2.5-1.2B-Instruct`` via transformers
    and computes layer-L silhouettes.

On any real-driver failure, the module falls back to mock mode and emits
``ASI real_driver=failed``.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Concept battery (matches lanes/lane_03.py)
# ---------------------------------------------------------------------------

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

_D_EMBD = 2048
_N_LAYERS = 16


# ---------------------------------------------------------------------------
# Cosine + silhouette helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _silhouette(warm_by_concept: dict[str, list[np.ndarray]]) -> float | None:
    """Cosine silhouette = mean intra-concept cosine - mean inter-concept cosine."""
    concepts = list(warm_by_concept)
    intra: list[float] = []
    inter: list[float] = []
    for i, ci in enumerate(concepts):
        si = warm_by_concept[ci]
        for a_idx in range(len(si)):
            for b_idx in range(a_idx + 1, len(si)):
                intra.append(_cosine(si[a_idx], si[b_idx]))
        for cj in concepts[i + 1 :]:
            sj = warm_by_concept[cj]
            for a_idx in range(len(si)):
                for b_idx in range(len(sj)):
                    inter.append(_cosine(si[a_idx], sj[b_idx]))
    if not intra or not inter:
        return None
    return float(np.mean(intra) - np.mean(inter))


# ---------------------------------------------------------------------------
# Mock driver
# ---------------------------------------------------------------------------


def _mock_hidden_vectors(
    layer_idx: int, *, rng: np.random.RandomState | None = None
) -> np.ndarray:
    """Return a deterministic, norm-matched random vector for mock smoke runs."""
    if rng is None:
        rng = np.random.RandomState(0)
    vec = rng.standard_normal(_D_EMBD).astype(np.float32)
    # Norm-match to roughly unit vectors; slight per-layer scaling so mock
    # conditions are not identical.
    target_norm = 1.0 + 0.05 * layer_idx
    vec /= np.linalg.norm(vec)
    return (vec * target_norm).astype(np.float32)


def _mock_probe() -> dict[str, float]:
    """Fast semantics-free probe: random floor + layer-scaled random vectors."""
    from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

    all_phrases = [p for ps in _CONCEPTS.values() for p in ps]
    rng = np.random.RandomState(0)
    phrase_vectors: dict[str, list[np.ndarray]] = {}
    for layer_idx in (0, 9, 13):
        per_layer: list[np.ndarray] = []
        for _ in all_phrases:
            per_layer.append(_mock_hidden_vectors(layer_idx, rng=rng))
        phrase_vectors[layer_idx] = per_layer

    def _sil_for(layer_idx: int) -> float | None:
        warm_by_concept: dict[str, list[np.ndarray]] = {}
        for concept, phrases in _CONCEPTS.items():
            cortex = KVCortex(
                KVCortexConfig(
                    d_cortex=128,
                    d_embd=_D_EMBD,
                    n_layers=_N_LAYERS,
                    seed=0,
                )
            )
            cortex.reset_warm_to_zeros()
            phrase_iter = iter(phrase_vectors[layer_idx])
            states: list[np.ndarray] = []
            for _ in phrases:
                states.append(cortex.observe(next(phrase_iter), correction_signal=0.0))
            warm_by_concept[concept] = states
        return _silhouette(warm_by_concept)

    silhouettes: dict[str, float] = {
        "R_random": 0.0,
        "L0_last": _sil_for(0) or 0.0,
        "L9_last": _sil_for(9) or 0.0,
        "L13_last": _sil_for(13) or 0.0,
        "L9_meanpool": 0.0,
    }
    return silhouettes


# ---------------------------------------------------------------------------
# Real driver
# ---------------------------------------------------------------------------


def _hf_probe() -> dict[str, float] | None:
    """Load the HF LFM2.5-1.2B-Instruct twin and compute layer-L silhouettes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    model.eval()

    all_phrases = [p for ps in _CONCEPTS.values() for p in ps]
    phrase_hiddens: dict[str, list] = {}
    with torch.no_grad():
        for phrase in all_phrases:
            enc = tokenizer(phrase, return_tensors="pt")
            out = model(**enc)
            hs = out.hidden_states
            if hs is None or len(hs) != _N_LAYERS + 1:
                return None
            phrase_hiddens[phrase] = [
                h[0].to(torch.float32).numpy() for h in hs
            ]

    def _last_token(seq_arr: np.ndarray) -> np.ndarray:
        return seq_arr[-1, :]

    def _mean_pool(seq_arr: np.ndarray) -> np.ndarray:
        return seq_arr.mean(axis=0)

    silhouettes: dict[str, float] = {}
    conditions = [
        ("L0_last", 0, _last_token),
        ("L9_last", 9, _last_token),
        ("L13_last", 13, _last_token),
        ("L9_meanpool", 9, _mean_pool),
    ]

    for label, idx, pool in conditions:
        warm_by_concept: dict[str, list[np.ndarray]] = {}
        for concept, phrases in _CONCEPTS.items():
            cortex = KVCortex(
                KVCortexConfig(
                    d_cortex=128,
                    d_embd=_D_EMBD,
                    n_layers=_N_LAYERS,
                    seed=0,
                )
            )
            cortex.reset_warm_to_zeros()
            states: list[np.ndarray] = []
            for phrase in phrases:
                vec = pool(phrase_hiddens[phrase][idx])
                if vec.shape[0] != _D_EMBD:
                    return None
                states.append(cortex.observe(vec, correction_signal=0.0))
            warm_by_concept[concept] = states
        s = _silhouette(warm_by_concept)
        silhouettes[label] = 0.0 if s is None else s

    # Best-effort GGUF final-layer mean-pool baseline.
    try:
        from llama_cpp import Llama

        from oczy.experiments.multi_fact_stressor import _resolve_gguf_path
        from oczy.lm.cvec_driver import LlamaCVecDriver

        resolved = _resolve_gguf_path()
        if resolved is not None:
            llm = Llama(
                model_path=str(resolved),
                n_ctx=512,
                n_threads=4,
                embedding=True,
                verbose=False,
            )
            driver = LlamaCVecDriver(llm)
            if driver.n_embd == _D_EMBD:
                gguf_warm: dict[str, list[np.ndarray]] = {}
                ok = True
                for concept, phrases in _CONCEPTS.items():
                    cortex = KVCortex(
                        KVCortexConfig(
                            d_cortex=128,
                            d_embd=driver.n_embd,
                            n_layers=_N_LAYERS,
                            seed=0,
                        )
                    )
                    cortex.reset_warm_to_zeros()
                    states = []
                    for phrase in phrases:
                        emb = driver.peek_embedding(
                            phrase, last_token_only=False
                        )
                        if emb.shape[0] != driver.n_embd:
                            ok = False
                            break
                        states.append(
                            cortex.observe(emb, correction_signal=0.0)
                        )
                    if not ok:
                        break
                    gguf_warm[concept] = states
                if ok:
                    s = _silhouette(gguf_warm)
                    if s is not None:
                        silhouettes["final_meanpool"] = s
    except Exception:
        pass

    return silhouettes


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_gap(silhouettes: dict[str, float]) -> dict[str, Any]:
    """Return primary metric and per-condition gap info."""
    final = silhouettes.get("final_meanpool")
    mid_layer_labels = [
        k
        for k in silhouettes
        if k not in ("R_random", "final_meanpool")
    ]
    mid_values = [silhouettes[k] for k in mid_layer_labels]
    max_mid = max(mid_values) if mid_values else 0.0
    gap = (
        (max_mid - final)
        if final is not None
        else 0.0
    )
    return {
        "gap": gap,
        "max_mid": max_mid,
        "final": final,
        "mid_labels": mid_layer_labels,
    }


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer-L hidden extraction probe"
    )
    parser.add_argument(
        "--driver",
        choices=["mock", "real"],
        default="mock",
        help="mock = fast deterministic floor; real = HF LFM2.5-1.2B-Instruct",
    )
    args = parser.parse_args(argv)

    if args.driver == "real":
        try:
            silhouettes = _hf_probe()
        except Exception:
            silhouettes = None
        if silhouettes is None:
            print("ASI real_driver=failed")
            silhouettes = _mock_probe()
    else:
        silhouettes = _mock_probe()

    result = _compute_gap(silhouettes)
    gap = result["gap"]

    if math.isnan(gap):
        gap = 0.0
    print(f"METRIC layer_l_silhouette_gap={gap}")
    for label, value in silhouettes.items():
        print(f"ASI warm_sep_silhouette_{label}={value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
