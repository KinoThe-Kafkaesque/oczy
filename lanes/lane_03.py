"""Lane 03: Real Hidden-State Extraction at Layer L -- warm_sep_silhouette.

Measures cosine silhouette (mean intra-concept cosine minus mean
inter-concept cosine) of ``warm_state`` vectors over a labeled paraphrase
battery. Originally a single GGUF final-layer mean-pool baseline; now
extended (research/03 H1, run #156) to probe mid-layer last-token and
max-pool hidden states from the HF LFM2.5-1.2B-Instruct model. The lane
returns the MAX silhouette across all conditions:

  * last-token at layer 9  (hidden_states[10][0,-1,:])
  * last-token at layer 13 (hidden_states[14][0,-1,:])
  * last-token at layer 15 (hidden_states[16][0,-1,:])
  * max-pool   at layer 14 (hidden_states[15][0].max(dim=0).values)
  * GGUF final-layer mean-pool baseline (best-effort, inner try/except)

Real-driver only: the LFM2.5-1.2B-Instruct residual must be exercised;
mock hash-embeddings carry no semantics. On HF load failure (or any
crash), returns ``float('nan')``. The GGUF baseline is best-effort: if
the GGUF is missing or its driver fails, the HF conditions still stand.
Never raises; deterministic (fixed seed + fixed phrase order).
"""

from __future__ import annotations

from lanes._common import cosine, lane_measure

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




def _silhouette(warm_by_concept: dict[str, list]) -> float | None:
    """Cosine silhouette = mean intra-concept cosine - mean inter-concept cosine.

    Returns ``None`` when there are too few pairs to score.
    """
    import numpy as np

    concepts = list(warm_by_concept)
    intra: list[float] = []
    inter: list[float] = []
    for i, ci in enumerate(concepts):
        si = warm_by_concept[ci]
        for a in range(len(si)):
            for b in range(a + 1, len(si)):
                intra.append(cosine(si[a], si[b]))
        for cj in concepts[i + 1:]:
            sj = warm_by_concept[cj]
            for a in range(len(si)):
                for b in range(len(sj)):
                    inter.append(cosine(si[a], sj[b]))
    if not intra or not inter:
        return None
    return float(np.mean(intra) - np.mean(inter))


def name() -> str:
    return "lane_03_warm_sep_silhouette"


@lane_measure
def measure() -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig

    # --- HF load (no device_map: accelerate is not available) ---
    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    model.eval()

    D_EMBD = 2048
    N_LAYERS = 16  # LFM2.5-1.2B has 16 transformer layers.

    # --- cache per-phrase hidden states: one HF forward per phrase ---
    all_phrases = [p for ps in _CONCEPTS.values() for p in ps]
    phrase_hiddens: dict[str, list] = {}
    with torch.no_grad():
        for phrase in all_phrases:
            enc = tok(phrase, return_tensors="pt")
            out = model(**enc)
            hs = out.hidden_states  # tuple of (N_LAYERS+1) tensors (1, seq, d_embd)
            if hs is None or len(hs) != N_LAYERS + 1:
                return float("nan")
            # Store each layer's (seq, d_embd) slice as float32 numpy.
            phrase_hiddens[phrase] = [
                h[0].to(torch.float32).numpy() for h in hs
            ]

    # --- pooling pickers per condition ---
    def last_token(seq_arr):
        return seq_arr[-1, :]

    def max_pool(seq_arr):
        return seq_arr.max(axis=0)

    # (label, hidden_states index, pool_fn)
    conditions = [
        ("last_L9", 10, last_token),
        ("last_L13", 14, last_token),
        ("last_L15", 16, last_token),
        ("maxpool_L14", 15, max_pool),
    ]

    def silhouette_for(idx: int, pool) -> float | None:
        warm_by_concept: dict[str, list] = {}
        for concept, phrases in _CONCEPTS.items():
            cortex = KVCortex(
                KVCortexConfig(
                    d_cortex=128,
                    d_embd=D_EMBD,
                    n_layers=N_LAYERS,
                    seed=0,
                )
            )
            cortex.reset_warm_to_zeros()
            states = []
            for phrase in phrases:
                vec = pool(phrase_hiddens[phrase][idx])
                if vec.shape[0] != D_EMBD:
                    return None
                state = cortex.observe(vec, correction_signal=0.0)
                states.append(state)
            warm_by_concept[concept] = states
        return _silhouette(warm_by_concept)

    silhouettes: list[float] = []
    for _label, idx, pool in conditions:
        try:
            s = silhouette_for(idx, pool)
            if s is not None:
                silhouettes.append(s)
        except Exception:
            continue

    # --- GGUF final-layer mean-pool baseline (best-effort) ---
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
            if driver.n_embd == D_EMBD:
                gguf_warm: dict[str, list] = {}
                ok = True
                for concept, phrases in _CONCEPTS.items():
                    cortex = KVCortex(
                        KVCortexConfig(
                            d_cortex=128,
                            d_embd=driver.n_embd,
                            n_layers=N_LAYERS,
                            seed=0,
                        )
                    )
                    cortex.reset_warm_to_zeros()
                    states = []
                    for phrase in phrases:
                        emb = driver.peek_embedding(phrase, last_token_only=False)
                        if emb.shape[0] != driver.n_embd:
                            ok = False
                            break
                        states.append(cortex.observe(emb, correction_signal=0.0))
                    if not ok:
                        break
                    gguf_warm[concept] = states
                if ok:
                    s = _silhouette(gguf_warm)
                    if s is not None:
                        silhouettes.append(s)
    except Exception:
        pass

    if not silhouettes:
        return float("nan")
    return float(max(silhouettes))
