"""S1.4 — HF-substrate layer-L hidden probe (pre-registered).

Runs the pre-registered layer-L probe from ``research/10-hf-layer-l-hidden-probe.md``:
- Primary: mean-pool over content tokens (stopword-excluded)
- Secondaries: last-token, max-pool
- Silhouette score per layer per pooling
- Gap = max(mid-layer primary silhouette) − primary silhouette(final layer)
- ACCEPT if gap >= +0.10, else REFUTE

Also attempts secondary LFM2.5-1.2B probe if the HF checkpoint is cached.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Corpus — verbatim from lanes/lane_03.py (eval-protected; imported, not edited)
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

# ---------------------------------------------------------------------------
# Stopwords — minimal hardcoded set for content-token filtering
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
    "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "than", "too", "very",
    "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "we", "you",
    "i", "me", "my", "our", "your", "his", "her", "their",
    "because", "if", "then", "else", "when", "where", "which", "who", "whom",
    "what", "how", "why",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with safe zero-norm handling."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _silhouette(vectors_by_concept: dict[str, list[np.ndarray]]) -> float | None:
    """Cosine silhouette = mean intra-concept cosine − mean inter-concept cosine."""
    concepts = list(vectors_by_concept)
    intra: list[float] = []
    inter: list[float] = []
    for i, ci in enumerate(concepts):
        si = vectors_by_concept[ci]
        for a_idx in range(len(si)):
            for b_idx in range(a_idx + 1, len(si)):
                intra.append(_cosine(si[a_idx], si[b_idx]))
        for cj in concepts[i + 1 :]:
            sj = vectors_by_concept[cj]
            for a_idx in range(len(si)):
                for b_idx in range(len(sj)):
                    inter.append(_cosine(si[a_idx], sj[b_idx]))
    if not intra or not inter:
        return None
    return float(np.mean(intra) - np.mean(inter))


def _content_mask(phrase: str, tokenizer: Any) -> np.ndarray:
    """Boolean mask: True at content-token positions (alpha, non-stopword).

    Falls back to all alphabetic tokens if no content tokens found;
    falls back to all tokens if still nothing.
    """
    encoding = tokenizer(phrase, return_tensors="pt")
    input_ids = encoding.input_ids[0].tolist()

    mask = np.zeros(len(input_ids), dtype=bool)
    for i, tid in enumerate(input_ids):
        token_text = tokenizer.decode([tid]).strip().lower()
        if token_text.isalpha() and token_text not in _STOPWORDS:
            mask[i] = True

    if not mask.any():
        # Fallback: all alphabetic tokens
        for i, tid in enumerate(input_ids):
            token_text = tokenizer.decode([tid]).strip().lower()
            if token_text.isalpha():
                mask[i] = True

    if not mask.any():
        mask[:] = True  # ultimate fallback: all tokens

    return mask


def _corpus_hash() -> str:
    """Stable hash of the concept battery for provenance."""
    payload = "".join(
        f"{k}:{sorted(v)}" for k, v in sorted(_CONCEPTS.items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _mid_layer_range(n_layers: int) -> tuple[int, int]:
    """Return (start, end) of mid-layer indices (25–75% depth), exclusive end."""
    start = int(math.floor(0.25 * n_layers))
    end = int(math.floor(0.75 * n_layers))
    return start, end


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------


def _forward_all_phrases(
    model: Any, tokenizer: Any, phrases: list[str]
) -> dict[str, list[np.ndarray]]:
    """One forward pass per phrase, returning hidden_states per layer as float32 numpy.

    Returns:
        ``{phrase: [hs_layer0, hs_layer1, …, hs_final_norm?]}`` where each
        element is ``(seq_len, n_embd)`` float32 ndarray.
    """
    import torch

    phrase_hiddens: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for phrase in phrases:
            enc = tokenizer(phrase, return_tensors="pt")
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states")
            phrase_hiddens[phrase] = [
                h[0].to(torch.float32).cpu().numpy() for h in hs
            ]
    return phrase_hiddens


def _pool_hidden(
    hidden: np.ndarray, pooling: str, content_mask: np.ndarray | None = None
) -> np.ndarray:
    """Pool a (seq_len, n_embd) hidden-state array.

    ``pooling``: ``"mean"`` (content-token mean if mask given, else all-token),
    ``"last"``, or ``"max"``.
    """
    if pooling == "last":
        return hidden[-1, :].copy()
    elif pooling == "max":
        return hidden.max(axis=0).copy()
    elif pooling == "mean":
        if content_mask is not None and content_mask.any():
            return hidden[content_mask].mean(axis=0).copy()
        return hidden.mean(axis=0).copy()
    else:
        raise ValueError(f"Unknown pooling: {pooling!r}")


def _compute_silhouettes(
    phrase_hiddens: dict[str, list[np.ndarray]],
    n_layers: int,
    content_masks: dict[str, np.ndarray],
    pooling: str,
) -> dict[str, float]:
    """Compute silhouette for every decoder layer plus final_norm (if present).

    Layer keys: ``"L0"``, ``"L1"``, …, ``"L{n_layers-1}"``, ``"final_norm"``.
    """
    n_hs = len(next(iter(phrase_hiddens.values())))
    silhouettes: dict[str, float] = {}

    # Decoder layers: hidden_states[1] through hidden_states[n_layers]
    for layer_idx in range(n_layers):
        hs_idx = layer_idx + 1  # skip embeddings at index 0
        by_concept: dict[str, list[np.ndarray]] = {}
        for concept, phrases in _CONCEPTS.items():
            vecs = []
            for phrase in phrases:
                cmask = content_masks.get(phrase) if pooling == "mean" else None
                vec = _pool_hidden(phrase_hiddens[phrase][hs_idx], pooling, cmask)
                vecs.append(vec)
            by_concept[concept] = vecs
        s = _silhouette(by_concept)
        silhouettes[f"L{layer_idx}"] = 0.0 if s is None else s

    # Final norm (if present): hidden_states[n_layers + 1]
    if n_hs > n_layers + 1:
        final_norm_idx = n_layers + 1
        by_concept = {}
        for concept, phrases in _CONCEPTS.items():
            vecs = []
            for phrase in phrases:
                cmask = content_masks.get(phrase) if pooling == "mean" else None
                vec = _pool_hidden(
                    phrase_hiddens[phrase][final_norm_idx], pooling, cmask
                )
                vecs.append(vec)
            by_concept[concept] = vecs
        s = _silhouette(by_concept)
        silhouettes["final_norm"] = 0.0 if s is None else s

    return silhouettes


def run_probe(
    model_id: str,
    poolings: tuple[str, ...] = ("mean", "last", "max"),
) -> dict[str, Any]:
    """Run the layer-L hidden probe on *model_id*.

    Returns a dict with keys:
    - ``model_id``, ``n_layers``, ``n_embd``, ``corpus_hash``
    - ``silhouettes``: ``{pooling: {layer_key: score}}``
    - ``gap``, ``verdict`` (for primary pooling — the first in *poolings*)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    n_embd = int(model.config.hidden_size)
    primary_pooling = poolings[0]

    all_phrases = [p for ps in _CONCEPTS.values() for p in ps]

    # Pre-compute content masks for mean-pool
    content_masks: dict[str, np.ndarray] = {}
    if "mean" in poolings:
        for phrase in all_phrases:
            content_masks[phrase] = _content_mask(phrase, tokenizer)

    # Forward all phrases once, cache all hidden states
    phrase_hiddens = _forward_all_phrases(model, tokenizer, all_phrases)

    # Compute silhouettes per pooling
    all_silhouettes: dict[str, dict[str, float]] = {}
    for pooling in poolings:
        all_silhouettes[pooling] = _compute_silhouettes(
            phrase_hiddens, n_layers, content_masks, pooling
        )

    # Compute gap and verdict for primary pooling
    primary_sils = all_silhouettes[primary_pooling]
    final_key = f"L{n_layers - 1}"
    final_score = primary_sils[final_key]
    start, end = _mid_layer_range(n_layers)
    mid_keys = [f"L{i}" for i in range(start, end)]
    mid_scores = [primary_sils[k] for k in mid_keys]
    max_mid = max(mid_scores) if mid_scores else float("-inf")
    gap = max_mid - final_score
    verdict = "ACCEPT" if gap >= 0.10 else "REFUTE"

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "n_embd": n_embd,
        "corpus_hash": _corpus_hash(),
        "primary_pooling": primary_pooling,
        "mid_layer_range": f"{start}-{end - 1}",
        "silhouettes": all_silhouettes,
        "final_score": final_score,
        "max_mid": max_mid,
        "gap": gap,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# LFM2.5-1.2B secondary probe
# ---------------------------------------------------------------------------


def run_lfm_probe() -> dict[str, Any] | None:
    """Attempt to probe LFM2.5-1.2B-Instruct from HF cache.

    Returns ``None`` if the model is not cached / not loadable.
    """
    import os

    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    lfm_dir = os.path.join(
        cache_root, "models--LiquidAI--LFM2.5-1.2B-Instruct"
    )
    if not os.path.isdir(lfm_dir):
        return None

    try:
        return run_probe("LiquidAI/LFM2.5-1.2B-Instruct")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_results_table(results: dict[str, Any], heading_level: int = 1) -> str:
    """Format probe results as a markdown table.

    ``heading_level`` controls the top-level heading (1 = ``#``, 2 = ``##``, etc.).
    """
    h = "#" * heading_level
    lines: list[str] = []
    lines.append(f"{h} S1.4 HF Layer-L Hidden Probe Results")
    lines.append("")
    lines.append(f"- **Model**: `{results['model_id']}`")
    lines.append(f"- **Layers**: {results['n_layers']} decoder layers")
    lines.append(f"- **Embedding dim**: {results['n_embd']}")
    lines.append(f"- **Corpus hash**: `{results['corpus_hash']}`")
    lines.append(f"- **Primary pooling**: `{results['primary_pooling']}` (mean-pool over content tokens)")
    lines.append(f"- **Mid-layer range**: {results['mid_layer_range']} (25–75% depth)")
    lines.append(f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    all_sils = results["silhouettes"]
    poolings = list(all_sils.keys())
    primary = results["primary_pooling"]

    # Collect all layer keys in order
    layer_keys = sorted(
        all_sils[primary].keys(),
        key=lambda k: (
            999 if k == "final_norm" else int(k[1:]),
            k,
        ),
    )

    # Header
    header = "| Layer | " + " | ".join(
        f"{p}{' **← primary**' if p == primary else ''}" for p in poolings
    ) + " |"
    lines.append(header)
    sep = "|-------|" + "|".join(["-------" for _ in poolings]) + "|"
    lines.append(sep)

    for lk in layer_keys:
        vals = [f"{all_sils[p].get(lk, '—'):.4f}" if isinstance(all_sils[p].get(lk), float) else "—" for p in poolings]
        lines.append(f"| {lk} | " + " | ".join(vals) + " |")

    lines.append("")

    # Gap and verdict
    lines.append(f"{h}# Verdict")
    lines.append("")
    lines.append(f"- **Final-layer ({primary}) silhouette**: {results['final_score']:.4f}")
    lines.append(f"- **Max mid-layer ({primary}) silhouette**: {results['max_mid']:.4f}")
    lines.append(f"- **Gap**: {results['gap']:+.4f} (threshold: +0.10)")
    lines.append(f"- **Verdict**: **{results['verdict']}**")
    lines.append("")
    if results["verdict"] == "ACCEPT":
        lines.append(
            "H-L accepted: at some mid-depth layer L, sense-corrected phrase "
            "hiddens cluster by concept better than at the final layer "
            "(gap >= +0.10)."
        )
    else:
        lines.append(
            "H-L refuted: mid-layer hiddens do NOT cluster by concept better "
            "than the final layer (gap < +0.10). This confirms lane_03's "
            "refutation on a substrate that can see every layer."
        )
    lines.append("")
    lines.append("---")
    lines.append("*Pre-registered spec: `research/10-hf-layer-l-hidden-probe.md`*")

    return "\n".join(lines)


def format_results_table_raw(results: dict[str, Any], poolings: list[str]) -> str:
    """Plain-text results for CLI output."""
    lines: list[str] = []
    primary = results["primary_pooling"]
    all_sils = results["silhouettes"]

    n_layers = results["n_layers"]
    lines.append(f"Model: {results['model_id']}  Layers: {n_layers}  Corpus: {results['corpus_hash']}")
    lines.append(f"Primary: {primary}  Mid-range: {results['mid_layer_range']}")

    layer_keys = [f"L{i}" for i in range(n_layers)]
    if "final_norm" in all_sils.get(primary, {}):
        layer_keys.append("final_norm")

    for p in poolings:
        marker = " ← PRIMARY" if p == primary else ""
        lines.append(f"\n  {p}{marker}:")
        for lk in layer_keys:
            s = all_sils[p].get(lk)
            if s is not None:
                lines.append(f"    {lk:>12s}  {s:+.4f}")

    lines.append(f"\nGap: {results['gap']:+.4f}  Verdict: {results['verdict']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S1.4 HF layer-L hidden probe"
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HF model ID (default: from hf_model_choice.HF_MODEL_ID)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results markdown to this path",
    )
    parser.add_argument(
        "--lfm",
        action="store_true",
        help="Also attempt LFM2.5-1.2B secondary probe",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args(argv)

    model_id = args.model_id
    if model_id is None:
        try:
            from oczy.lm.hf_model_choice import HF_MODEL_ID  # type: ignore[import-untyped]
            model_id = HF_MODEL_ID
        except ImportError:
            print("ERROR: --model-id required (hf_model_choice not available)")
            return 1

    if not args.quiet:
        print(f"Probing: {model_id}")

    try:
        results = run_probe(model_id)
    except Exception as exc:
        print(f"Probe failed: {exc}", file=sys.stderr)
        return 1

    poolings = list(results["silhouettes"].keys())

    if not args.quiet:
        print(format_results_table_raw(results, poolings))

    # Write markdown log
    output_path = args.output
    if output_path is None:
        output_path = "experiments_logs/2026-07-01_s1_4_hf_layer_probe.md"

    md = format_results_table(results)

    # Append LFM secondary if requested
    if args.lfm:
        lfm_results = run_lfm_probe()
        if lfm_results is not None:
            md += "\n\n---\n\n## Secondary #3: LFM2.5-1.2B-Instruct\n\n"
            md += format_results_table(lfm_results, heading_level=3)
        else:
            md += "\n\n---\n\n## Secondary #3: LFM2.5-1.2B-Instruct\n\n"
            md += "Not loadable from HF cache — skipped.\n"

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)

    if not args.quiet:
        print(f"\nWrote: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
