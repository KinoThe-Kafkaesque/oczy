#!/usr/bin/env python3
"""Collect final-layer LM hiddens at correction moments for SVD-init.

Purpose: produce ``reports/correction_hiddens.npy`` -- a stack of
``(N, d_embd)`` float32 vectors, one per correction-style utterance --
so that ``KVCortex.init_proj_c_from_svd`` can initialise the cortex's
``proj_c`` from the leading right singular vectors. With the projector
seeded on real correction-aligned structure, the cortex's emitted cvec
direction lives in ``proj_c`` (which is persisted) rather than in the
warm-side ``last_correction_hidden`` (which is not). See GOALS.md
"meaningful cvec steering" sub-goal.

Distribution match: the harness uses the same CortexAgent + driver +
``peek_embedding(last_token_only=False)`` path that ``perceive()``
calls (``experiments/cortex_agent.py``), so the SVD basis is drawn
from the same input distribution the cortex sees at runtime. Layer-L
(mid-network) extraction (Goal 2) is not a prerequisite.

Run:
    uv run python experiments/lm_perception/collect_correction_hiddens.py
    uv run python experiments/lm_perception/collect_correction_hiddens.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from correction_benchmark.dataset import build_dataset

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.lm import CVecDriverConfig
from plastic_cortex.kv_cortex import KVCortexConfig


REPORTS_DIR = Path(__file__).resolve().parent / "reports"
HIDDENS_PATH = REPORTS_DIR / "correction_hiddens.npy"
META_PATH = REPORTS_DIR / "correction_hiddens_meta.json"


# Templatings for paraphrasing each canonical correction episode into
# multiple distinct utterances. The dataset has 12 episodes; each is
# expanded through these wrappers so N >= d_cortex (64 for production).
# The wrappers vary surface form (question framing vs imperative vs
# statement) so the SVD captures content structure rather than boilerplate.
_PARAPHRASE_TEMPLATES = [
    "{request}  -- to be clear: {correction}",
    "Earlier I asked: {request!r}. {correction}",
    "For the record, {correction} (re: {request})",
    "Just so we're aligned: when I said {request!r}, what I meant was: {correction}",
    "{correction} That's the rule for {request!r}.",
    "Heads up on {request!r}: {correction}",
]


def _paraphrase_episodes(episodes) -> list[str]:
    """Expand each canonical correction episode into templated utterances."""
    out: list[str] = []
    for ep in episodes:
        for tmpl in _PARAPHRASE_TEMPLATES:
            out.append(
                tmpl.format(request=ep.request, correction=ep.correction)
            )
    return out


def collect(force: bool = False) -> Path:
    if HIDDENS_PATH.exists() and not force:
        print("already exists: %s (use --force to re-run)" % HIDDENS_PATH)
        return HIDDENS_PATH

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    episodes = build_dataset()
    utterances = _paraphrase_episodes(episodes)
    print("collecting hiddens for %d utterances (%d episodes x %d templates)" % (
        len(utterances), len(episodes), len(_PARAPHRASE_TEMPLATES)
    ))

    # Match the demo's CortexAgent + driver shape exactly so the
    # collected hiddens are drawn from the same LM (LFM2.5-1.2B-Instruct
    # Q4_K_M) and pooling (last_token_only=False) that perceive() uses.
    agent = CortexAgent(
        CortexAgentConfig(
            cortex=KVCortexConfig(d_cortex=64),
            driver=CVecDriverConfig(n_ctx=512, verbose=False, embedding=True),
        )
    )
    agent.boot()

    n_embd = agent.driver.n_embd
    hiddens = np.zeros((len(utterances), n_embd), dtype=np.float32)
    t0 = time.time()
    for i, utt in enumerate(utterances):
        # perceive() runs peek_embedding(last_token_only=False) -> d_embd,
        # then observe(hidden, correction_signal=...). We capture the
        # hidden vector by calling peek_embedding directly to mirror the
        # perceive path WITHOUT mutating the cortex's warm_state -- the
        # harness should not bleed correction logs into the basis.
        hidden = agent.driver.peek_embedding(utt, last_token_only=False)
        hiddens[i] = hidden
        if (i + 1) % 12 == 0 or i == len(utterances) - 1:
            print("  %3d/%d  (%.1fs)" % (i + 1, len(utterances), time.time() - t0))

    np.save(HIDDENS_PATH, hiddens, allow_pickle=False)
    meta = {
        "model_tag": "LFM2.5-1.2B-Instruct Q4_K_M",
        "d_embd": int(n_embd),
        "n_layers": int(agent.driver.n_layers),
        "n_hiddens": int(hiddens.shape[0]),
        "pooling": "last_token_only=False  (final-layer mean over prompt tokens)",
        "extraction_path": "LlamaCVecDriver.peek_embedding()",
        "source_pipelined_through": "experiments.cortex_agent.perceive",
        "templates": _PARAPHRASE_TEMPLATES,
        "n_episodes": int(len(episodes)),
        "utterances": utterances,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print("wrote %s  shape=%s  (%d bytes)" % (
        HIDDENS_PATH, hiddens.shape, HIDDENS_PATH.stat().st_size
    ))
    print("wrote %s" % META_PATH)

    # Verify the basis is non-trivial: the leading singular values of
    # the centered hiddens should have meaningful spread, not collapse
    # to a single dominant direction (which would mean the projector
    # degenerates to one basis vector).
    centered = hiddens - hiddens.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    print("top singular values: %s" % np.round(s[:8], 2))
    return HIDDENS_PATH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if .npy exists")
    args = ap.parse_args()
    collect(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())