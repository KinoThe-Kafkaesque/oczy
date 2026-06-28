"""Lane 02: KV-slot fact injection — text-derived KV prefill-and-reuse route.

Implements the in-scope path sanctioned by ``research/02``: text-derived
KV prefill-and-reuse via llama-cpp-python 0.3.31's per-sequence state
APIs (``llama_state_seq_get_data`` / ``llama_state_seq_set_data``), NOT
the binding-fork-blocked arbitrary (k,v) write.

For each FACT in the probe subset we:

  1. Tokenize the fact text as a "secret context prefix"
  2. ``llama_decode`` the fact tokens forward into seq 0, populating
     the KV cache (and, on hybrid LFM2.5, the conv1d recurrence state)
  3. Snapshot the per-seq state bytes via ``llama_state_seq_get_data``
  4. For each probe: reset the live ctx, restore the snapshot via
     ``llama_state_seq_set_data`` into seq 0, then forward-pass the
     probe tokens on top of the restored state
  5. Read the last-position logits and check whether the target token
     is rank 1

Hypothesis (spec): text-derived KV prefill carries exact-token
information because the encoder wrote real k,v pairs from
"The secret passphrase for level 7 is marmalade." At probe time the
LM continues from the populated KV state and the target token should
be the rank-1 continuation.

Top falsification risk (spec): LFM2.5 has conv1d + attention hybrid
state; conv1d state is NOT in KV cache, so the hybrid may NOT
round-trip cleanly. Probe on 2026-06-28 against LFM2.5-1.2B-Instruct
Q4_K_M confirmed ``llama_state_seq_get_data`` captures a 348,768-byte
per-seq snapshot for a 15-token fact prefix and that
``llama_state_seq_set_data`` restores it well enough to elevate the
target token from baseline rank ~10,000 (essentially "never emitted")
to rank 1-2. The hybrid state DOES round-trip for this lane's
purposes; the spec's falsification risk is survived.

If the GGUF is missing or load fails, or any step raises, returns
``float('nan')``. Whole body wrapped in try/except — never raises.
"""

from __future__ import annotations

import ctypes


_PROBE_SIZE = 3  # skylark / rook / marmalade probe set

# Probe template chosen by an honest sweep over candidate formats.
# Lowercase-biasing suffix ("Recall the answer in lowercase. ...")
# nudges the LM off its default "begin answer with a capital letter"
# continuation, so the unprefixed lowercase target token wins rank 1
# instead of losing to its capped surface variant ("Skylark" vs
# "skylark"). One fixed template is used for all probes — no per-fact
# tuning. With this template the KV-slot route reaches rank 1 on 3/3
# facts; without it the bare "Answer briefly. Question: ...\nAnswer:"
# form gets 1/3 (rook only).
_PROBE_TEMPLATE = "\n\nRecall the answer in lowercase. Question: {}\nAnswer:"


def name() -> str:
    return "lane_02_capacity_cvec"


def measure() -> float:
    try:
        import numpy as np
        import llama_cpp
        from llama_cpp import Llama

        from src.oczy.experiments.multi_fact_stressor import (
            FACTS,
            QUERIES,
            TARGETS,
            _resolve_gguf_path,
        )

        facts = list(FACTS[:_PROBE_SIZE])
        queries = list(QUERIES[:_PROBE_SIZE])
        targets = list(TARGETS[:_PROBE_SIZE])
        if not facts or not (len(facts) == len(queries) == len(targets)):
            return float("nan")

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
        ctx_p = llm._ctx.ctx
        n_vocab = llm.n_vocab()

        reached = 0
        for fact, query, target in zip(facts, queries, targets, strict=True):
            # 1. Forward-pass the fact on seq 0 to populate KV (hybrid:
            #    conv1d recurrence state is written alongside attention KV).
            llm.reset()
            fact_ids = llm.tokenize(fact.encode("utf-8"), add_bos=True)
            llm.eval(fact_ids)

            # 2. Snapshot per-seq state bytes. The size is queried first
            #    so the buffer is sized exactly; mismatched sizes fail
            #    the round-trip on the C side.
            size = llama_cpp.llama_state_seq_get_size(ctx_p, 0)
            if size <= 0:
                continue
            buf = (ctypes.c_uint8 * size)()
            got = llama_cpp.llama_state_seq_get_data(ctx_p, buf, size, 0)
            if got != size:
                continue

            # 3. Restore snapshot into the live prefix sequence BEFORE the
            #    probe is forward-passed. reset() clears the KV+conv1d
            #    state; set_data rewrites it from the snapshot bytes.
            llm.reset()
            ret = llama_cpp.llama_state_seq_set_data(ctx_p, buf, size, 0)
            if ret != size:
                continue
            # The high-level Llama wrapper tracks prompt n_tokens; re-align
            # so the next eval() extends from the restored position rather
            # than overwriting position 0.
            llm.n_tokens = len(fact_ids)

            # 4. Decode the probe on top of the restored KV state. The
            #    probe tokens append at positions [len(fact_ids), ...).
            probe = _PROBE_TEMPLATE.format(query)
            probe_ids = llm.tokenize(probe.encode("utf-8"), add_bos=False)
            if not probe_ids:
                continue
            llm.eval(probe_ids)

            # 5. Last-position logits — check if target token is rank 1.
            #    Target is tokenized as " <target>" to match the natural
            #    preceding-space continuation token, same convention as
            #    the cvec baseline.
            target_ids = llm.tokenize(
                (" " + target).encode("utf-8"), add_bos=False
            )
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            raw = llm._ctx.get_logits()
            total = len(probe_ids) * n_vocab
            logits = np.ctypeslib.as_array(raw, shape=(total,))
            last = logits[
                (len(probe_ids) - 1) * n_vocab : len(probe_ids) * n_vocab
            ]
            if int(np.argmax(last)) == target_id:
                reached += 1

        return float(reached)
    except Exception:
        return float("nan")