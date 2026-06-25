"""LLM-backed NL<->Episode adapter.

Sits between the user (free-form NL) and the Oczy organism (which only
operates on structured Episodes per ``oczy.common.episode``).  The LM
itself is invoked only for IO-bound parsing/rendering; the organs and
the agent glue layer never see tokens.

Fail-soft contract:

* If the LM returns malformed JSON or unknown Episode keys,
  :meth:`nl_to_episode` returns a minimal valid Episode with the raw
  text in ``query`` and ``outcome="accepted"``.  The caller then treats
  the utterance as a query and routes it to ``agent.answer(query)``.
  A single bad LM call cannot crash the organism loop.
* If the LM's answer renders as empty/garbled, :meth:`episode_to_nl`
  falls back to a deterministic JSON-of-the-dict rendering.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from oczy.common.episode import EPISODE_FIELDS, validate_episode

log = logging.getLogger(__name__)


# The Pareto-optimal backend identified by bench_cross_backend.py on
# this host.  Hard-coded so callers that construct ``LanguageAdapter()``
# without arguments still get the right model.
DEFAULT_REPO_ID = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
DEFAULT_FILE_NAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

_PARSE_SYSTEM_PROMPT = """You are an instruction classifier. Classify ONE user utterance as either a plain question or a correction. Return a JSON object with these exact keys.

Classification rules:
- If the utterance is a plain question or request and contains NO correction wording: outcome="accepted" and "correction" AND "corrected_answer" MUST both be "".
- If the utterance contains correction wording ("No, X means Y", "X means Y", "X is Y", "X should be Y"): outcome="corrected". Put the whole correction sentence in "correction" and extract ONLY the NEW MEANING (the Y part, without the word being redefined) into "corrected_answer".

Examples:

User: "What is the weather today?"
{
  "query": "What is the weather today?",
  "answer": "",
  "correction": "",
  "corrected_answer": "",
  "outcome": "accepted",
  "source": "user_utterance"
}

User: "Update the user's profile. No, 'profile' means business vertical."
{
  "query": "Update the user's profile.",
  "answer": "",
  "correction": "No, 'profile' means business vertical.",
  "corrected_answer": "business vertical",
  "outcome": "corrected",
  "source": "user_utterance"
}

User: "Schedule the batch. No, 'batch' here means ML training batch."
{
  "query": "Schedule the batch.",
  "answer": "",
  "correction": "No, 'batch' here means ML training batch.",
  "corrected_answer": "ML training batch",
  "outcome": "corrected",
  "source": "user_utterance"
}

JSON keys:
{
  "query": "<the request part, or empty string if the utterance is only a correction>",
  "answer": "",
  "correction": "<raw correction sentence, or empty>",
  "corrected_answer": "<the new meaning only, without the redefined word; empty if no correction>",
  "outcome": "accepted" or "corrected",
  "source": "user_utterance"
}

Return ONLY the JSON. No prose, no markdown."""

_RENDER_SYSTEM_PROMPT = """You are the output layer for the Oczy cognitive agent.

You will be given an Episode as a JSON object.  Render it as natural
English.  Be concise: one sentence if possible, two short sentences
otherwise.  Do not echo the JSON.  Do not say what you are doing.
Just return the rendered English."""


@dataclass
class LanguageAdapterConfig:
    """LM loading and generation settings.

    Most defaults come from the cross-backend bench, which identified
    Q4_K_M as the Pareto winner on this host -- 38 tok/s sustained,
    1.6 GB peak RSS, 697 MB on disc.
    """

    repo_id: str = DEFAULT_REPO_ID
    file_name: str = DEFAULT_FILE_NAME
    n_threads: int = 4
    n_ctx: int = 1024
    n_gpu_layers: int = 0  # Force CPU; this host has no usable GPU.
    use_mmap: bool = True  # Keep disc out of RSS until pages are touched.
    use_mlock: bool = False
    verbose: bool = False

    # Generation knobs.
    temperature: float = 0.0    # Greedy for deterministic parsing.
    top_p: float = 1.0
    max_tokens_parse: int = 300   # Episodes are tiny JSON but the LM
                              # occasionally pads with whitespace.
    max_tokens_render: int = 80   # One sentence max in render path.


class LanguageAdapter:
    """Thin LFM2.5-1.2B Q4_K_M wrapper for NL<->Episode translation."""

    def __init__(self, config: LanguageAdapterConfig | None = None) -> None:
        self.config = config or LanguageAdapterConfig()
        self._llm: Any = None
        self._loaded: bool = False
        # Public counters for callers that want to instrument the
        # perception layer's reliability.
        self.n_parse_calls: int = 0
        self.n_parse_failures: int = 0
        self.n_render_calls: int = 0
        self.n_render_fallbacks: int = 0

    # ------------------------------------------------------------------
    # LM lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Lazy-load the GGUF on first use.

        Wraps ``Llama.from_pretrained`` so the rest of the codebase doesn't
        have to import llama_cpp at module-load time.  Idempotent.
        """
        if self._loaded:
            return
        # Import inside the method so ``import oczy.lm`` stays cheap and
        # so callers that never invoke the adapter don't pay the
        # llama-cpp-python import cost.
        from llama_cpp import Llama

        self._llm = Llama.from_pretrained(
            repo_id=self.config.repo_id,
            filename=self.config.file_name,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            verbose=self.config.verbose,
        )
        self._loaded = True

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._loaded = False

    # ------------------------------------------------------------------
    # NL -> Episode
    # ------------------------------------------------------------------
    def nl_to_episode(self, text: str) -> dict[str, Any]:
        """Parse a user utterance into a canonical Episode dict.

        Fail-soft: returns ``{"query": text, "outcome": "accepted", ...}``
        on any parsing error so the caller can route the utterance as a
        plain query and the organism loop never breaks.

        Args:
            text: A free-form natural-language user utterance, like
                  ``"Update the user profile. No, 'profile' means
                  business vertical."``.

        Returns:
            A dict conforming to :class:`oczy.common.episode.Episode`.
        """
        self.load()
        self.n_parse_calls += 1

        raw = self._chat(_PARSE_SYSTEM_PROMPT, text,
                         self.config.max_tokens_parse)
        cleaned = _strip_code_fence(raw)

        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("top-level is not a dict")
        except (json.JSONDecodeError, ValueError) as e:
            self.n_parse_failures += 1
            log.warning("LM parse failure on %r: %s; falling back to "
                        "raw-NL-as-query", text[:60], e)
            return _minimal_episode(text)

        # Reject any unknown keys -- they indicate schema drift.
        unknown = validate_episode(data)
        if unknown:
            log.warning("LM produced unknown Episode keys %s; stripping", unknown)
            data = {k: v for k, v in data.items() if k in EPISODE_FIELDS}

        # Fill in missing canonical fields with safe defaults so callers
        # never have to defensive-check after this point.
        for key, default in _DEFAULTS_FOR(text).items():
            data.setdefault(key, default)

        # Sanity check: if outcome says "corrected" but corrected_answer is
        # empty, downgrade outcome so the caller doesn't misroute.
        if data.get("outcome") == "corrected" and not data.get("corrected_answer"):
            log.warning("LM marked outcome=corrected but gave no "
                        "corrected_answer; downgrading to accepted")
            data["outcome"] = "accepted"

        # Sanity check: discard hallucinated corrected_answer from accepted
        # utterances, e.g., "Paris" leaking out of "What is the capital of
        # France?" (LM answered the factual question rather than classifying
        # the utterance).
        if data.get("outcome") == "accepted" and data.get("corrected_answer"):
            log.warning("LM marked outcome=accepted but gave a "
                        "spurious corrected_answer=%r; clearing",
                        data.get("corrected_answer"))
            data["corrected_answer"] = ""

        return data

    # ------------------------------------------------------------------
    # Episode -> NL
    # ------------------------------------------------------------------
    def episode_to_nl(self, episode: dict[str, Any]) -> str:
        """Render an Episode as natural English.

        Fail-soft: if the LM renders nothing useful, returns a
        deterministic ``"<key>: <value>; ..."`` flattening.
        """
        self.load()
        self.n_render_calls += 1

        # Only render canonical fields -- strip any internal tags the
        # LM wasn't meant to see (``id``, ``replay_count``, etc.).
        canonical = {k: episode[k] for k in EPISODE_FIELDS
                     if k in episode and k not in ("id", "replay_count")}
        body = json.dumps(canonical, indent=2, default=str)

        raw = self._chat(_RENDER_SYSTEM_PROMPT, body,
                         self.config.max_tokens_render)
        rendered = raw.strip()
        if not rendered or rendered.startswith("{"):
            # LM returned empty or echoed our JSON -- fall back.
            self.n_render_fallbacks += 1
            return _deterministic_render(canonical)
        return rendered

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=False,
        )
        return resp["choices"][0]["message"]["content"] or ""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE
)


def _strip_code_fence(text: str) -> str:
    """Strip a leading ```json ... ``` block if the LM added one.

    The Q4_K_M GGUF was tested in the cross-backend bench and usually
    returns bare JSON, but occasionally wraps in markdown -- defensive
    stripping avoids a JSONDecodeError class of failures.
    """
    m = _CODE_FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text.strip()


def _minimal_episode(text: str) -> dict[str, Any]:
    """Episode returned when the LM parse fails entirely.

    Caller can route this as a plain query without further inspection.
    """
    return {
        "query": text,
        "answer": "",
        "correction": "",
        "corrected_answer": "",
        "outcome": "accepted",
        "source": "user_utterance",
    }


def _DEFAULTS_FOR(text: str) -> dict[str, Any]:
    """Default values for any omitted canonical Episode field.

    ``text`` is the raw user utterance -- used so the query field is
    still populated sensibly when the LM omits it.
    """
    return {
        "query": text,
        "answer": "",
        "correction": "",
        "corrected_answer": "",
        "outcome": "accepted",
        "source": "user_utterance",
    }


def _deterministic_render(canonical: dict[str, Any]) -> str:
    """Fallback when the LM render is unusable.

    Produces a terse ``"query: X; correction: Y; corrected_answer: Z"``
    string so the user gets *something* readable, not silence.
    """
    parts = []
    for key in ("query", "correction", "corrected_answer"):
        val = canonical.get(key, "")
        if val:
            parts.append(f"{key}: {val}")
    return "; ".join(parts) if parts else "(empty episode)"