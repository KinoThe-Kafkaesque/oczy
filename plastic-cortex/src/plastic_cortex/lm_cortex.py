"""Trainable character-level language-model cortex.

This module implements a tiny NumPy-only recurrent language model that can
replace the toy word-association :class:`~plastic_cortex.cortex.PlasticCortex`.
It exposes the same public surface area:

    answer(query, max_tokens=...) -> str
    correct(correction_text, expected_answer) -> None
    reset_state()
    status() -> dict

plus training and serialization helpers.
"""

from __future__ import annotations

import math
import collections
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from .char_tokenizer import CharTokenizer


def _xavier_uniform(rng: np.random.RandomState, rows: int, cols: int) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = math.sqrt(6.0 / (rows + cols))
    return rng.uniform(-limit, limit, size=(rows, cols)).astype(np.float32)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    return shifted - np.log(np.sum(np.exp(shifted)))


def sample_token(logits: np.ndarray, temperature: float, rng: np.random.RandomState) -> int:
    """Sample one token from *logits*.

    * ``temperature == 0`` returns the argmax.
    * ``temperature > 0`` returns a softmax-normalized sample.
    """
    if temperature == 0.0:
        return int(np.argmax(logits))
    scaled = logits / max(temperature, 1e-8)
    probs = np.exp(log_softmax(scaled))
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0.0:
        return int(np.argmax(logits))
    probs /= total
    return int(rng.choice(probs.shape[0], p=probs))


class FastWeightLM:
    """Fast-weight adapter that stores token -> logit boosts.

    Each update records an association from a trigger character (the last
    character of the trigger string) to a target character.  At generation time
    the boosts for recent context tokens are added to the output logits,
    increasing the probability of the target characters without overwriting the
    slow base weights.
    """

    def __init__(self, vocab_size: int, context_window: int = 8) -> None:
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.boosts = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    def _char_id(self, token: str, tokenizer: CharTokenizer) -> int:
        if not token:
            return tokenizer.unk_id
        ch = token[-1]  # last character is the trigger/target character
        return tokenizer._token_to_id.get(ch, tokenizer.unk_id)

    def update(
        self,
        tokenizer: CharTokenizer,
        trigger: str,
        target: str,
        strength: float = 1.0,
    ) -> None:
        """Increment the boost from *trigger*'s last char to *target*'s char."""
        trigger_id = self._char_id(trigger, tokenizer)
        target_id = self._char_id(target, tokenizer)
        self.boosts[trigger_id, target_id] += float(strength)

    def boost(
        self,
        logits: np.ndarray,
        tokenizer: CharTokenizer,
        context: str | None,
    ) -> np.ndarray:
        """Add relevant stored boosts to *logits* based on recent *context*."""
        if not context:
            return logits
        boosted = np.array(logits, dtype=np.float32, copy=True)
        window = context[-self.context_window :]
        for ch in window:
            token_id = tokenizer._token_to_id.get(ch)
            if token_id is not None:
                boosted += self.boosts[token_id]
        return boosted

    def reset_state(self) -> None:
        """Clear all stored boosts."""
        self.boosts.fill(0.0)

    def state_snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the fast-weight matrix."""
        return {
            "vocab_size": self.vocab_size,
            "context_window": self.context_window,
            "boosts": self.boosts.tolist(),
        }

    def apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore from a snapshot produced by :meth:`state_snapshot`."""
        self.vocab_size = snapshot["vocab_size"]
        self.context_window = snapshot["context_window"]
        self.boosts = np.array(snapshot["boosts"], dtype=np.float32)


class LMPlasticCortex:
    """Small trainable character-level RNN cortex.

    The model contains:

    * an embedding matrix ``E`` (vocab_size x hidden_dim)
    * an input-to-hidden projection ``W_xh`` (vocab_size x hidden_dim)
    * a hidden-to-hidden recurrent matrix ``W_hh`` (hidden_dim x hidden_dim)
    * output projection ``W_vocab`` (hidden_dim x vocab_size)

    plus biases, a character tokenizer, and a :class:`FastWeightLM` adapter.

    Args:
        config: Optional configuration dictionary.  Recognized keys:
            ``hidden_dim`` (default 64), ``seed`` (default 0),
            ``vocab_size`` (default from :class:`CharTokenizer`).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.hidden_dim = int(self.config.get("hidden_dim", 64))
        self.seed = int(self.config.get("seed", 0))

        self.tokenizer = CharTokenizer()
        # Use the configured vocabulary size when provided; otherwise fall back
        # to the default tokenizer size. Callers that fit a larger tokenizer
        # should pass its size explicitly in the config.
        self.vocab_size = int(
            self.config.get("vocab_size", self.tokenizer.vocab_size)
        )

        self.fast = FastWeightLM(self.vocab_size)
        self._rng = np.random.RandomState(self.seed)

        # Xavier-ish initialization for all weight matrices.
        self.E = _xavier_uniform(self._rng, self.vocab_size, self.hidden_dim)
        self.W_xh = _xavier_uniform(self._rng, self.vocab_size, self.hidden_dim)
        self.W_hh = _xavier_uniform(self._rng, self.hidden_dim, self.hidden_dim)
        self.b_h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_vocab = _xavier_uniform(self._rng, self.hidden_dim, self.vocab_size)
        self.b_vocab = np.zeros(self.vocab_size, dtype=np.float32)

        # Transient generation / training state.
        self._h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        # Observation statistics for curiosity signals.
        self._seen_tokens: collections.Counter[int] = collections.Counter()
        self._seen_bigrams: collections.Counter[tuple[int, int]] = collections.Counter()
        self._token_total: int = 0
        self._recent_novel: collections.deque[tuple[str, float]] = collections.deque(maxlen=100)
        self.correction_count = 0

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Migrate pickled objects created before curiosity signals existed."""
        self.__dict__.update(state)
        # Ensure observation statistics exist for unpickled old models.
        if "_seen_tokens" not in self.__dict__:
            self._seen_tokens: collections.Counter[int] = collections.Counter()
        if "_seen_bigrams" not in self.__dict__:
            self._seen_bigrams: collections.Counter[tuple[int, int]] = collections.Counter()
        if "_token_total" not in self.__dict__:
            self._token_total = 0
        if "_recent_novel" not in self.__dict__:
            self._recent_novel = collections.deque(maxlen=100)

    def grow(self, new_hidden_dim: int) -> LMPlasticCortex:
        """Return a larger-capacity cortex with preserved behavioral state.

        The current slow weights are expanded by padding with small random
        values, while fast weights, observation statistics, and the tokenizer
        are copied over unchanged.  This lets the organism respond to training
        difficulty by increasing capacity without losing what it already knows.
        """
        if new_hidden_dim <= self.hidden_dim:
            raise ValueError(
                f"new_hidden_dim ({new_hidden_dim}) must exceed "
                f"current hidden_dim ({self.hidden_dim})"
            )

        new_config = dict(self.config)
        new_config["hidden_dim"] = new_hidden_dim
        new_config["vocab_size"] = self.vocab_size
        new_config["seed"] = self.seed

        child = LMPlasticCortex(new_config)
        child.tokenizer = self.tokenizer
        child.fast = self.fast
        child._seen_tokens = self._seen_tokens.copy()
        child._seen_bigrams = self._seen_bigrams.copy()
        child._token_total = self._token_total
        child._recent_novel = collections.deque(self._recent_novel, maxlen=100)
        child.correction_count = self.correction_count

        rng = self._rng
        pad_E = _xavier_uniform(rng, self.vocab_size, new_hidden_dim - self.hidden_dim)
        child.E = np.concatenate([self.E, pad_E], axis=1)

        pad_W_xh = _xavier_uniform(rng, self.vocab_size, new_hidden_dim - self.hidden_dim)
        child.W_xh = np.concatenate([self.W_xh, pad_W_xh], axis=1)

        pad_top = _xavier_uniform(rng, self.hidden_dim, new_hidden_dim - self.hidden_dim)
        W_hh_old = np.concatenate([self.W_hh, pad_top], axis=1)
        pad_bottom = _xavier_uniform(rng, new_hidden_dim - self.hidden_dim, new_hidden_dim)
        child.W_hh = np.concatenate([W_hh_old, pad_bottom], axis=0)

        child.b_h[: self.hidden_dim] = self.b_h

        pad_W_vocab_rows = _xavier_uniform(rng, new_hidden_dim - self.hidden_dim, self.vocab_size)
        child.W_vocab = np.concatenate([self.W_vocab, pad_W_vocab_rows], axis=0)
        child.b_vocab = self.b_vocab.copy()

        return child
    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _update_seen(self, tokens: list[int]) -> None:
        """Update token and bigram observation statistics."""
        if not tokens:
            return
        self._seen_tokens.update(tokens)
        self._token_total += len(tokens)
        for i in range(len(tokens) - 1):
            self._seen_bigrams[(tokens[i], tokens[i + 1])] += 1

    def _record_recent_novel(self, tokens: list[int]) -> None:
        """Remember rare tokens/bigrams observed during this session."""
        if self._token_total == 0 or not tokens:
            return
        total = max(1, self._token_total)
        for tok in tokens:
            if self._seen_tokens[tok] <= 2:
                freq = self._seen_tokens[tok] / total
                score = -math.log10(freq + 1e-12)
                self._recent_novel.append((self.tokenizer.decode([tok]), score))
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            if self._seen_bigrams[bigram] <= 2:
                freq = self._seen_bigrams[bigram] / total
                score = -math.log10(freq + 1e-12)
                self._recent_novel.append((self.tokenizer.decode(list(bigram)), score))
    def _reset_hidden(self) -> None:
        self._h = np.zeros(self.hidden_dim, dtype=np.float32)

    def _step(self, token_id: int) -> np.ndarray:
        """Advance the recurrent state by one token and return logits."""
        if self._h is None:
            self._reset_hidden()
        x_proj = self.E[token_id] + self.W_xh[token_id]
        self._h = np.tanh(x_proj + self._h @ self.W_hh + self.b_h)
        return self._h @ self.W_vocab + self.b_vocab

    def _forward_string(self, text: str) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
        """Feed *text* through the network and return tokens, hidden states, logits."""
        # Each training string is treated as a fresh sequence.
        self._reset_hidden()
        tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_id]
        hiddens: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        for tok in tokens:
            hiddens.append(np.array(self._h, copy=True))
            logit = self._step(tok)
            logits.append(logit)
        return tokens, hiddens, logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(self, query: str, max_tokens: int = 100, temperature: float = 1.0, stop_at_eos: bool = True) -> str:
        """Generate a continuation for *query*.

        Args:
            query: Prompt text to condition on.
            max_tokens: Maximum number of new characters to generate.
            temperature: Sampling temperature; 0 means greedy argmax.
            stop_at_eos: Stop generation when the ``<EOS>`` token is sampled.
        """
        self._reset_hidden()

        # Condition the hidden state on the query.
        query_ids = self.tokenizer.encode(query)
        for tok in query_ids:
            self._step(tok)

        generated_ids: list[int] = []
        context = query
        for _ in range(max_tokens):
            logits = self._h @ self.W_vocab + self.b_vocab
            logits = self.fast.boost(logits, self.tokenizer, context)
            next_id = sample_token(logits, temperature, self._rng)
            if stop_at_eos and next_id == self.tokenizer.eos_id:
                break
            generated_ids.append(next_id)
            context = query + self.tokenizer.decode(generated_ids)
            self._step(next_id)

        full_ids = query_ids + generated_ids
        self._update_seen(full_ids)
        self._record_recent_novel(full_ids)
        return self.tokenizer.decode(generated_ids)

    def uncertainty(self, query: str, max_tokens: int = 40) -> float:
        """Return the mean per-step entropy of the model continuation.

        The hidden state is reset, the query is fed in, and *max_tokens*
        characters are generated with a small non-zero temperature.  For each
        step the entropy of the temperature-scaled softmax is computed; the
        returned value is the mean entropy over generated tokens.
        """
        self._reset_hidden()
        query_ids = self.tokenizer.encode(query)
        for tok in query_ids:
            self._step(tok)

        temperature = 0.5
        entropies: list[float] = []
        for _ in range(max_tokens):
            logits = self._h @ self.W_vocab + self.b_vocab
            scaled = logits / max(temperature, 1e-12)
            log_p = log_softmax(scaled)
            probs = np.exp(log_p)
            ent = -np.sum(probs * log_p)
            entropies.append(float(ent))
            next_id = sample_token(scaled, 1.0, self._rng)
            self._step(next_id)

        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def novelty(self, query: str) -> float:
        """Return a novelty score in [0, 1] relative to observed statistics."""
        tokens = self.tokenizer.encode(query)
        if not tokens:
            return 0.0

        total = max(1, self._token_total)
        eps = 1e-12
        scores: list[float] = []
        for tok in tokens:
            freq = self._seen_tokens.get(tok, 0) / total
            scores.append(-math.log10(freq + eps))
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            freq = self._seen_bigrams.get(bigram, 0) / total
            scores.append(-math.log10(freq + eps))

        raw = sum(scores) / len(scores)
        return 1.0 / (1.0 + raw)

    def wonder(self, top_k: int = 5) -> dict[str, Any]:
        """Return curiosity summary: most uncertain bigrams and recent novel items."""
        # Bigrams observed fewest times are the most "uncertain"/novel.
        rarest_bigrams = sorted(self._seen_bigrams.items(), key=lambda kv: kv[1])[:top_k]
        most_uncertain: list[list[Any]] = []
        for (a, b), count in rarest_bigrams:
            ngram = self.tokenizer.decode([a, b])
            most_uncertain.append([ngram, float(count)])

        recent_novel: list[list[Any]] = [
            [ngram, float(score)] for ngram, score in list(self._recent_novel)[-top_k:]
        ]

        suggested_question = "Can you explain something more?"
        if recent_novel:
            suggested_question = f"Can you explain {recent_novel[-1][0]} more?"
        elif most_uncertain:
            suggested_question = f"Can you explain {most_uncertain[0][0]} more?"

        return {
            "most_uncertain": most_uncertain,
            "most_novel_recent": recent_novel,
            "suggested_question": suggested_question,
        }
    def correct(self, correction_text: str, expected_answer: str) -> None:
        """Apply an explicit correction through the fast-weight adapter.

        Every character observed in *correction_text* is given a small boost
        toward the first character of *expected_answer*, and the expected
        answer is linked as a Markov chain so that once generation enters the
        target string it is likely to continue through it.  If
        *expected_answer* is empty, a naive ``"should be"`` regex is tried.
        """
        if not expected_answer:
            match = re.search(r"should\s+be\s+(.+?)(?:[.!?]|$)", correction_text, re.IGNORECASE)
            if match:
                expected_answer = match.group(1).strip()
        if not expected_answer:
            return

        # Cues: small nudges from every correction-text character toward the
        # first character of the desired output.  The space/last punctuation in
        # the correction will fire when the user later supplies a similar
        # context (e.g. "say ").
        cue_strength = 0.5
        seen_cues = dict.fromkeys(correction_text)
        for ch in seen_cues:
            self.fast.update(self.tokenizer, ch, expected_answer[0], strength=cue_strength)

        # Strong chain: once the target string starts, force the next char.
        # The boost is made large relative to the random Xavier-initialized
        # base logits so that the corrected continuation is reliably preferred.
        for i in range(len(expected_answer) - 1):
            self.fast.update(
                self.tokenizer,
                expected_answer[i],
                expected_answer[i + 1],
                strength=50.0 + 5.0 * i,
            )
        self.correction_count += 1

    def train_step(
        self, text: str, lr: float = 0.01, grad_clip: float | None = None
    ) -> float:
        """One unrolled BPTT step on *text* using SGD.

        Returns the mean cross-entropy loss over the sequence.

        Args:
            text: Training sequence.
            lr: SGD learning rate.
            grad_clip: If provided, clip total gradient norm to this value.
        """
        tokens, hiddens, logits = self._forward_string(text)
        T = len(tokens) - 1  # number of next-token predictions
        if T <= 0:
            return 0.0

        loss = 0.0
        probs = np.zeros((T, self.vocab_size), dtype=np.float64)
        for t in range(T):
            log_p = log_softmax(logits[t])
            target = tokens[t + 1]
            probs[t] = np.exp(log_p)
            loss -= log_p[target]
        loss /= T

        # Accumulators for gradients.
        dE = np.zeros_like(self.E)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_vocab = np.zeros_like(self.W_vocab)
        db_vocab = np.zeros_like(self.b_vocab)

        dh_next = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            target = tokens[t + 1]
            d_logit = probs[t].copy()
            d_logit[target] -= 1.0
            d_logit /= float(T)

            h_t = hiddens[t]
            dW_vocab += np.outer(h_t, d_logit)
            db_vocab += d_logit
            dh = d_logit @ self.W_vocab.T + dh_next
            dtanh = dh * (1.0 - h_t * h_t)

            token_id = tokens[t]
            dE[token_id] += dtanh
            dW_xh[token_id] += dtanh
            if t > 0:
                h_prev = hiddens[t - 1]
            else:
                h_prev = np.zeros(self.hidden_dim, dtype=np.float32)
            dW_hh += np.outer(h_prev, dtanh)
            db_h += dtanh
            dh_next = dtanh @ self.W_hh.T

        # Gradient clipping to prevent recurrent explosions.
        max_grad_norm = grad_clip if grad_clip is not None else 5.0
        for grad in (dE, dW_xh, dW_hh, db_h, dW_vocab, db_vocab):
            norm = float(np.sqrt(np.sum(grad * grad)) + 1e-12)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm

        # SGD update.
        self.E -= lr * dE
        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.b_h -= lr * db_h
        self.W_vocab -= lr * dW_vocab
        self.b_vocab -= lr * db_vocab

        return float(loss)

    def status(self) -> dict[str, Any]:
        """Return a serializable status dictionary."""
        param_bytes = int(
            self.E.nbytes
            + self.W_xh.nbytes
            + self.W_hh.nbytes
            + self.b_h.nbytes
            + self.W_vocab.nbytes
            + self.b_vocab.nbytes
        )
        fast_count = int(np.count_nonzero(self.fast.boosts))
        return {
            "type": "LMPlasticCortex",
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "param_bytes": param_bytes,
            "fast_weights_count": fast_count,
            "corrections": self.correction_count,
        }

    def reset_state(self) -> None:
        """Clear recurrent state and fast weights."""
        self._reset_hidden()
        self.fast.reset_state()
        self.correction_count = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle the complete model state to *path*."""
        Path(path).write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "LMPlasticCortex":
        """Load a model previously saved with :meth:`save`."""
        obj = pickle.loads(Path(path).read_bytes())
        if not isinstance(obj, cls):
            raise TypeError(f"loaded object is not a {cls.__name__}")
        return obj
