"""Numba-accelerated kernels for LMPlasticCortex.

If numba is not installed, the module exposes equivalent (slower) pure-NumPy
implementations so the rest of the codebase continues to work.

All kernels avoid Numba's typed linear-algebra operators (``@`` / ``np.dot``)
to keep from pulling in scipy as a runtime dependency.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs) -> Callable:  # type: ignore[misc]
        """No-op decorator fallback when numba is unavailable."""

        def _decorator(func: Callable) -> Callable:
            return func

        if args and callable(args[0]):
            return args[0]
        return _decorator


@njit(cache=True)
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last 1-D axis."""
    n = logits.shape[0]
    max_logit = logits[0]
    for i in range(1, n):
        v = logits[i]
        if v > max_logit:
            max_logit = v
    out = np.empty(n, dtype=logits.dtype)
    exp_sum = 0.0
    for i in range(n):
        out[i] = logits[i] - max_logit
        exp_sum += np.exp(out[i])
    log_sum = np.log(exp_sum)
    for i in range(n):
        out[i] -= log_sum
    return out


@njit(cache=True)
def _rnn_step(
    token_id: int,
    E: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Single token forward recurrence matching original orientation h @ W_hh."""
    hidden_dim = b_h.shape[0]
    x = E[token_id] + W_xh[token_id]
    h_next = np.empty(hidden_dim, dtype=np.float32)
    for d in range(hidden_dim):
        s = b_h[d] + x[d]
        for dd in range(hidden_dim):
            s += h[dd] * W_hh[dd, d]
        h_next[d] = np.tanh(s)
    return h_next


@njit(cache=True)
def _rnn_forward(
    tokens: np.ndarray,
    E: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_vocab: np.ndarray,
    b_vocab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unroll an entire sequence and return hiddens[0:T] and logits[0:T-1]."""
    T = tokens.shape[0] - 1
    hidden_dim = W_hh.shape[0]
    vocab_size = W_vocab.shape[1]
    hiddens = np.empty((T, hidden_dim), dtype=np.float32)
    logits = np.empty((T, vocab_size), dtype=np.float32)
    h = np.zeros(hidden_dim, dtype=np.float32)
    for t in range(T):
        hiddens[t] = h
        # logits[t] predicts tokens[t + 1]
        for v in range(vocab_size):
            s = b_vocab[v]
            for d in range(hidden_dim):
                s += W_vocab[d, v] * h[d]
            logits[t, v] = s
        token_id = tokens[t]
        h = _rnn_step(token_id, E, W_xh, W_hh, b_h, h)
    return hiddens, logits


@njit(cache=True)
def _rnn_backward(
    tokens: np.ndarray,
    hiddens: np.ndarray,
    logits: np.ndarray,
    W_vocab: np.ndarray,
    W_hh: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Back-propagate through the unrolled RNN and return loss + gradients."""
    T = tokens.shape[0] - 1
    hidden_dim = W_hh.shape[0]
    vocab_size = W_vocab.shape[1]

    dE = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    dW_xh = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    dW_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    db_h = np.zeros(hidden_dim, dtype=np.float32)
    dW_vocab = np.zeros((hidden_dim, vocab_size), dtype=np.float32)
    db_vocab = np.zeros(vocab_size, dtype=np.float32)

    probs = np.empty((T, vocab_size), dtype=np.float64)
    loss = 0.0
    for t in range(T):
        log_p = _log_softmax(logits[t])
        target = tokens[t + 1]
        psum = 0.0
        for v in range(vocab_size):
            e = np.exp(log_p[v])
            probs[t, v] = e
            psum += e
        for v in range(vocab_size):
            probs[t, v] /= psum
        loss -= log_p[target]
    loss /= float(T)

    dh_next = np.zeros(hidden_dim, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        target = tokens[t + 1]
        d_logit = probs[t].copy()
        d_logit[target] -= 1.0
        inv_T = 1.0 / float(T)
        for v in range(vocab_size):
            d_logit[v] *= inv_T

        h_t = hiddens[t]
        for d in range(hidden_dim):
            for v in range(vocab_size):
                dW_vocab[d, v] += h_t[d] * d_logit[v]
        for v in range(vocab_size):
            db_vocab[v] += d_logit[v]

        dh = np.zeros(hidden_dim, dtype=np.float32)
        for d in range(hidden_dim):
            for v in range(vocab_size):
                dh[d] += W_vocab[d, v] * d_logit[v]
            dh[d] += dh_next[d]

        dtanh = np.empty(hidden_dim, dtype=np.float32)
        for d in range(hidden_dim):
            dtanh[d] = dh[d] * (1.0 - h_t[d] * h_t[d])

        token_id = tokens[t]
        for d in range(hidden_dim):
            dE[token_id, d] += dtanh[d]
            dW_xh[token_id, d] += dtanh[d]

        if t > 0:
            h_prev = hiddens[t - 1]
        else:
            h_prev = np.zeros(hidden_dim, dtype=np.float32)
        for d in range(hidden_dim):
            db_h[d] += dtanh[d]
            for dd in range(hidden_dim):
                dW_hh[dd, d] += h_prev[dd] * dtanh[d]

        for d in range(hidden_dim):
            s = 0.0
            for dd in range(hidden_dim):
                s += W_hh[dd, d] * dtanh[dd]
            dh_next[d] = s

    return loss, dE, dW_xh, dW_hh, db_h, dW_vocab, db_vocab


@njit(cache=True)
def _sample_token(logits: np.ndarray, temperature: float, rng_state: np.ndarray) -> int:
    """Sample one token from scaled logits using a tiny xorshift RNG.

    Args:
        logits: 1-D array of unnormalized log probabilities.
        temperature: softmax temperature; 0 means argmax.
        rng_state: mutable int64 array of length 1 used as RNG state.
    """
    n = logits.shape[0]
    if temperature == 0:
        best = 0
        best_val = logits[0]
        for i in range(1, n):
            if logits[i] > best_val:
                best_val = logits[i]
                best = i
        return best

    scaled = logits / temperature
    log_p = _log_softmax(scaled)

    max_log_p = log_p[0]
    for i in range(1, n):
        if log_p[i] > max_log_p:
            max_log_p = log_p[i]

    probs = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        e = np.exp(log_p[i] - max_log_p)
        probs[i] = e
        total += e

    # xorshift64* draw, normalise to [0, 1)
    s = rng_state[0]
    s = np.int64(s ^ np.int64(s >> 17))
    s = np.int64(s ^ np.int64(s << 25))
    s = np.int64(s * np.int64(2685821657736338717))
    rng_state[0] = s
    r = np.float64(s >> 11) / np.float64(1 << 21)
    if r < 0:
        r = -r
    if r >= 1:
        r = 0.0

    cdf = 0.0
    for i in range(n):
        cdf += probs[i] / total
        if r < cdf:
            return i
    return n - 1
