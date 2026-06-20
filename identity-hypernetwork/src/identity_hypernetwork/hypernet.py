"""Core implementation of the Identity Hypernetwork.

A tiny hypernetwork maps a compact latent identity into adapter score deltas.
The four identity components (user, domain, style, mistakes) are stored in
:class:`IdentityLatents` and projected through a learned linear layer.

This is intentionally minimal: NumPy only, small fixed concept vocabulary, and
a simple Hebbian-style update that moves the relevant identity slice toward
the weight vector of the target concept so that future adapter scores shift.
"""

from __future__ import annotations

import pickle

import numpy as np

from .latents import IdentityLatents


# Fixed concept vocabulary for the first prototype.  Each string maps to one
# output score in the generated adapter.
CONCEPT_VOCABULARY: list[str] = [
    "profile",
    "account",
    "business",
    "vertical",
    "project",
    "domain",
    "formal",
    "casual",
    "concise",
    "verbose",
    "error",
    "bug",
    "mistake",
    "correct",
]

# Stopwords that ``_extract_first_concept`` will refuse to auto-register as new
# concepts.  Kept as a hardcoded inline list (no cross-organ import) so the
# auto-grow path can stay self-contained.
_AUTO_GROW_STOPWORDS: frozenset[str] = frozenset(
    {"the", "is", "no", "here", "means", "and", "or", "a", "an"}
)


class IdentityHypernetwork:
    """Compact latent identity vectors generated into adapter score deltas."""

    _Z_FIELDS: tuple[str, str, str, str] = (
        "z_user",
        "z_domain",
        "z_style",
        "z_mistakes",
    )

    def __init__(
        self,
        latent_dim: int = 8,
        seed: int = 0,
        learning_rate: float = 0.1,
    ) -> None:
        """Create the hypernetwork.

        Args:
            latent_dim: dimensionality of each of the four identity vectors.
            seed: random seed for the tiny projection matrix.
            learning_rate: step size used when a lesson updates an identity slice.
        """
        self.latents = IdentityLatents(dim=latent_dim)
        self.rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim
        self.input_dim = 4 * latent_dim
        self.concepts = list(CONCEPT_VOCABULARY)
        self.concept_index = {concept: i for i, concept in enumerate(self.concepts)}
        self.output_dim = len(self.concepts)
        # Small random projection so different concepts receive different scores.
        scale = 1.0 / np.sqrt(self.input_dim)
        self.W = self.rng.standard_normal((self.output_dim, self.input_dim)) * scale
        self.lr = learning_rate

    def generate_adapters(self) -> dict[str, dict[str, float]]:
        """Return adapter score deltas derived from the current identity latent.

        Returns a dictionary with a single key ``concept_scores`` mapping each
        known concept to a scalar delta.
        """
        z = self.latents.to_array()
        scores = self.W @ z
        return {"concept_scores": {concept: float(scores[i]) for i, concept in enumerate(self.concepts)}}

    def update_identity(self, lesson: dict) -> None:
        """Apply a learning signal to the relevant identity component.

        ``lesson`` must contain at least:

        - ``source``: one of ``user_correction`` / ``user`` -> ``z_user``,
          ``domain`` / ``project`` -> ``z_domain``,
          ``style`` / ``tone`` -> ``z_style``,
          ``mistake`` / ``error`` / ``bug`` -> ``z_mistakes``.
        - ``correct_label`` (or ``token``): text from which the target concept is
          extracted.  The first known concept found in ``correct_label`` (or
          ``token``) is the one whose score will be increased.
        """
        source = str(lesson.get("source", "user_correction")).lower()
        label_text = str(lesson.get("correct_label", lesson.get("token", ""))).lower()

        z_field = self._resolve_source(source)
        target_concept = self._extract_first_concept(label_text)
        if target_concept is None or z_field is None:
            return

        target_idx = self.concept_index[target_concept]
        # Gradient of score[target_idx] with respect to the full identity vector
        # is W[target_idx].  Moving the relevant slice in that direction raises the
        # target score.
        direction = self.W[target_idx]
        start, end = self._field_slice(z_field)
        slice_dir = direction[start:end]
        norm = float(np.linalg.norm(slice_dir))
        if norm == 0:
            return
        # Normalised step keeps updates stable regardless of the random matrix.
        step = self.lr * slice_dir / norm
        updated = getattr(self.latents, z_field).copy()
        updated += step
        setattr(self.latents, z_field, updated)

    def status(self) -> dict:
        """Return a serialisable status snapshot."""
        return {
            "project": "identity_hypernetwork",
            "ready": True,
            "latent_dim": self.latent_dim,
            "num_concepts": self.output_dim,
            "latents": self.latents.to_dict(),
            "serialized_bytes": len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)),
            "record_count": len(self.concepts),
        }

    def grow_vocab(self, new_concepts: list[str]) -> None:
        """Add new concepts to the vocabulary, extending ``W`` with one fresh row each.

        Each candidate is validated (lowercased, alphanumeric-only, non-empty,
        and not already in ``self.concepts``) before a single row of small
        random init is appended to ``W``.  The init uses the same
        ``1.0 / sqrt(input_dim)`` scale convention as ``__init__``.  The
        ``concepts`` list, ``concept_index`` mapping, and ``output_dim`` are
        updated in place.

        This method is also invoked on the fly from ``_extract_first_concept``
        when ``update_identity`` encounters a label token that is not in the
        initial 14-token vocabulary, so the closed-vocab blocker described in
        H3 is removed without every caller needing to pre-register concepts.
        Auto-growth is gated (alnum, length >= 3, not in
        ``_AUTO_GROW_STOPWORDS``) to keep vocab inflation bounded and to avoid
        registering junk tokens such as ``the`` or ``a``.
        """
        scale = 1.0 / np.sqrt(self.input_dim)
        for raw in new_concepts:
            clean = "".join(ch for ch in str(raw) if ch.isalnum()).lower()
            if not clean or clean in self.concept_index:
                continue
            new_row = self.rng.standard_normal((1, self.input_dim)) * scale
            self.W = np.concatenate([self.W, new_row], axis=0)
            self.concept_index[clean] = self.output_dim
            self.concepts.append(clean)
            self.output_dim += 1

    def _resolve_source(self, source: str) -> str | None:
        mapping: dict[str, str] = {
            "user_correction": "z_user",
            "user": "z_user",
            "profile": "z_user",
            "domain": "z_domain",
            "project": "z_domain",
            "style": "z_style",
            "tone": "z_style",
            "communication": "z_style",
            "mistake": "z_mistakes",
            "error": "z_mistakes",
            "bug": "z_mistakes",
            "failure": "z_mistakes",
        }
        return mapping.get(source)

    def _extract_first_concept(self, text: str) -> str | None:
        words = text.split()
        for word in words:
            clean = "".join(ch for ch in word if ch.isalnum()).lower()
            if not clean:
                continue
            if clean in self.concept_index:
                return clean
            # Auto-grow path: register an unknown token as a new concept on
            # the fly so ``update_identity`` learns from curriculum labels
            # that were not in the initial 14-token vocab.  The filter
            # (alnum is implicit from cleaning; length >= 3; not a stopword)
            # bounds inflation so common words and noise tokens (``the``,
            # ``a``, ``is``) are not registered.
            if len(clean) >= 3 and clean not in _AUTO_GROW_STOPWORDS:
                self.grow_vocab([clean])
                return clean
        return None

    def grow(self, new_latent_dim: int) -> "IdentityHypernetwork":
        """Return a larger-capacity hypernetwork preserving learned latents.

        Each latent vector is zero-padded and the projection matrix ``W`` is
        expanded with small random columns matching the original initializer.
        """
        if new_latent_dim <= self.latent_dim:
            raise ValueError(
                f"new_latent_dim ({new_latent_dim}) must exceed "
                f"current latent_dim ({self.latent_dim})"
            )

        child = IdentityHypernetwork(
            latent_dim=new_latent_dim, seed=self.rng.integers(2**31), learning_rate=self.lr
        )
        # Restore deterministic RNG state so new columns use same distribution.
        child.rng = self.rng
        child.latents = self.latents.grow(new_latent_dim)
        child.concepts = list(self.concepts)
        child.concept_index = dict(self.concept_index)
        child.output_dim = self.output_dim

        new_input_dim = 4 * new_latent_dim
        new_cols = new_input_dim - self.input_dim
        scale = 1.0 / np.sqrt(new_input_dim)
        pad = self.rng.standard_normal((self.output_dim, new_cols)) * scale
        child.W = np.concatenate([self.W, pad], axis=1)
        return child


    def _field_slice(self, field: str) -> tuple[int, int]:
        idx = self._Z_FIELDS.index(field)
        start = idx * self.latent_dim
        return start, start + self.latent_dim
