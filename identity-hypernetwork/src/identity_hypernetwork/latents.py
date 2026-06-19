"""Compact latent identity store."""

from __future__ import annotations

import numpy as np


class IdentityLatents:
    """Maintains four latent vectors that summarise learned identity.

    Attributes:
        z_user: what has been learned about the user.
        z_domain: project / domain knowledge.
        z_style: communication style preferences.
        z_mistakes: error patterns to avoid.
    """

    _FIELDS: tuple[str, str, str, str] = ("z_user", "z_domain", "z_style", "z_mistakes")

    def __init__(self, dim: int = 8, rng: np.random.Generator | int | None = None) -> None:
        """Initialise four zero latent vectors.

        Args:
            dim: dimensionality of each individual latent vector.
            rng: optional random source; only used if future variants sample
                initial values.
        """
        self.dim = dim
        self.z_user = np.zeros(dim, dtype=np.float64)
        self.z_domain = np.zeros(dim, dtype=np.float64)
        self.z_style = np.zeros(dim, dtype=np.float64)
        self.z_mistakes = np.zeros(dim, dtype=np.float64)
        # Keep the rng argument for API stability; ignore for zero init.
        _ = rng

    def to_array(self) -> np.ndarray:
        """Return the four latent vectors concatenated into one array."""
        return np.concatenate(
            [self.z_user, self.z_domain, self.z_style, self.z_mistakes]
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise the latents to a plain dictionary."""
        return {field: getattr(self, field).tolist() for field in self._FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "IdentityLatents":
        """Restore latents from a plain dictionary."""
        dim = len(data["z_user"])
        inst = cls(dim=dim)
        for field in cls._FIELDS:
            setattr(inst, field, np.asarray(data[field], dtype=np.float64))
        return inst

    def grow(self, new_dim: int) -> "IdentityLatents":
        """Return a larger IdentityLatents with old values in the leading slice."""
        if new_dim <= self.dim:
            raise ValueError(
                f"new_dim ({new_dim}) must exceed current dim ({self.dim})"
            )

        grown = IdentityLatents(dim=new_dim)
        for field in self._FIELDS:
            old = getattr(self, field)
            padded = np.zeros(new_dim, dtype=np.float64)
            padded[: self.dim] = old
            setattr(grown, field, padded)
        return grown

