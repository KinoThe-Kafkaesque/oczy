"""Canonical episode schema for cross-organ data exchange.

The Oczy architecture moves experience through an "organ metabolism":

    experience -> fast change -> replay -> compression -> slow change
              -> forgetting raw trace

Each hand-off crosses an organ boundary, and every boundary crossing has
been a place where a field-name drift bug can hide (the
``correction`` vs ``corrected_answer`` mismatch that silently disabled
``OrganismAgent``'s replay path was exactly this kind of bug --- see the
2026-06-21 session log).  This module is the single source of truth for
the keys that appear on an episode dict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Episode(TypedDict, total=False):
    """A unit of experience passed through the organ metabolism.

    Organs and the glue layer exchange episodes as plain dicts.  All
    known fields are listed here so every writer and reader can agree on
    the contract.  Every field is optional (``total=False``) because a
    fresh episode at the start of the pipeline has none of the
    derived/recovered fields yet, and an old episode at the end of the
    pipeline has had several fields dropped during compression.

    Fields:

    - ``query``: the original user request that triggered this episode.
      Written by the agent; consumed by every downstream organ as the
      retrieval key.
    - ``answer``: the agent's *prior* answer to ``query`` (the answer
      that was correct or corrected).
    - ``correction``: the raw correction sentence a user supplied
      ("No, 'model' here means ML model.").  Always a free-text string;
      never the recovered label.
    - ``corrected_answer``: the recovered corrected *label* (e.g.
      "ML model").  Optional because not every correction carries one.
      This is the field the hippocampus stores and the organism replays
      back into PlasticCortex's ranker.  When absent, callers fall back
      to extracting it from ``correction`` via
      :func:`oczy_common.text_utils.extract_expected_from_correction`.
    - ``outcome``: one of ``accepted`` / ``corrected`` / ``failed`` /
      ``unknown``.  Used by the experience autoencoder to pick an
      outcome bucket.
    - ``prediction_error``: scalar surprise in ``[0, 1]`` produced by
      :class:`WorldModelCritic`.  Used by the hippocampus to gate writes.
    - ``source``: short tag describing where the episode came from
      (``user_correction`` / ``self_play`` / ``feedback`` ...).  Used by
      the identity hypernetwork to route the update into the right
      identity slice.
    - ``id``: a stable episode id assigned by the hippocampus at write
      time.  Used during consolidation to map slow updates back to the
      raw traces that produced them.
    - ``replay_count``: how many times the hippocampus has touched this
      episode via ``reinforce``.  Used to gate consolidation.
    """

    query: str
    answer: str
    correction: str
    corrected_answer: str
    outcome: str
    prediction_error: float
    source: str
    id: str
    replay_count: int


#: The complete set of keys that may appear on an :class:`Episode`.
EPISODE_FIELDS: frozenset[str] = frozenset(Episode.__annotations__.keys())


def validate_episode(episode: dict[str, Any]) -> list[str]:
    """Return the list of *unexpected* keys on ``episode``.

    An empty result means every key on the dict is part of the canonical
    :class:`Episode` shape.  Non-empty results are pointers to schema
    drift --- a writer using a key no reader expects.

    This is a pure check; it never mutates ``episode``.

    Example:
        >>> validate_episode({"query": "x", "corrected_answer": "y"})
        []
        >>> validate_episode({"query": "x", "corrected_label": "y"})
        ['corrected_label']
    """
    unknown = set(episode.keys()) - EPISODE_FIELDS
    return sorted(unknown)