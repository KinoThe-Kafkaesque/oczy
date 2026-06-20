"""Minimal Skill Immune Cortex.

A small, pure-Python mistake-immune layer.  Past corrections become bounded
failure signatures (``MistakeDetector``s).  Repeated signatures are merged and,
when the same mistake class keeps appearing, compiled into reusable
``Skill`` objects.

Episode contract (keys match ``oczy_common.episode.Episode``):
- ``add_detector(correction_text, mistake_class, response)`` consumes the raw
  correction string produced by ``OrganismAgent`` (no preprocessing expected).
- ``check(query, proposed_answer)`` is called with the user's query and the
  candidate answer; it returns the active immune responses to surface.

No new fields are introduced; this module only reads/writes dict keys that
match the cross-organ Episode schema.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set


_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "and",
    "or",
    "but",
    "for",
    "with",
    "on",
    "at",
    "from",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "have",
    "has",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "not",
    "no",
    "yes",
    "so",
    "if",
    "then",
    "than",
    "very",
    "just",
    "only",
    "also",
    "about",
    "into",
    "through",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "what",
    "which",
    "who",
    "whom",
    "whose",
}


def _extract_triggers(text: str) -> List[str]:
    """Turn free-form correction text into a small set of trigger tokens."""
    tokens: Set[str] = set()
    for match in re.finditer(r"[a-zA-Z0-9_\-]+", text):
        token = match.group(0).lower()
        if len(token) >= 3 and token not in _STOPWORDS:
            tokens.add(token)
    return sorted(tokens)


@dataclass
class MistakeDetector:
    """A bounded failure signature.

    Attributes:
        triggers: Keywords/patterns that activate the detector.
        response: The forced distinction or check to apply when triggered.
        mistake_class: Logical grouping for merging/compiling into skills.
    """

    triggers: List[str]
    response: str
    mistake_class: str = "general"
    hit_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.triggers = [t.lower() for t in self.triggers]

    def matches(self, text: str) -> bool:
        """Return True if any trigger is a substring of *text*."""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in self.triggers)

    def to_dict(self) -> dict:
        return {
            "type": "detector",
            "triggers": self.triggers,
            "response": self.response,
            "mistake_class": self.mistake_class,
            "hit_count": self.hit_count,
        }


@dataclass
class Skill:
    """A reusable, trigger-gated competence distilled from repeated mistakes."""

    name: str
    triggers: List[str]
    policy: str
    usage_count: int = 0

    def __post_init__(self) -> None:
        self.triggers = [t.lower() for t in self.triggers]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in self.triggers)

    def to_dict(self) -> dict:
        return {
            "type": "skill",
            "name": self.name,
            "triggers": self.triggers,
            "policy": self.policy,
            "usage_count": self.usage_count,
        }


class SkillImmuneCortex:
    """Mistake detectors and option-like executable skills distilled from corrections."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.detectors: List[MistakeDetector] = []
        self.merged: List[MistakeDetector] = []
        self.skills: Dict[str, Skill] = {}
        self._class_counts: Dict[str, int] = {}
        self._skill_threshold: int = int(self.config.get("skill_threshold", 3))

    def add_detector(
        self, correction_text: str, mistake_class: str, response: str
    ) -> MistakeDetector:
        """Create a detector from a past correction and register it."""
        triggers = _extract_triggers(correction_text)
        if not triggers:
            # If no usable token survived filtering, fall back to whole words.
            triggers = [t.lower() for t in correction_text.split() if t]
        detector = MistakeDetector(
            triggers=triggers,
            response=response,
            mistake_class=mistake_class,
        )
        self.detectors.append(detector)
        self._class_counts[mistake_class] = self._class_counts.get(mistake_class, 0) + 1

        # Compile into a skill when the same mistake class repeats enough.
        if self._class_counts[mistake_class] >= self._skill_threshold:
            self._compile_skill(mistake_class)

        return detector

    def _compile_skill(self, mistake_class: str) -> Skill | None:
        """Turn repeated detectors of the same class into a reusable Skill."""
        if mistake_class in self.skills:
            return self.skills[mistake_class]

        class_detectors = [d for d in self.detectors if d.mistake_class == mistake_class]
        all_triggers: Set[str] = set()
        policies: Set[str] = set()
        for d in class_detectors:
            all_triggers.update(d.triggers)
            policies.add(d.response)

        skill = Skill(
            name=f"skill_{mistake_class}",
            triggers=sorted(all_triggers),
            policy=f"[{mistake_class}] " + " ".join(sorted(policies)),
        )
        self.skills[mistake_class] = skill
        return skill

    def check(self, query: str, proposed_answer: str) -> List[str]:
        """Return active immune responses for *query* and *proposed_answer*."""
        active: List[str] = []
        seen: Set[str] = set()
        combined = f"{query} {proposed_answer}"

        for detector in self.detectors:
            if detector.matches(combined):
                detector.hit_count += 1
                if detector.response not in seen:
                    seen.add(detector.response)
                    active.append(detector.response)

        for skill in self.skills.values():
            if skill.matches(combined):
                skill.usage_count += 1
                if skill.policy not in seen:
                    seen.add(skill.policy)
                    active.append(skill.policy)

        return active

    def merge_detectors(self) -> None:
        """Combine detectors that share the same mistake_class into broader ones."""
        grouped: Dict[str, List[MistakeDetector]] = {}
        for detector in self.detectors:
            grouped.setdefault(detector.mistake_class, []).append(detector)

        merged: List[MistakeDetector] = []
        leftover: List[MistakeDetector] = []
        for cls, group in grouped.items():
            if len(group) > 1:
                triggers: Set[str] = set()
                responses: Set[str] = set()
                hits = 0
                for d in group:
                    triggers.update(d.triggers)
                    responses.add(d.response)
                    hits += d.hit_count
                merged.append(
                    MistakeDetector(
                        triggers=sorted(triggers),
                        response=" | ".join(sorted(responses)),
                        mistake_class=cls,
                        hit_count=hits,
                    )
                )
            else:
                leftover.extend(group)

        self.merged = merged
        self.detectors = merged + leftover

    def status(self) -> dict:
        """Return a serializable status snapshot.

        ``bytes`` measures the JSON-serialized snapshot byte length (kept for
        backwards compat).  ``serialized_bytes`` is the canonical organ weight
        via ``pickle.dumps(self)`` and ``record_count`` reports the total
        immune records (active detectors plus compiled skills).
        """
        payload = {
            "project": "skill_immune_cortex",
            "ready": True,
            "detector_count": len(self.detectors),
            "merged_count": len(self.merged),
            "skill_count": len(self.skills),
            "class_counts": dict(self._class_counts),
            "record_count": len(self.detectors) + len(self.skills),
            "serialized_bytes": len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)),
        }
        payload["bytes"] = len(json.dumps(payload, default=str).encode("utf-8"))
        return payload

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "detectors": [d.to_dict() for d in self.detectors],
            "merged": [m.to_dict() for m in self.merged],
            "skills": {name: skill.to_dict() for name, skill in self.skills.items()},
            "class_counts": dict(self._class_counts),
        }
