"""Oczy organism curriculum.

A progressive set of learning experiences for the multi-organ agent.  Each
stage targets a specific organ or cross-organ flow:

  stage_0_grounding        -> PlasticCortex fast weights
  stage_1_transfer         -> NeuralHippocampus replay
  stage_2_scope            -> SkillImmuneCortex + IdentityHypernetwork
  stage_3_dialog           -> full organ metabolism
  stage_4_consolidation    -> hippocampal consolidation + autoencoder
  stage_5_cross_domain     -> critic + identity + immune together

The curriculum can be consumed directly as structured episodes or, when the
LM perception layer is available, as natural-language utterances that are
parsed into canonical Episodes before being fed to the agent.
"""

from __future__ import annotations

from .dataset import Episode, Probe, Stage, build_curriculum

__all__ = [
    "Episode",
    "Probe",
    "Stage",
    "build_curriculum",
]
