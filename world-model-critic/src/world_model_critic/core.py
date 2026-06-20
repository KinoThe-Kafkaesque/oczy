"""Public exports for World-Model Critic.

The v1 organ implementation lives entirely in :mod:`world_model_critic.critic`;
this module is kept only as the historical import path for callers that
predate the split into ``critic.py``.

H4 (core.py/wrapper inconsistency) is tracked as a lower-priority issue; for
now this file simply re-exports :class:`WorldModelCritic`.
"""

from .critic import WorldModelCritic

__all__ = ["WorldModelCritic"]
