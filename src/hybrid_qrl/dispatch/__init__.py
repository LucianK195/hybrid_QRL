"""Dynamic resource-dispatch benchmark for hybrid action proposals.

The subpackage contains a small, dependency-light research benchmark rather
than a production scheduler.  It combines a dynamic weighted independent-set
environment, an on-policy actor-critic learner, classical optimization
baselines, and a classical surrogate for Rydberg-blockade proposal dynamics.
"""

from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel, TrainingConfig, train_actor_critic

__all__ = [
    "ActorCriticModel",
    "DispatchConfig",
    "DispatchEnvironment",
    "DispatchState",
    "TrainingConfig",
    "train_actor_critic",
]
