"""Dynamic resource-dispatch benchmark for hybrid action proposals.

The subpackage contains a small, dependency-light research benchmark rather
than a production scheduler.  It combines a dynamic weighted independent-set
environment, an on-policy actor-critic learner, classical optimization
baselines, and a classical surrogate for Rydberg-blockade proposal dynamics.
"""

from .dataset import DatasetExportSummary, export_test_dataset
from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel, TrainingConfig, train_actor_critic
from .sampler_loop import SamplerLoopTrainingConfig, train_sampler_in_loop

__all__ = [
    "ActorCriticModel",
    "DatasetExportSummary",
    "DispatchConfig",
    "DispatchEnvironment",
    "DispatchState",
    "SamplerLoopTrainingConfig",
    "TrainingConfig",
    "export_test_dataset",
    "train_actor_critic",
    "train_sampler_in_loop",
]
