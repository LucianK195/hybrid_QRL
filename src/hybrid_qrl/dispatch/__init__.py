"""Dynamic resource-dispatch benchmark for hybrid action proposals.

The subpackage contains a small, dependency-light research benchmark rather
than a production scheduler.  It combines a dynamic weighted independent-set
environment, an on-policy actor-critic learner, classical optimization
baselines, and a classical surrogate for Rydberg-blockade proposal dynamics.
"""

from .dataset import DatasetExportSummary, export_test_dataset
from .backlog_benchmark import BacklogBenchmarkConfig, run_backlog_benchmark
from .azure_bundle import AzureBundleConfig, run_azure_bundle_benchmark
from .azure_packing import AzurePackingConfig, run_azure_packing_benchmark
from .environment import (
    DispatchConfig,
    DispatchEnvironment,
    DispatchState,
    induced_dispatch_state,
)
from .generalization_benchmark import (
    GeneralizationBenchmarkConfig,
    run_generalization_benchmark,
)
from .latency_benchmark import (
    LatencyAwareConfig,
    LatencyTrace,
    run_latency_aware_benchmark,
)
from .learning import ActorCriticModel, TrainingConfig, train_actor_critic
from .sampler_loop import SamplerLoopTrainingConfig, train_sampler_in_loop

__all__ = [
    "ActorCriticModel",
    "BacklogBenchmarkConfig",
    "AzureBundleConfig",
    "AzurePackingConfig",
    "DatasetExportSummary",
    "DispatchConfig",
    "DispatchEnvironment",
    "DispatchState",
    "GeneralizationBenchmarkConfig",
    "LatencyAwareConfig",
    "LatencyTrace",
    "SamplerLoopTrainingConfig",
    "TrainingConfig",
    "export_test_dataset",
    "induced_dispatch_state",
    "run_backlog_benchmark",
    "run_azure_bundle_benchmark",
    "run_azure_packing_benchmark",
    "run_generalization_benchmark",
    "run_latency_aware_benchmark",
    "train_actor_critic",
    "train_sampler_in_loop",
]
