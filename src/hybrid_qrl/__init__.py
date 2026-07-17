"""Public API for hybrid classical-quantum action sampling.

The package exposes four groups of components:

``core``
    Binary action and conflict-graph records.
``classical``
    Minimal encoder, utility head, and randomized-greedy baseline.
``quantum``
    Qiskit, dense-statevector, QuTiP, and downloaded neutral-atom adapters.
``pipeline``
    Authoritative safety filtering and critic best-of-K selection.

Typical use
-----------
Construct a :class:`ConflictGraph`, assemble a :class:`HybridActionHead`, and
call :meth:`HybridActionHead.select` for each environment observation.  The
returned :class:`Decision` includes the safe action and diagnostics needed for
experimental reporting.

Optional quantum dependencies are loaded lazily when their sampler is used.
Importing :mod:`hybrid_qrl` itself only requires the core package dependencies.
"""

from .classical import IdentityEncoder, RandomizedWeightedGreedy, StaticUtilityHead
from .core import Action, ConflictGraph, Decision
from .pipeline import HybridActionHead, SafetyFilter, UtilityCritic
from .quantum import (
    DenseRydbergStatevectorSampler,
    ManualNeutralAtomBackendSampler,
    PulseSchedule,
    QuantumSamplerTemplate,
    QuTiPRydbergSampler,
    RydbergEmulatorSampler,
)

__all__ = [
    "Action",
    "ConflictGraph",
    "Decision",
    "DenseRydbergStatevectorSampler",
    "ManualNeutralAtomBackendSampler",
    "PulseSchedule",
    "HybridActionHead",
    "IdentityEncoder",
    "QuantumSamplerTemplate",
    "QuTiPRydbergSampler",
    "RandomizedWeightedGreedy",
    "RydbergEmulatorSampler",
    "SafetyFilter",
    "StaticUtilityHead",
    "UtilityCritic",
]
