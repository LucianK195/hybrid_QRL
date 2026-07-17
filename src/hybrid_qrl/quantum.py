"""Quantum candidate-sampler contracts and neutral-atom/Rydberg emulators.

All samplers in this module implement the :class:`CandidateSampler` contract:
they map a utility vector and a :class:`~hybrid_qrl.core.ConflictGraph` to raw
binary candidates.  The idealized Rydberg Hamiltonian used by the built-in
emulators is

``H(t) = Omega(t)/2 sum_i X_i - Delta(t) sum_i u_i n_i``
``       + U sum_(i,j in E) n_i n_j``,

where ``u_i`` is a state-dependent utility and ``U`` penalizes simultaneous
selection of conflict-edge endpoints.  The implementations differ in how they
construct and evolve this Hamiltonian:

* :class:`RydbergEmulatorSampler` builds dense pulse unitaries and executes a
  Qiskit circuit;
* :class:`DenseRydbergStatevectorSampler` applies the same pulse schedule
  directly with SciPy matrix exponentials;
* :class:`QuTiPRydbergSampler` integrates a continuous Hamiltonian with QuTiP;
* :class:`ManualNeutralAtomBackendSampler` delegates evolution to the downloaded
  multi-layer neutral-atom simulator.

Conventions
-----------
Every returned tuple is in graph-node order, independent of a backend's native
bitstring convention.  ``Action[i]`` therefore always refers to graph node
``i``.  Backend measurements may violate intended constraints, so the
downstream safety filter remains the authoritative validator.

These classes are research simulators and integration adapters.  Dense
simulation cost, cache behavior, and solver timing are not representative of
neutral-atom hardware performance and do not establish quantum advantage.
Optional Qiskit and QuTiP dependencies are imported lazily.
"""

from __future__ import annotations

import math
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import expm

_WINDOWS_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    # The msvc-runtime wheel places redistributable DLLs in these environment
    # directories. Python 3.8+ requires explicit DLL search paths on Windows.
    for candidate in (Path(sys.prefix), Path(sys.prefix) / "Scripts"):
        if candidate.exists():
            _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(str(candidate)))

qt = None
_QUTIP_IMPORT_ERROR: ImportError | None = None

QuantumCircuit = None
transpile = None
UnitaryGate = None
BasicSimulator = None
SparsePauliOp = None
AerSimulator = None
_QISKIT_IMPORT_ERROR: ImportError | None = None

from .core import Action, ConflictGraph


def _load_qutip():
    """Return the lazily imported QuTiP module or raise an actionable error."""
    global qt, _QUTIP_IMPORT_ERROR
    if qt is not None:
        return qt
    try:
        import qutip as qutip_module
    except ImportError as error:
        _QUTIP_IMPORT_ERROR = error
        raise ImportError(
            "QuTiPRydbergSampler requires the optional QuTiP dependencies. "
            "Install the project with the 'qutip' or 'quantum' extra."
        ) from error
    qt = qutip_module
    return qt


def _load_qiskit() -> None:
    """Populate lazily imported Qiskit symbols used by the dense circuit path."""
    global QuantumCircuit
    global transpile
    global UnitaryGate
    global BasicSimulator
    global SparsePauliOp
    global AerSimulator
    global _QISKIT_IMPORT_ERROR

    if QuantumCircuit is not None:
        return
    try:
        from qiskit import QuantumCircuit as quantum_circuit
        from qiskit import transpile as qiskit_transpile
        from qiskit.circuit.library import UnitaryGate as unitary_gate
        from qiskit.providers.basic_provider import (
            BasicSimulator as basic_simulator,
        )
        from qiskit.quantum_info import SparsePauliOp as sparse_pauli_op
    except ImportError as error:
        _QISKIT_IMPORT_ERROR = error
        raise ImportError(
            "RydbergEmulatorSampler requires the optional Qiskit dependencies. "
            "Install the project with the 'qiskit' or 'quantum' extra."
        ) from error

    try:
        from qiskit_aer import AerSimulator as aer_simulator
    except (ImportError, OSError):
        aer_simulator = None

    QuantumCircuit = quantum_circuit
    transpile = qiskit_transpile
    UnitaryGate = unitary_gate
    BasicSimulator = basic_simulator
    SparsePauliOp = sparse_pauli_op
    AerSimulator = aer_simulator


@dataclass(frozen=True)
class PulseSchedule:
    """Parameters for the shared adiabatic-inspired pulse schedule.

    Parameters
    ----------
    duration:
        Total evolution time in simulator units.  Must be positive.
    steps:
        Number of equal piecewise-constant intervals.  Must be positive.
    omega_max:
        Peak transverse Rabi amplitude.  The envelope follows
        ``omega_max * sin(pi * progress)``.
    delta_start, delta_end:
        Initial and final global detuning values for a linear sweep.
    blockade:
        Interaction penalty ``U`` applied when both endpoints of a conflict
        edge are excited.

    Notes
    -----
    The QuTiP sampler evaluates a continuous version of the same envelope,
    whereas the dense and Qiskit samplers evaluate interval midpoints.
    """

    duration: float = 10.0
    steps: int = 40
    omega_max: float = 1.5
    delta_start: float = -3.0
    delta_end: float = 3.0
    blockade: float = 10.0

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")

    def at(self, step: int) -> tuple[float, float]:
        """Evaluate Rabi amplitude and detuning at one interval midpoint.

        Parameters
        ----------
        step:
            Zero-based interval index in ``[0, steps)``.

        Returns
        -------
        tuple of float
            ``(omega, delta)`` for the selected interval.

        Raises
        ------
        IndexError
            If ``step`` lies outside the configured schedule.
        """
        if not 0 <= step < self.steps:
            raise IndexError("pulse step is outside the schedule")
        progress = (step + 0.5) / self.steps
        omega = self.omega_max * math.sin(math.pi * progress)
        delta = self.delta_start + progress * (self.delta_end - self.delta_start)
        return omega, delta


@dataclass(frozen=True)
class _QiskitProblem:
    """Internal immutable input for Qiskit's Hamiltonian builder."""

    weights: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def nodes(self) -> int:
        """Number of binary decisions/qubits represented by ``weights``."""
        return len(self.weights)


def _qiskit_hamiltonian(
    problem: _QiskitProblem,
    omega: float,
    delta: float,
    blockade: float,
) -> np.ndarray:
    """Construct one dense Qiskit Hamiltonian matrix for a pulse interval."""
    terms: list[tuple[str, list[int], complex]] = []
    identity_shift = 0.0
    for node, weight in enumerate(problem.weights):
        terms.append(("X", [node], omega / 2.0))
        identity_shift += -delta * weight / 2.0
        terms.append(("Z", [node], delta * weight / 2.0))

    for left, right in problem.edges:
        identity_shift += blockade / 4.0
        terms.append(("Z", [left], -blockade / 4.0))
        terms.append(("Z", [right], -blockade / 4.0))
        terms.append(("ZZ", [left, right], blockade / 4.0))

    terms.append(("I", [0], identity_shift))
    operator = SparsePauliOp.from_sparse_list(
        terms, num_qubits=problem.nodes
    ).simplify()
    return np.asarray(operator.to_matrix(), dtype=complex)


def _qiskit_rydberg_samples(
    problem: _QiskitProblem,
    schedule: PulseSchedule,
    shots: int,
    seed: int,
) -> tuple[list[Action], float, str]:
    """Compile, execute, and decode a piecewise-unitary Qiskit experiment."""
    _load_qiskit()

    circuit = QuantumCircuit(problem.nodes)
    time_step = schedule.duration / schedule.steps
    start = time.perf_counter()
    for step in range(schedule.steps):
        omega, delta = schedule.at(step)
        hamiltonian = _qiskit_hamiltonian(
            problem, omega, delta, schedule.blockade
        )
        unitary = expm(-1j * hamiltonian * time_step)
        circuit.append(
            UnitaryGate(unitary, label=f"pulse_{step}"),
            range(problem.nodes),
        )
    circuit.measure_all()

    if AerSimulator is not None:
        backend = AerSimulator(method="statevector")
        backend_name = "qiskit_aer.AerSimulator(statevector)"
    else:
        backend = BasicSimulator()
        backend_name = "qiskit.providers.basic_provider.BasicSimulator"
    compiled = transpile(circuit, backend, optimization_level=0)
    counts = (
        backend.run(compiled, shots=shots, seed_simulator=seed)
        .result()
        .get_counts()
    )
    elapsed = time.perf_counter() - start

    samples: list[Action] = []
    for qiskit_bits, count in counts.items():
        bits = tuple(int(bit) for bit in reversed(qiskit_bits.replace(" ", "")))
        samples.extend([bits] * count)
    return samples, elapsed, backend_name


class QuantumSamplerTemplate(ABC):
    """Manual integration skeleton for a real or simulated quantum backend.

    Subclasses define how utilities and constraints become a backend program,
    how that program is executed, and how raw measurements become bit actions.
    The inherited ``sample`` method makes the result compatible with the hybrid
    pipeline. Safety is deliberately enforced again outside this class.

    Subclasses normally override ``name`` and implement the three abstract
    translation hooks.  The base :meth:`sample` method derives a backend seed
    from the caller-owned NumPy generator and requests one shot per candidate.
    """

    name = "custom_quantum_sampler"

    @abstractmethod
    def build_program(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> Any:
        """Translate utilities and constraints into a backend program.

        Parameters
        ----------
        utilities:
            One state-dependent utility per graph node.
        graph:
            Intended pairwise constraints and cardinality bounds.

        Returns
        -------
        Any
            Backend-specific circuit, pulse sequence, Hamiltonian, or job
            description accepted by :meth:`execute`.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, program: Any, shots: int, seed: int) -> Iterable[Any]:
        """Execute ``program`` and yield raw backend measurements.

        Parameters
        ----------
        program:
            Object produced by :meth:`build_program`.
        shots:
            Requested measurement count; normally equal to candidate budget
            ``K``.
        seed:
            Backend seed derived from the pipeline's NumPy generator.

        Returns
        -------
        iterable
            Raw measurement objects consumed one at a time by :meth:`decode`.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, measurement: Any, nodes: int) -> Action:
        """Convert one native measurement to graph-node action order.

        Parameters
        ----------
        measurement:
            Backend-specific measurement record.
        nodes:
            Required number of action bits.

        Returns
        -------
        Action
            Binary tuple whose index order matches the conflict graph.
        """
        raise NotImplementedError

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Build, execute, and decode one candidate batch.

        Parameters
        ----------
        utilities, graph, candidates, rng:
            See :meth:`hybrid_qrl.core.CandidateSampler.sample`.

        Returns
        -------
        list of Action
            Decoded measurements.  Feasibility is not guaranteed.

        Raises
        ------
        ValueError
            If ``candidates`` is not positive.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        program = self.build_program(utilities, graph)
        measurements = self.execute(program, candidates, seed)
        return [self.decode(item, graph.nodes) for item in measurements]


class RydbergEmulatorSampler:
    """Emulate a piecewise Rydberg pulse as a dense Qiskit circuit.

    Parameters
    ----------
    schedule:
        Pulse configuration.  ``None`` selects :class:`PulseSchedule` defaults.

    Attributes
    ----------
    last_backend:
        Name of the most recently used Qiskit simulator, or ``None`` before the
        first call.
    last_elapsed_seconds:
        Most recent circuit construction, compilation, and execution time.

    Notes
    -----
    Each pulse interval materializes a dense ``2**n`` square unitary.  This
    implementation is limited to small graphs and its elapsed time is emulator
    overhead, not a QPU latency estimate.
    """

    name = "qiskit_rydberg_emulator"

    def __init__(self, schedule: PulseSchedule | None = None):
        self.schedule = schedule or PulseSchedule()
        self.last_backend: str | None = None
        self.last_elapsed_seconds: float | None = None

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Sample raw candidates with the Qiskit emulator.

        Parameters
        ----------
        utilities:
            Floating-point vector with shape ``(graph.nodes,)``.
        graph:
            Graph whose edges define interaction penalties.
        candidates:
            Number of measurement shots/candidates.
        rng:
            Generator used to derive the Qiskit simulator seed.

        Returns
        -------
        list of Action
            Exactly ``candidates`` decoded Qiskit measurements.

        Raises
        ------
        ImportError
            If the optional Qiskit dependencies are unavailable.
        ValueError
            If the candidate budget or utility shape is invalid.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        if utilities.shape != (graph.nodes,):
            raise ValueError("utilities must contain one value per graph node")
        problem = _QiskitProblem(tuple(map(float, utilities)), graph.edges)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        samples, elapsed, backend = _qiskit_rydberg_samples(
            problem, self.schedule, candidates, seed
        )
        self.last_backend = backend
        self.last_elapsed_seconds = elapsed
        return samples


@dataclass
class DenseRydbergStatevectorSampler:
    """Small cached dense emulator for repeatedly changing local utilities.

    This avoids Qiskit compilation at every environment step. Utility vectors
    are rounded only for the cache key, making the approximation explicit. It
    remains a classical ideal-state emulator and is intended only for small
    integration tests such as the two-qubit CartPole action head.

    Parameters
    ----------
    schedule:
        Piecewise pulse and interaction parameters.
    cache_decimals:
        Decimal precision used both for cache keys and evolved utilities.
        Lower values increase cache reuse but introduce a coarser approximation.
    name:
        Diagnostic sampler identifier stored in :class:`~hybrid_qrl.core.Decision`.

    Attributes
    ----------
    cache_hits, cache_misses:
        Counts of probability-distribution cache accesses since construction.

    Notes
    -----
    For ``n`` nodes, this implementation constructs dense ``2**n`` by ``2**n``
    Hamiltonians and exponentiates one per pulse step.  Memory and runtime grow
    exponentially.  Cached arrays are returned by reference and should be
    treated as read-only by callers.
    """

    schedule: PulseSchedule = PulseSchedule()
    cache_decimals: int = 2
    name: str = "dense_rydberg_statevector_emulator"

    def __post_init__(self) -> None:
        self._probability_cache: dict[tuple[object, ...], np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> tuple[object, ...]:
        """Build an immutable key from graph, rounded utilities, and schedule."""
        return (
            graph.nodes,
            graph.edges,
            tuple(np.round(utilities, self.cache_decimals)),
            self.schedule,
        )

    def probabilities(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> np.ndarray:
        """Evolve ``|00...0>`` and return final basis probabilities.

        Parameters
        ----------
        utilities:
            Per-node utilities with shape ``(graph.nodes,)``.  Values are rounded
            to ``cache_decimals`` before Hamiltonian construction.
        graph:
            Conflict edges used as pair-interaction penalties.

        Returns
        -------
        numpy.ndarray
            Normalized vector of length ``2**graph.nodes``.  Basis index bit
            ``i`` corresponds to graph node ``i``.

        Raises
        ------
        ValueError
            If the utility shape does not match the graph.
        """
        if utilities.shape != (graph.nodes,):
            raise ValueError("utilities must contain one value per graph node")
        key = self._cache_key(utilities, graph)
        cached = self._probability_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        nodes = graph.nodes
        dimension = 1 << nodes
        state = np.zeros(dimension, dtype=complex)
        state[0] = 1.0
        dt = self.schedule.duration / self.schedule.steps
        rounded_utilities = np.round(utilities, self.cache_decimals)

        for step in range(self.schedule.steps):
            omega, delta = self.schedule.at(step)
            hamiltonian = np.zeros((dimension, dimension), dtype=complex)
            for basis in range(dimension):
                bits = tuple((basis >> node) & 1 for node in range(nodes))
                diagonal = -delta * float(np.dot(rounded_utilities, bits))
                diagonal += self.schedule.blockade * sum(
                    bits[left] * bits[right] for left, right in graph.edges
                )
                hamiltonian[basis, basis] = diagonal
                for node in range(nodes):
                    hamiltonian[basis ^ (1 << node), basis] += omega / 2.0
            state = expm(-1j * hamiltonian * dt) @ state

        probabilities = np.abs(state) ** 2
        probabilities /= probabilities.sum()
        self._probability_cache[key] = probabilities
        return probabilities

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Draw candidates from the cached or newly evolved distribution.

        Parameters
        ----------
        utilities, graph, candidates, rng:
            See :meth:`hybrid_qrl.core.CandidateSampler.sample`.

        Returns
        -------
        list of Action
            Exactly ``candidates`` actions in graph-node order.

        Raises
        ------
        ValueError
            If the candidate budget is not positive or utilities are malformed.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        probabilities = self.probabilities(utilities, graph)
        states = rng.choice(len(probabilities), size=candidates, p=probabilities)
        return [
            tuple((int(state) >> node) & 1 for node in range(graph.nodes))
            for state in states
        ]


@dataclass
class QuTiPRydbergSampler:
    """Time-dependent ideal Rydberg sampler using the QuTiP solver API.

    The Hamiltonian is

        H(t) = Omega(t)/2 sum_i X_i
               - Delta(t) sum_i u_i n_i
               + U sum_(i,j in E) n_i n_j.

    QuTiP's ``QobjEvo`` and ``sesolve`` integrate the continuous pulse. Final
    state probabilities are sampled with the pipeline-provided NumPy RNG. The
    hard safety layer remains authoritative after measurement.

    Parameters
    ----------
    schedule:
        Continuous pulse-envelope and interaction parameters.
    cache_decimals:
        Utility precision used for both the cache key and Hamiltonian.  ``None``
        disables rounding and distribution caching.
    solver_options:
        Options forwarded to :func:`qutip.sesolve`.  Defaults retain the final
        state, suppress progress output, and normalize solver output.
    name:
        Diagnostic sampler identifier.

    Attributes
    ----------
    cache_hits, cache_misses:
        Probability-cache counters.
    last_solver_stats:
        QuTiP statistics from the most recent uncached evolution.

    Notes
    -----
    QuTiP tensor order places graph node ``0`` at the most-significant basis
    digit.  :meth:`sample` preserves graph-node tuple order when decoding.
    """

    schedule: PulseSchedule = PulseSchedule()
    cache_decimals: int | None = 3
    solver_options: dict[str, object] = field(
        default_factory=lambda: {
            "store_final_state": True,
            "progress_bar": "",
            "normalize_output": True,
        }
    )
    name: str = "qutip_rydberg_sesolve"

    def __post_init__(self) -> None:
        _load_qutip()
        self._probability_cache: dict[tuple[object, ...], np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_solver_stats: dict[str, object] | None = None

    @property
    def qutip_version(self) -> str:
        """Return the version string of the lazily imported QuTiP module."""
        return str(qt.__version__)

    @staticmethod
    def _embedded_operator(operator, node: int, nodes: int):
        """Embed a one-qubit QuTiP operator at ``node`` in an ``nodes`` register."""
        factors = [qt.qeye(2) for _ in range(nodes)]
        factors[node] = operator
        return qt.tensor(factors)

    def _operators(self, graph: ConflictGraph):
        """Build the transverse, number, and interaction operators."""
        identity = qt.qeye([2] * graph.nodes)
        number = (qt.qeye(2) - qt.sigmaz()) / 2.0
        x_sum = 0 * identity
        numbers = []
        for node in range(graph.nodes):
            x_sum += self._embedded_operator(qt.sigmax(), node, graph.nodes)
            numbers.append(self._embedded_operator(number, node, graph.nodes))
        interactions = 0 * identity
        for left, right in graph.edges:
            interactions += numbers[left] * numbers[right]
        return x_sum, numbers, interactions

    def _rounded_utilities(self, utilities: np.ndarray) -> np.ndarray:
        """Convert utilities to float and apply configured rounding."""
        if self.cache_decimals is None:
            return np.asarray(utilities, dtype=float)
        return np.round(np.asarray(utilities, dtype=float), self.cache_decimals)

    def _cache_key(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> tuple[object, ...] | None:
        """Return an evolution cache key, or ``None`` when caching is disabled."""
        if self.cache_decimals is None:
            return None
        return (
            graph.nodes,
            graph.edges,
            tuple(self._rounded_utilities(utilities)),
            self.schedule,
        )

    def probabilities(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> np.ndarray:
        """Integrate the continuous Hamiltonian and return final probabilities.

        Parameters
        ----------
        utilities:
            Per-node detuning weights with shape ``(graph.nodes,)``.
        graph:
            Conflict graph whose edges receive blockade interactions.

        Returns
        -------
        numpy.ndarray
            Normalized computational-basis distribution of length
            ``2**graph.nodes`` in QuTiP tensor order.

        Raises
        ------
        ValueError
            If the utility vector has the wrong shape.

        Notes
        -----
        A cache hit returns the stored NumPy array without invoking the solver or
        changing ``last_solver_stats``.  Treat returned arrays as read-only.
        """
        utilities = np.asarray(utilities, dtype=float)
        if utilities.shape != (graph.nodes,):
            raise ValueError("utilities must contain one value per graph node")

        key = self._cache_key(utilities, graph)
        if key is not None and key in self._probability_cache:
            self.cache_hits += 1
            return self._probability_cache[key]

        self.cache_misses += 1
        rounded = self._rounded_utilities(utilities)
        x_sum, numbers, interactions = self._operators(graph)
        weighted_number = sum(
            (float(weight) * operator for weight, operator in zip(rounded, numbers)),
            0 * interactions,
        )

        duration = self.schedule.duration

        def hamiltonian(time: float, _args: dict[str, object] | None = None):
            progress = min(1.0, max(0.0, time / duration))
            omega = self.schedule.omega_max * math.sin(math.pi * progress)
            delta = self.schedule.delta_start + progress * (
                self.schedule.delta_end - self.schedule.delta_start
            )
            return (
                (omega / 2.0) * x_sum
                - delta * weighted_number
                + self.schedule.blockade * interactions
            )

        initial = qt.tensor([qt.basis(2, 0) for _ in range(graph.nodes)])
        hamiltonian_evolution = qt.QobjEvo(hamiltonian)
        times = np.linspace(0.0, duration, self.schedule.steps + 1)
        result = qt.sesolve(
            hamiltonian_evolution,
            initial,
            times,
            e_ops=[],
            options=dict(self.solver_options),
        )
        final_state = (
            result.final_state
            if result.final_state is not None
            else result.states[-1]
        )
        amplitudes = np.asarray(final_state.full()).reshape(-1)
        probabilities = np.abs(amplitudes) ** 2
        probabilities /= probabilities.sum()
        self.last_solver_stats = dict(result.stats)
        if key is not None:
            self._probability_cache[key] = probabilities
        return probabilities

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Draw raw action candidates from the QuTiP final state.

        Parameters
        ----------
        utilities, graph, candidates, rng:
            See :meth:`hybrid_qrl.core.CandidateSampler.sample`.

        Returns
        -------
        list of Action
            Exactly ``candidates`` tuples in graph-node order.

        Raises
        ------
        ValueError
            If the candidate budget or utility shape is invalid.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        probabilities = self.probabilities(utilities, graph)
        states = rng.choice(len(probabilities), size=candidates, p=probabilities)
        # QuTiP tensor order is node 0, node 1, ... from the most-significant
        # basis digit to the least-significant basis digit.
        return [
            tuple(int(bit) for bit in format(int(state), f"0{graph.nodes}b"))
            for state in states
        ]


@dataclass
class ManualNeutralAtomBackendSampler:
    """Adapter for the downloaded ``neutral_atom.simulator`` multi-layer API.

    The external simulator owns pulse construction, QuTiP evolution, geometry,
    blockade interactions, and measurement. This adapter only translates the
    project's ``utilities + ConflictGraph + K`` contract into:

    ``create_simulator() -> evolve_adiabatic() -> sample()``.

    Important: the external backend derives interactions from atom positions,
    not directly from ``graph.edges``. The project safety filter therefore
    remains the final authority on feasibility.

    Parameters
    ----------
    positions:
        Optional atom coordinates with shape ``(n_qubits, 2)``.  If omitted, a
        unit-spaced line is generated for the current graph.
    C6:
        Van der Waals interaction coefficient passed to the simulator factory.
    protocol:
        Backend-native adiabatic protocol.  ``None`` creates a documented
        default :class:`neutral_atom.simulator.AdiabaticProtocol`.
    backend_source:
        Root of a local ``QML-Platform-for-Neutral-Atom`` checkout.  When
        omitted, parent directories and then the working directory are searched.
    cache_decimals:
        Decimal precision for utility and position cache keys.  ``None``
        disables distribution caching without rounding physical inputs.
    utility_scale:
        Multiplicative conversion from learned utilities to local detuning
        weights.
    backend_name:
        Name supplied to ``create_simulator``; defaults to ``"qutip"``.
    simulator_kwargs:
        Additional keyword arguments forwarded to the simulator factory, for
        example a noise configuration.
    name:
        Diagnostic sampler identifier.

    Attributes
    ----------
    cache_hits, cache_misses:
        Probability-cache counters.
    last_evolution_result:
        Backend result from the most recent uncached adiabatic evolution.
    last_backend_object:
        Simulator instance used by the most recent uncached evolution.

    Notes
    -----
    This adapter mutates ``sys.path`` only when it discovers an uninstalled
    local backend package.  For deployed environments, installing the backend
    normally or passing ``backend_source`` explicitly is preferable.
    """

    positions: np.ndarray | None = None
    C6: float = 10.0
    protocol: object | None = None
    backend_source: Path | str | None = None
    cache_decimals: int | None = 3
    utility_scale: float = 1.0
    backend_name: str = "qutip"
    simulator_kwargs: dict[str, object] = field(default_factory=dict)
    name: str = "manual_neutral_atom_qutip"

    def __post_init__(self) -> None:
        if self.backend_source is not None:
            source = Path(self.backend_source)
        else:
            source = next(
                (
                    parent / "QML-Platform-for-Neutral-Atom"
                    for parent in Path(__file__).resolve().parents
                    if (parent / "QML-Platform-for-Neutral-Atom").exists()
                ),
                Path.cwd() / "QML-Platform-for-Neutral-Atom",
            )
        source_package = source / "src"
        if source_package.exists() and str(source_package) not in sys.path:
            sys.path.insert(0, str(source_package))

        try:
            from neutral_atom.simulator import AdiabaticProtocol, create_simulator
        except ImportError as error:
            raise ImportError(
                "Could not import the downloaded neutral_atom.simulator package. "
                "Pass backend_source=<path-to-QML-Platform-for-Neutral-Atom> or "
                "install that package in the active environment."
            ) from error

        self._create_simulator = create_simulator
        self.protocol = self.protocol or AdiabaticProtocol(
            total_time=4.0,
            n_steps=40,
            omega_max=1.5,
            delta_g_initial=-3.0,
            delta_l_max=3.0,
        )
        if self.positions is not None:
            positions = np.asarray(self.positions, dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("positions must have shape (n_qubits, 2)")
            self.positions = positions
        self._probability_cache: dict[tuple[object, ...], np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_evolution_result: object | None = None
        self.last_backend_object: object | None = None

    @staticmethod
    def linear_positions(nodes: int, spacing: float = 1.0) -> np.ndarray:
        """Construct a simple two-dimensional line geometry.

        Parameters
        ----------
        nodes:
            Number of atom coordinates to generate.
        spacing:
            Distance between consecutive atoms along the x-axis.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(nodes, 2)`` and zero y-coordinates.
        """
        return np.column_stack(
            (np.arange(nodes, dtype=float) * spacing, np.zeros(nodes, dtype=float))
        )

    def _positions_for(self, graph: ConflictGraph) -> np.ndarray:
        """Resolve and validate atom coordinates for ``graph``."""
        if self.positions is None:
            return self.linear_positions(graph.nodes)
        if self.positions.shape != (graph.nodes, 2):
            raise ValueError(
                f"positions shape {self.positions.shape} does not match "
                f"{graph.nodes} graph nodes"
            )
        return self.positions

    def _weighted_utilities(self, utilities: np.ndarray) -> np.ndarray:
        """Scale utilities for detuning and reject non-finite values."""
        values = np.asarray(utilities, dtype=float) * self.utility_scale
        if not np.all(np.isfinite(values)):
            raise ValueError("utilities must be finite")
        return values

    def geometry_report(self, graph: ConflictGraph) -> dict[str, object]:
        """Measure how strongly the geometry separates edges from non-edges.

        Parameters
        ----------
        graph:
            Abstract constraint graph to compare against physical interactions.

        Returns
        -------
        dict
            JSON-serializable report containing coordinates, the weakest edge
            interaction, strongest non-edge interaction, their ratio, and a
            warning describing the approximate embedding.

        Notes
        -----
        Interactions are computed as ``C6 / max(distance, 0.1)**6`` to match the
        finite-distance guard used by this diagnostic.  A larger separation
        ratio is preferable, but no ratio turns the physical geometry into an
        exact arbitrary conflict mask.
        """
        positions = self._positions_for(graph)
        edge_set = set(graph.edges)
        edge_interactions = []
        nonedge_interactions = []
        for left in range(graph.nodes):
            for right in range(left + 1, graph.nodes):
                distance = max(
                    float(np.linalg.norm(positions[left] - positions[right])),
                    0.1,
                )
                interaction = self.C6 / distance**6
                if (left, right) in edge_set:
                    edge_interactions.append(interaction)
                else:
                    nonedge_interactions.append(interaction)
        minimum_edge = min(edge_interactions, default=0.0)
        maximum_nonedge = max(nonedge_interactions, default=0.0)
        return {
            "positions": positions.tolist(),
            "minimum_edge_interaction": minimum_edge,
            "maximum_nonedge_interaction": maximum_nonedge,
            "edge_to_nonedge_separation": (
                minimum_edge / maximum_nonedge
                if maximum_nonedge > 0.0
                else float("inf")
            ),
            "warning": (
                "The backend uses all pair interactions C6/r^6; graph edges "
                "are an intended geometry, not an exact interaction mask."
            ),
        }

    def _cache_key(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        positions: np.ndarray,
    ) -> tuple[object, ...] | None:
        """Build a backend-evolution cache key or disable caching with ``None``."""
        if self.cache_decimals is None:
            return None
        rounded_utilities = tuple(np.round(utilities, self.cache_decimals))
        rounded_positions = tuple(
            map(tuple, np.round(positions, self.cache_decimals))
        )
        return (
            graph.nodes,
            graph.edges,
            rounded_utilities,
            rounded_positions,
            self.C6,
            repr(self.protocol),
            self.backend_name,
        )

    def probabilities(
        self, utilities: np.ndarray, graph: ConflictGraph
    ) -> np.ndarray:
        """Run one backend evolution and return its basis distribution.

        Parameters
        ----------
        utilities:
            One finite local-detuning utility per graph node.
        graph:
            Abstract constraints used for shape validation and cache identity.
            Its edges are not passed directly to the physical backend.

        Returns
        -------
        numpy.ndarray
            Normalized vector of length ``2**graph.nodes`` in backend bitstring
            order, which :meth:`sample` converts to graph-node tuples.

        Raises
        ------
        ValueError
            If utility or position shapes are invalid, or utilities are not
            finite.

        Notes
        -----
        When the backend exposes ``get_state``, probabilities are exact for the
        simulated final state.  Otherwise they are estimated from at least 4096
        fixed-seed calibration shots.  This fallback is a simulator integration
        convenience, not a hardware sampling protocol.
        """
        utilities = np.asarray(utilities, dtype=float)
        if utilities.shape != (graph.nodes,):
            raise ValueError("utilities must contain one value per graph node")
        weighted = self._weighted_utilities(utilities)
        positions = self._positions_for(graph)
        key = self._cache_key(weighted, graph, positions)
        if key is not None and key in self._probability_cache:
            self.cache_hits += 1
            return self._probability_cache[key]

        self.cache_misses += 1
        simulator = self._create_simulator(
            self.backend_name,
            positions=positions,
            C6=self.C6,
            **self.simulator_kwargs,
        )
        simulator.reset()
        self.last_evolution_result = simulator.evolve_adiabatic(
            self.protocol,
            {
                node: float(weight)
                for node, weight in enumerate(weighted)
                if weight != 0.0
            },
        )
        self.last_backend_object = simulator

        if hasattr(simulator, "get_state"):
            amplitudes = np.asarray(simulator.get_state().full()).reshape(-1)
            probabilities = np.abs(amplitudes) ** 2
            probabilities = np.clip(probabilities, 0.0, None)
            probabilities /= probabilities.sum()
        else:
            # Backend-neutral fallback: approximate the distribution with a
            # larger measurement batch when direct state access is unavailable.
            calibration_shots = max(4096, 256 * graph.nodes)
            bitstrings = simulator.sample(shots=calibration_shots, seed=0)
            counts = np.zeros(1 << graph.nodes, dtype=float)
            for bitstring in bitstrings:
                counts[int(bitstring, 2)] += 1.0
            probabilities = counts / counts.sum()

        if key is not None:
            self._probability_cache[key] = probabilities
        return probabilities

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Sample candidates from the downloaded backend distribution.

        Parameters
        ----------
        utilities, graph, candidates, rng:
            See :meth:`hybrid_qrl.core.CandidateSampler.sample`.

        Returns
        -------
        list of Action
            Exactly ``candidates`` action tuples in graph-node order.

        Raises
        ------
        ValueError
            If the candidate budget, utility vector, or positions are invalid.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        probabilities = self.probabilities(utilities, graph)
        states = rng.choice(len(probabilities), size=candidates, p=probabilities)
        return [
            tuple(int(bit) for bit in format(int(state), f"0{graph.nodes}b"))
            for state in states
        ]
