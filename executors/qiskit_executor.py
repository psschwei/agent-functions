"""Qiskit executor for running quantum workloads on AerSimulator."""
from typing import List, Dict, Any, Union

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit import Parameter

from config import QISKIT_CONFIG


def get_aer_simulator() -> AerSimulator:
    """
    Get an instance of AerSimulator with default configuration.

    Returns:
        Configured AerSimulator instance
    """
    simulator = AerSimulator()
    return simulator


def run_estimator(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    observables: Union[SparsePauliOp, List[SparsePauliOp]],
    parameter_values: List[List[float]] = None,
) -> Any:
    """
    Run the Estimator primitive with circuits and observables on AerSimulator.

    Args:
        circuits: Single circuit or list of circuits to execute
        observables: Single observable or list of observables to measure
        parameter_values: Optional parameter values for parameterized circuits

    Returns:
        EstimatorResult object containing expectation values
    """
    # Create Aer Estimator
    estimator = AerEstimator()

    # Prepare circuits and observables as lists
    if not isinstance(circuits, list):
        circuits = [circuits]
    if not isinstance(observables, list):
        observables = [observables]

    # Run estimator
    if parameter_values is not None:
        # Create list of tuples (circuit, observable, params) for each measurement
        pub_list = []
        for i, (circ, obs) in enumerate(zip(circuits, observables)):
            if i < len(parameter_values):
                pub_list.append((circ, obs, parameter_values[i]))
            else:
                pub_list.append((circ, obs))

        job = estimator.run(pub_list)
    else:
        # No parameters, just run circuits with observables
        pub_list = [(circ, obs) for circ, obs in zip(circuits, observables)]
        job = estimator.run(pub_list)

    # Get results
    result = job.result()

    return result


def run_sampler(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    shots: int = None,
) -> Any:
    """
    Run the Sampler primitive with circuits on AerSimulator.

    Args:
        circuits: Single circuit or list of circuits to execute
        shots: Number of shots (uses config default if not specified)

    Returns:
        SamplerResult object containing measurement outcomes
    """
    # Use configured shots if not specified
    if shots is None:
        shots = QISKIT_CONFIG["shots"]

    # Create Aer Sampler
    sampler = AerSampler()

    # Prepare circuits as list
    if not isinstance(circuits, list):
        circuits = [circuits]

    # Run sampler
    job = sampler.run(circuits, shots=shots)
    result = job.result()

    return result


def execute_circuit(
    circuit: QuantumCircuit,
    shots: int = None,
    seed_simulator: int = None,
) -> Dict[str, Any]:
    """
    Execute a quantum circuit on AerSimulator and return counts.

    Args:
        circuit: Quantum circuit to execute
        shots: Number of shots
        seed_simulator: Random seed for reproducibility

    Returns:
        Dictionary with execution results
    """
    # Get simulator
    simulator = get_aer_simulator()

    # Use config defaults if not specified
    if shots is None:
        shots = QISKIT_CONFIG["shots"]
    if seed_simulator is None:
        seed_simulator = QISKIT_CONFIG["seed_simulator"]

    # Run circuit
    job = simulator.run(circuit, shots=shots, seed_simulator=seed_simulator)
    result = job.result()

    # Get counts
    counts = result.get_counts(circuit)

    return {
        "counts": counts,
        "shots": shots,
        "success": result.success,
    }
