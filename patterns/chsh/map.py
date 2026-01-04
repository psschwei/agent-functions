#!/usr/bin/env python3
"""
CHSH Pattern - Map Stage

Creates parameterized Bell state circuits and observables for CHSH inequality test.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


def create_bell_circuit(theta: Parameter) -> QuantumCircuit:
    """
    Create a parameterized Bell state circuit for CHSH experiment.

    The circuit creates an entangled Bell state and applies a rotation
    on qubit 1 based on parameter theta.

    Args:
        theta: Parameter for rotation angle

    Returns:
        Parameterized quantum circuit
    """
    qc = QuantumCircuit(2)

    # Create Bell state (|00⟩ + |11⟩) / √2
    qc.h(0)
    qc.cx(0, 1)

    # Apply parameterized rotation on qubit 1
    qc.ry(theta, 1)

    return qc


def create_chsh_observables():
    """
    Create the observables for CHSH inequality measurement.

    The CHSH inequality uses two observables per qubit:
    - Observable A1, A2 for qubit 0 (Alice)
    - Observable B1, B2 for qubit 1 (Bob)

    Returns:
        Tuple of (observable1, observable2) as SparsePauliOp
    """
    # For CHSH, we measure correlations between qubits
    # Observable 1: Z⊗Z (measure both qubits in Z basis)
    obs1 = SparsePauliOp(["ZZ"], coeffs=[1.0])

    # Observable 2: X⊗X (measure both qubits in X basis)
    obs2 = SparsePauliOp(["XX"], coeffs=[1.0])

    return obs1, obs2


def main():
    """Map stage: Create Bell circuit and observables for CHSH pattern."""
    parser = argparse.ArgumentParser(
        description="CHSH Map Stage - Create Bell circuit and observables"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHSH Pattern - Map Stage")
    print("=" * 60)

    # Create parameter for the circuit
    theta = Parameter('θ')

    print("\n[1/3] Creating parameterized Bell circuit...")
    circuit = create_bell_circuit(theta)
    print(f"Circuit created with {circuit.num_qubits} qubits and {circuit.num_parameters} parameters")

    print("\n[2/3] Creating CHSH observables...")
    obs1, obs2 = create_chsh_observables()
    print(f"Observable 1: {obs1}")
    print(f"Observable 2: {obs2}")

    print("\n[3/3] Defining phase sweep for CHSH test...")
    # Create phase values for parameter sweep
    # We'll sweep theta from 0 to 2π
    phases = np.linspace(0, 2 * np.pi, 16)
    print(f"Created {len(phases)} phase values from 0 to 2π")

    # Package the results
    result = {
        "circuit": circuit,
        "observable1": obs1,
        "observable2": obs2,
        "phases": phases,
        "parameter": theta,
    }

    # Save to pickle file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✓ Map stage complete!")
    print(f"  Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
