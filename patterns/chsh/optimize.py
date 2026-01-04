#!/usr/bin/env python3
"""
CHSH Pattern - Optimize Stage

Optimizes/transpiles the circuit for execution on AerSimulator.
"""
import argparse
import pickle
from pathlib import Path

from qiskit import transpile
from qiskit_aer import AerSimulator


def optimize_circuit(circuit, backend):
    """
    Optimize circuit for the target backend.

    For AerSimulator, this is a simple transpilation without
    hardware-specific optimizations.

    Args:
        circuit: Quantum circuit to optimize
        backend: Target backend (AerSimulator)

    Returns:
        Transpiled circuit
    """
    # Simple transpilation for simulator
    # optimization_level=1 provides basic optimizations without being too aggressive
    optimized = transpile(
        circuit,
        backend=backend,
        optimization_level=1
    )

    return optimized


def main():
    """Optimize stage: Transpile circuit for AerSimulator."""
    parser = argparse.ArgumentParser(
        description="CHSH Optimize Stage - Transpile circuit for execution"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input pickle file from map stage"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHSH Pattern - Optimize Stage")
    print("=" * 60)

    # Load input data from map stage
    print("\n[1/3] Loading data from map stage...")
    with open(args.input, 'rb') as f:
        map_data = pickle.load(f)

    circuit = map_data["circuit"]
    print(f"Loaded circuit with {circuit.num_qubits} qubits")

    # Get simulator backend
    print("\n[2/3] Transpiling circuit for AerSimulator...")
    backend = AerSimulator()
    optimized_circuit = optimize_circuit(circuit, backend)

    print(f"Original circuit depth: {circuit.depth()}")
    print(f"Optimized circuit depth: {optimized_circuit.depth()}")

    # Prepare output data
    print("\n[3/3] Saving optimized circuit...")
    result = {
        "circuit": optimized_circuit,
        "observable1": map_data["observable1"],
        "observable2": map_data["observable2"],
        "phases": map_data["phases"],
        "parameter": map_data["parameter"],
        "original_depth": circuit.depth(),
        "optimized_depth": optimized_circuit.depth(),
    }

    # Save to pickle file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✓ Optimize stage complete!")
    print(f"  Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
