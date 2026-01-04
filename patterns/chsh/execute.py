#!/usr/bin/env python3
"""
CHSH Pattern - Execute Stage

Executes the parameterized circuit on AerSimulator for each phase value.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator


def execute_chsh_circuit(circuit, observables, phases, parameter):
    """
    Execute CHSH circuit with parameter sweep on AerSimulator.

    Args:
        circuit: Parameterized quantum circuit
        observables: List of observables to measure
        phases: Array of phase values for parameter sweep
        parameter: Circuit parameter to bind

    Returns:
        Dictionary with results for each observable
    """
    # Create Aer Estimator
    estimator = AerEstimator()

    results = {
        "observable1_values": [],
        "observable2_values": [],
        "phases": phases,
    }

    print(f"Running {len(phases)} phase values...")

    # Run estimator for each phase and observable
    for i, phase in enumerate(phases):
        # Bind parameter value
        param_values = [phase]

        # Measure observable 1
        job1 = estimator.run([circuit], [observables[0]], [param_values])
        result1 = job1.result()
        exp_val1 = result1.values[0]

        # Measure observable 2
        job2 = estimator.run([circuit], [observables[1]], [param_values])
        result2 = job2.result()
        exp_val2 = result2.values[0]

        results["observable1_values"].append(float(exp_val1))
        results["observable2_values"].append(float(exp_val2))

        if (i + 1) % 4 == 0:
            print(f"  Completed {i + 1}/{len(phases)} phase values")

    # Convert to numpy arrays
    results["observable1_values"] = np.array(results["observable1_values"])
    results["observable2_values"] = np.array(results["observable2_values"])

    return results


def main():
    """Execute stage: Run circuit on AerSimulator."""
    parser = argparse.ArgumentParser(
        description="CHSH Execute Stage - Run circuit on simulator"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input pickle file from optimize stage"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHSH Pattern - Execute Stage")
    print("=" * 60)

    # Load input data from optimize stage
    print("\n[1/3] Loading data from optimize stage...")
    with open(args.input, 'rb') as f:
        optimize_data = pickle.load(f)

    circuit = optimize_data["circuit"]
    observables = [optimize_data["observable1"], optimize_data["observable2"]]
    phases = optimize_data["phases"]
    parameter = optimize_data["parameter"]

    print(f"Loaded circuit with depth {circuit.depth()}")
    print(f"Will execute with {len(phases)} different phase values")

    # Execute circuit on simulator
    print("\n[2/3] Executing circuit on AerSimulator...")
    results = execute_chsh_circuit(circuit, observables, phases, parameter)

    print(f"Completed all executions!")
    print(f"  Observable 1 values range: [{results['observable1_values'].min():.3f}, {results['observable1_values'].max():.3f}]")
    print(f"  Observable 2 values range: [{results['observable2_values'].min():.3f}, {results['observable2_values'].max():.3f}]")

    # Prepare output data
    print("\n[3/3] Saving execution results...")
    output_data = {
        "observable1_values": results["observable1_values"],
        "observable2_values": results["observable2_values"],
        "phases": results["phases"],
    }

    # Save to pickle file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Execute stage complete!")
    print(f"  Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
