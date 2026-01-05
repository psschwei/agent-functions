"""
CHSH Pattern - Decorator-Based Implementation

Implements all four stages of the CHSH inequality test using decorators.
This single file replaces the four separate script files.
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator

from patterns.decorators import (
    map_stage,
    optimize_stage,
    execute_stage,
    post_process_stage,
    PatternContext,
)


@map_stage()
def create_bell_state_and_observables(ctx: PatternContext) -> dict:
    """
    Map Stage: Create parameterized Bell circuit and observables for CHSH test.

    Creates an entangled Bell state circuit with a parameterized rotation,
    and defines the observables for CHSH inequality measurements.
    """
    ctx.log("Creating parameterized Bell circuit...")

    # Create parameter for the circuit
    theta = Parameter('θ')

    # Create Bell state (|00⟩ + |11⟩) / √2
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Apply parameterized rotation on qubit 1
    qc.ry(theta, 1)

    ctx.log(f"Circuit created with {qc.num_qubits} qubits and {qc.num_parameters} parameters")

    # Create CHSH observables
    # Observable 1: Z⊗Z (measure both qubits in Z basis)
    obs1 = SparsePauliOp(["ZZ"], coeffs=[1.0])

    # Observable 2: X⊗X (measure both qubits in X basis)
    obs2 = SparsePauliOp(["XX"], coeffs=[1.0])

    ctx.log(f"Created observables: {obs1}, {obs2}")

    # Create phase sweep for parameter
    # We'll sweep theta from 0 to 2π
    phases = np.linspace(0, 2 * np.pi, 16)
    ctx.log(f"Created {len(phases)} phase values from 0 to 2π")

    return {
        "circuit": qc,
        "observable1": obs1,
        "observable2": obs2,
        "phases": phases,
        "parameter": theta,
    }


@optimize_stage()
def transpile_circuit(ctx: PatternContext) -> dict:
    """
    Optimize Stage: Transpile circuit for AerSimulator execution.

    Applies basic circuit optimizations to reduce gate count and depth
    while maintaining the circuit's functionality.
    """
    ctx.log("Loading data from map stage...")
    map_data = ctx.load_input()

    circuit = map_data["circuit"]
    original_depth = circuit.depth()
    ctx.log(f"Original circuit depth: {original_depth}")

    # Transpile for AerSimulator
    ctx.log("Transpiling circuit for AerSimulator...")
    backend = AerSimulator()
    optimized_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=1
    )

    optimized_depth = optimized_circuit.depth()
    ctx.log(f"Optimized circuit depth: {optimized_depth}")

    if optimized_depth < original_depth:
        improvement = ((original_depth - optimized_depth) / original_depth) * 100
        ctx.log(f"Achieved {improvement:.1f}% depth reduction")

    return {
        "circuit": optimized_circuit,
        "observable1": map_data["observable1"],
        "observable2": map_data["observable2"],
        "phases": map_data["phases"],
        "parameter": map_data["parameter"],
        "original_depth": original_depth,
        "optimized_depth": optimized_depth,
    }


@execute_stage()
def run_quantum_simulation(ctx: PatternContext) -> dict:
    """
    Execute Stage: Run circuit on AerSimulator with parameter sweep.

    Executes the parameterized circuit for each phase value and measures
    the expectation values for both observables.
    """
    ctx.log("Loading data from optimize stage...")
    opt_data = ctx.load_input()

    circuit = opt_data["circuit"]
    observables = [opt_data["observable1"], opt_data["observable2"]]
    phases = opt_data["phases"]
    parameter = opt_data["parameter"]

    ctx.log(f"Loaded circuit with depth {circuit.depth()}")
    ctx.log(f"Will execute with {len(phases)} different phase values")

    # Create Aer Estimator
    estimator = AerEstimator()

    results = {
        "observable1_values": [],
        "observable2_values": [],
        "phases": phases,
    }

    ctx.log(f"Running {len(phases)} phase values...")

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
            ctx.log(f"Completed {i + 1}/{len(phases)} phase values")

    # Convert to numpy arrays
    results["observable1_values"] = np.array(results["observable1_values"])
    results["observable2_values"] = np.array(results["observable2_values"])

    ctx.log(f"Completed all executions!")
    ctx.log(f"Observable 1 range: [{results['observable1_values'].min():.3f}, {results['observable1_values'].max():.3f}]")
    ctx.log(f"Observable 2 range: [{results['observable2_values'].min():.3f}, {results['observable2_values'].max():.3f}]")

    return {
        "observable1_values": results["observable1_values"],
        "observable2_values": results["observable2_values"],
        "phases": results["phases"],
    }


@post_process_stage()
def analyze_and_visualize(ctx: PatternContext) -> dict:
    """
    Post-Process Stage: Analyze CHSH results and create visualization.

    Calculates the CHSH inequality value, checks for violation of the
    classical bound, and creates visualizations of the results.
    """
    ctx.log("Loading execution results...")
    exec_data = ctx.load_input()

    phases = exec_data["phases"]
    obs1_values = exec_data["observable1_values"]
    obs2_values = exec_data["observable2_values"]

    ctx.log(f"Loaded {len(phases)} measurement results")

    # Calculate CHSH values
    ctx.log("Calculating CHSH inequality values...")
    chsh_correlation = np.abs(obs1_values + obs2_values)
    max_chsh = np.max(chsh_correlation)
    classical_bound = 2.0
    quantum_bound = 2 * np.sqrt(2)

    ctx.log(f"Maximum CHSH value: {max_chsh:.3f}")
    ctx.log(f"Classical bound: {classical_bound:.3f}")
    ctx.log(f"Quantum bound: {quantum_bound:.3f}")

    violation_detected = max_chsh > classical_bound
    if violation_detected:
        ctx.log(f"VIOLATION DETECTED! (exceeds classical bound by {max_chsh - classical_bound:.3f})", level="INFO")
    else:
        ctx.log("No violation detected", level="WARNING")

    # Create visualization
    ctx.log("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot observable expectation values
    ax1.plot(phases, obs1_values, 'o-', label='Observable 1 (ZZ)', linewidth=2, markersize=6)
    ax1.plot(phases, obs2_values, 's-', label='Observable 2 (XX)', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Phase (radians)', fontsize=12)
    ax1.set_ylabel('Expectation Value', fontsize=12)
    ax1.set_title('CHSH Inequality Test - Observable Measurements', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot CHSH correlation
    ax2.plot(phases, chsh_correlation, 'o-', color='purple', linewidth=2, markersize=6, label='|E₁ + E₂|')

    # Classical bound at 2
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')

    # Quantum bound at 2√2
    ax2.axhline(y=quantum_bound, color='green', linestyle='--', linewidth=2, label=f'Quantum Bound ({quantum_bound:.3f})')

    ax2.set_xlabel('Phase (radians)', fontsize=12)
    ax2.set_ylabel('CHSH Correlation', fontsize=12)
    ax2.set_title('CHSH Inequality Violation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    # Save plot
    plot_path = ctx.get_output_path().with_suffix('.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    ctx.log(f"Plot saved to: {plot_path}")

    # Create summary
    summary = {
        "max_chsh_value": float(max_chsh),
        "classical_bound": float(classical_bound),
        "quantum_bound": float(quantum_bound),
        "violation_detected": bool(violation_detected),
        "violation_amount": float(max_chsh - classical_bound),
        "num_measurements": int(len(phases)),
        "observable1_mean": float(np.mean(obs1_values)),
        "observable2_mean": float(np.mean(obs2_values)),
    }

    ctx.log(f"CHSH Value: {summary['max_chsh_value']:.3f}")
    ctx.log(f"Violation: {'YES' if summary['violation_detected'] else 'NO'}")
    if summary['violation_detected']:
        ctx.log(f"Amount: +{summary['violation_amount']:.3f}")

    return {
        "plot_path": str(plot_path),
        "summary": summary,
    }
