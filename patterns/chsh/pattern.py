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

    # LLM Decision 1: How many phase samples?
    phase_count = ctx.decide(
        question="How many phase samples should I use for the CHSH inequality test?",
        options=[8, 16, 32, 64],
        context={
            "circuit_depth": qc.depth(),
            "circuit_gates": qc.num_qubits,
            "pattern": "chsh",
            "note": "More samples give better resolution but take longer"
        },
        default=16
    )

    # LLM Decision 2: Which observables to measure?
    observable_choice = ctx.decide(
        question="Which observables should I measure for the Bell state CHSH test?",
        options=["ZZ,XX", "ZZ,XX,ZX", "all_paulis"],
        context={
            "circuit_type": "Bell state (|00⟩ + |11⟩)/√2",
            "entanglement": "maximal",
            "note": "Standard CHSH uses ZZ and XX"
        },
        default="ZZ,XX"
    )

    # Create CHSH observables based on LLM decision
    observables = []
    observable_names = observable_choice.split(",")

    for obs_name in observable_names:
        obs = SparsePauliOp([obs_name], coeffs=[1.0])
        observables.append(obs)
        ctx.log(f"Created observable: {obs}")

    # Create phase sweep for parameter
    phases = np.linspace(0, 2 * np.pi, phase_count)
    ctx.log(f"Created {len(phases)} phase values from 0 to 2π")

    result = {
        "circuit": qc,
        "phases": phases,
        "parameter": theta,
    }

    # Add observables dynamically
    for i, obs in enumerate(observables):
        result[f"observable{i+1}"] = obs

    result["num_observables"] = len(observables)

    return result


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
    gate_count = len(circuit.data)
    ctx.log(f"Original circuit depth: {original_depth}, gates: {gate_count}")

    # LLM Decision: Choose optimization level based on circuit complexity
    opt_level = ctx.decide(
        question="What optimization level should I use for transpiling this circuit?",
        options=[0, 1, 2, 3],
        context={
            "circuit_depth": original_depth,
            "gate_count": gate_count,
            "num_qubits": circuit.num_qubits,
            "note": "Level 0=no optimization, 1=light, 2=medium, 3=heavy"
        },
        default=1
    )

    # Transpile for AerSimulator
    ctx.log(f"Transpiling circuit with optimization_level={opt_level}...")
    backend = AerSimulator()
    optimized_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=opt_level
    )

    optimized_depth = optimized_circuit.depth()
    ctx.log(f"Optimized circuit depth: {optimized_depth}")

    if optimized_depth < original_depth:
        improvement = ((original_depth - optimized_depth) / original_depth) * 100
        ctx.log(f"Achieved {improvement:.1f}% depth reduction")
    elif optimized_depth == original_depth:
        ctx.log("No depth reduction (circuit already optimal)")
    else:
        ctx.log(f"Circuit depth increased by {optimized_depth - original_depth} (optimization overhead)")

    # Pass through all observables dynamically
    result = {
        "circuit": optimized_circuit,
        "phases": map_data["phases"],
        "parameter": map_data["parameter"],
        "original_depth": original_depth,
        "optimized_depth": optimized_depth,
        "optimization_level": opt_level,
    }

    # Copy over observables
    num_observables = map_data.get("num_observables", 2)
    for i in range(1, num_observables + 1):
        result[f"observable{i}"] = map_data[f"observable{i}"]
    result["num_observables"] = num_observables

    return result


@execute_stage()
def run_quantum_simulation(ctx: PatternContext) -> dict:
    """
    Execute Stage: Run circuit on AerSimulator with parameter sweep.

    Executes the parameterized circuit for each phase value and measures
    the expectation values for all observables.
    """
    ctx.log("Loading data from optimize stage...")
    opt_data = ctx.load_input()

    circuit = opt_data["circuit"]
    phases = opt_data["phases"]
    parameter = opt_data["parameter"]

    # Load observables dynamically
    num_observables = opt_data.get("num_observables", 2)
    observables = []
    for i in range(1, num_observables + 1):
        observables.append(opt_data[f"observable{i}"])

    ctx.log(f"Loaded circuit with depth {circuit.depth()}")
    ctx.log(f"Will execute with {len(phases)} phase values and {num_observables} observables")

    # LLM Decision: Choose execution strategy
    strategy = ctx.decide(
        question="Should I batch measurements or run them sequentially?",
        options=["batch_all", "sequential", "adaptive"],
        context={
            "phase_count": len(phases),
            "observable_count": num_observables,
            "circuit_depth": circuit.depth(),
            "total_measurements": len(phases) * num_observables,
            "note": "Batching is faster but uses more memory; sequential is slower but safer"
        },
        default="sequential"
    )

    ctx.log(f"Using execution strategy: {strategy}")

    # Create Aer Estimator
    estimator = AerEstimator()

    # Initialize results storage
    results = {
        "phases": phases,
        "num_observables": num_observables,
    }
    for i in range(1, num_observables + 1):
        results[f"observable{i}_values"] = []

    ctx.log(f"Running {len(phases)} phase values...")

    if strategy == "batch_all":
        # Batch all measurements into single estimator call
        circuits = [circuit] * len(phases) * num_observables
        all_observables = []
        all_param_values = []

        for phase in phases:
            for obs in observables:
                all_observables.append(obs)
                all_param_values.append([phase])

        ctx.log(f"Batching {len(circuits)} measurements...")
        job = estimator.run(circuits, all_observables, all_param_values)
        result = job.result()

        # Unpack results
        idx = 0
        for phase in phases:
            for obs_idx in range(num_observables):
                exp_val = result.values[idx]
                results[f"observable{obs_idx+1}_values"].append(float(exp_val))
                idx += 1

    else:  # sequential or adaptive (treat adaptive as sequential for now)
        # Run estimator for each phase and observable
        for i, phase in enumerate(phases):
            param_values = [phase]

            for obs_idx, obs in enumerate(observables):
                job = estimator.run([circuit], [obs], [param_values])
                result = job.result()
                exp_val = result.values[0]
                results[f"observable{obs_idx+1}_values"].append(float(exp_val))

            if (i + 1) % 4 == 0:
                ctx.log(f"Completed {i + 1}/{len(phases)} phase values")

    # Convert to numpy arrays
    for i in range(1, num_observables + 1):
        results[f"observable{i}_values"] = np.array(results[f"observable{i}_values"])
        ctx.log(f"Observable {i} range: [{results[f'observable{i}_values'].min():.3f}, {results[f'observable{i}_values'].max():.3f}]")

    ctx.log(f"Completed all executions using {strategy} strategy!")

    return results


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
    num_observables = exec_data.get("num_observables", 2)

    # Load observable values dynamically
    observable_values = []
    for i in range(1, num_observables + 1):
        observable_values.append(exec_data[f"observable{i}_values"])

    ctx.log(f"Loaded {len(phases)} measurement results with {num_observables} observables")

    # Calculate CHSH values (use first two observables for CHSH)
    ctx.log("Calculating CHSH inequality values...")
    chsh_correlation = np.abs(observable_values[0] + observable_values[1])
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

    # LLM Analysis: Interpret results and recommend visualization strategy
    analysis = ctx.analyze_with_llm(
        prompt=f"""Analyze the CHSH inequality test results:

Results:
- Maximum CHSH correlation: {max_chsh:.3f}
- Classical bound: {classical_bound:.3f}
- Quantum bound: {quantum_bound:.3f}
- Violation detected: {violation_detected}
- Number of phase samples: {len(phases)}
- Number of observables measured: {num_observables}

Questions:
1. How strong is the quantum violation?
2. Are the results consistent with Bell state expectations?
3. What insights can we draw from the data?
4. Should we create basic or detailed visualization?

Provide your analysis as JSON with fields: strength_assessment, consistency_check, key_insights, visualization_type (basic/detailed).""",
        context={
            "max_chsh": float(max_chsh),
            "violation": violation_detected,
            "phase_count": len(phases),
            "num_observables": num_observables,
        },
        default_response={
            "strength_assessment": "moderate",
            "consistency_check": "consistent",
            "key_insights": "Standard CHSH violation observed",
            "visualization_type": "basic"
        }
    )

    ctx.log(f"LLM Analysis: {analysis.get('key_insights', 'N/A')}")

    # Determine visualization complexity based on LLM recommendation
    viz_type = analysis.get("visualization_type", "basic")
    ctx.log(f"Creating {viz_type} visualization...")

    if viz_type == "detailed" and num_observables > 2:
        # Create detailed multi-observable plot
        n_plots = num_observables + 1  # One per observable + CHSH correlation
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        # Plot each observable
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        for i, obs_vals in enumerate(observable_values):
            ax = axes[i] if n_plots > 1 else axes
            ax.plot(phases, obs_vals, 'o-', color=colors[i % len(colors)],
                   label=f'Observable {i+1}', linewidth=2, markersize=6)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Phase (radians)', fontsize=12)
            ax.set_ylabel('Expectation Value', fontsize=12)
            ax.set_title(f'Observable {i+1} Measurements', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # Plot CHSH correlation
        ax_chsh = axes[-1]
        ax_chsh.plot(phases, chsh_correlation, 'o-', color='purple', linewidth=2, markersize=6, label='|E₁ + E₂|')
        ax_chsh.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
        ax_chsh.axhline(y=quantum_bound, color='green', linestyle='--', linewidth=2, label=f'Quantum Bound ({quantum_bound:.3f})')
        ax_chsh.set_xlabel('Phase (radians)', fontsize=12)
        ax_chsh.set_ylabel('CHSH Correlation', fontsize=12)
        ax_chsh.set_title('CHSH Inequality Violation', fontsize=14, fontweight='bold')
        ax_chsh.legend(fontsize=10)
        ax_chsh.grid(True, alpha=0.3)
        ax_chsh.set_ylim(bottom=0)
    else:
        # Create basic two-panel plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot observable expectation values (first 2)
        observable_labels = ['ZZ', 'XX', 'ZX', 'YY', 'YX', 'ZY']
        for i in range(min(2, num_observables)):
            marker = 'o' if i == 0 else 's'
            ax1.plot(phases, observable_values[i], f'{marker}-',
                    label=f'Observable {i+1} ({observable_labels[i]})',
                    linewidth=2, markersize=6)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Phase (radians)', fontsize=12)
        ax1.set_ylabel('Expectation Value', fontsize=12)
        ax1.set_title('CHSH Inequality Test - Observable Measurements', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot CHSH correlation
        ax2.plot(phases, chsh_correlation, 'o-', color='purple', linewidth=2, markersize=6, label='|E₁ + E₂|')
        ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
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
        "num_observables": int(num_observables),
        "llm_analysis": analysis,
    }

    # Add mean values for each observable
    for i in range(num_observables):
        summary[f"observable{i+1}_mean"] = float(np.mean(observable_values[i]))

    ctx.log(f"CHSH Value: {summary['max_chsh_value']:.3f}")
    ctx.log(f"Violation: {'YES' if summary['violation_detected'] else 'NO'}")
    if summary['violation_detected']:
        ctx.log(f"Amount: +{summary['violation_amount']:.3f}")

    return {
        "plot_path": str(plot_path),
        "summary": summary,
    }
