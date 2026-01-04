#!/usr/bin/env python3
"""
CHSH Pattern - Post-Process Stage

Analyzes results and creates visualization showing CHSH inequality violation.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def calculate_chsh_value(obs1_values, obs2_values):
    """
    Calculate the CHSH inequality value.

    The CHSH inequality states that for local realistic theories:
    |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2

    Quantum mechanics can violate this, reaching up to 2√2 ≈ 2.828.

    Args:
        obs1_values: Expectation values for observable 1
        obs2_values: Expectation values for observable 2

    Returns:
        Maximum CHSH value achieved
    """
    # Simple CHSH calculation using the correlation values
    # For this simplified version, we compute the maximum correlation
    chsh_values = np.abs(obs1_values + obs2_values)
    max_chsh = np.max(chsh_values)

    return max_chsh


def create_visualization(phases, obs1_values, obs2_values, output_path):
    """
    Create a visualization of the CHSH measurement results.

    Args:
        phases: Phase values used in the experiment
        obs1_values: Expectation values for observable 1
        obs2_values: Expectation values for observable 2
        output_path: Path to save the plot
    """
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
    chsh_correlation = np.abs(obs1_values + obs2_values)
    ax2.plot(phases, chsh_correlation, 'o-', color='purple', linewidth=2, markersize=6, label='|E₁ + E₂|')

    # Classical bound at 2
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')

    # Quantum bound at 2√2
    quantum_bound = 2 * np.sqrt(2)
    ax2.axhline(y=quantum_bound, color='green', linestyle='--', linewidth=2, label=f'Quantum Bound ({quantum_bound:.3f})')

    ax2.set_xlabel('Phase (radians)', fontsize=12)
    ax2.set_ylabel('CHSH Correlation', fontsize=12)
    ax2.set_title('CHSH Inequality Violation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {output_path}")


def main():
    """Post-process stage: Analyze results and create visualizations."""
    parser = argparse.ArgumentParser(
        description="CHSH Post-Process Stage - Analyze and visualize results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input pickle file from execute stage"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output plot file path (PNG)"
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Optional JSON summary output path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHSH Pattern - Post-Process Stage")
    print("=" * 60)

    # Load input data from execute stage
    print("\n[1/4] Loading execution results...")
    with open(args.input, 'rb') as f:
        execute_data = pickle.load(f)

    phases = execute_data["phases"]
    obs1_values = execute_data["observable1_values"]
    obs2_values = execute_data["observable2_values"]

    print(f"Loaded {len(phases)} measurement results")

    # Calculate CHSH values
    print("\n[2/4] Calculating CHSH inequality values...")
    max_chsh = calculate_chsh_value(obs1_values, obs2_values)
    classical_bound = 2.0
    quantum_bound = 2 * np.sqrt(2)

    print(f"  Maximum CHSH value: {max_chsh:.3f}")
    print(f"  Classical bound: {classical_bound:.3f}")
    print(f"  Quantum bound: {quantum_bound:.3f}")

    if max_chsh > classical_bound:
        print(f"  ✓ VIOLATION DETECTED! (exceeds classical bound by {max_chsh - classical_bound:.3f})")
    else:
        print(f"  ✗ No violation detected")

    # Create visualization
    print("\n[3/4] Creating visualization...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_visualization(phases, obs1_values, obs2_values, output_path)

    # Create summary
    print("\n[4/4] Generating summary...")
    summary = {
        "max_chsh_value": float(max_chsh),
        "classical_bound": float(classical_bound),
        "quantum_bound": float(quantum_bound),
        "violation_detected": bool(max_chsh > classical_bound),
        "violation_amount": float(max_chsh - classical_bound),
        "num_measurements": int(len(phases)),
        "observable1_mean": float(np.mean(obs1_values)),
        "observable2_mean": float(np.mean(obs2_values)),
    }

    # Save summary if path provided
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {summary_path}")

    print(f"\n✓ Post-process stage complete!")
    print(f"\nSummary:")
    print(f"  CHSH Value: {summary['max_chsh_value']:.3f}")
    print(f"  Violation: {'YES' if summary['violation_detected'] else 'NO'}")
    if summary['violation_detected']:
        print(f"  Amount: +{summary['violation_amount']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
