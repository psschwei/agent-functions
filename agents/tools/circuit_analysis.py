"""Circuit analysis tools for orchestrator."""
import pickle
from pathlib import Path
from typing import Dict, Any


def analyze_circuit_complexity(circuit_data_path: str) -> Dict[str, Any]:
    """
    Analyze quantum circuit complexity to recommend optimization strategy.

    Loads circuit from pickle file and analyzes:
    - Gate count and depth
    - Qubit count
    - Gate type distribution
    - Entanglement structure

    Args:
        circuit_data_path: Path to pickle file containing circuit data

    Returns:
        Dictionary with analysis results:
        {
            "depth": int,
            "gate_count": int,
            "num_qubits": int,
            "gate_types": dict,
            "recommended_opt_level": int,  # 0-3
            "reasoning": str,
            "complexity_score": float,  # 0-1, higher = more complex
        }
    """
    try:
        # Load circuit data
        path = Path(circuit_data_path)
        if not path.exists():
            return {"error": f"File not found: {circuit_data_path}"}

        with open(path, 'rb') as f:
            data = pickle.load(f)

        circuit = data.get("circuit")
        if circuit is None:
            return {"error": "No circuit found in data"}

        # Analyze circuit structure
        depth = circuit.depth()
        gate_count = sum(circuit.count_ops().values())
        num_qubits = circuit.num_qubits
        gate_types = dict(circuit.count_ops())

        # Calculate complexity score
        # Simple heuristic: normalized combination of depth and gate count
        depth_score = min(depth / 50.0, 1.0)  # Normalize to 0-1
        gate_score = min(gate_count / 100.0, 1.0)
        complexity_score = (depth_score + gate_score) / 2.0

        # Recommend optimization level based on complexity
        if complexity_score < 0.1:
            recommended_opt_level = 0
            reasoning = "Circuit is already minimal - no optimization needed"
        elif complexity_score < 0.3:
            recommended_opt_level = 1
            reasoning = "Simple circuit - light optimization to reduce overhead"
        elif complexity_score < 0.6:
            recommended_opt_level = 2
            reasoning = "Moderate complexity - standard optimization recommended"
        else:
            recommended_opt_level = 3
            reasoning = "Complex circuit - aggressive optimization justified"

        return {
            "depth": depth,
            "gate_count": gate_count,
            "num_qubits": num_qubits,
            "gate_types": gate_types,
            "recommended_opt_level": recommended_opt_level,
            "reasoning": reasoning,
            "complexity_score": complexity_score,
        }

    except Exception as e:
        return {"error": f"Failed to analyze circuit: {str(e)}"}


# OpenAI function definition
ANALYZE_CIRCUIT_COMPLEXITY_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_circuit_complexity",
        "description": "Analyze quantum circuit structure to recommend optimization level. Returns gate count, depth, complexity metrics, and recommended optimization level (0-3).",
        "parameters": {
            "type": "object",
            "properties": {
                "circuit_data_path": {
                    "type": "string",
                    "description": "Path to pickle file containing circuit data from map stage",
                }
            },
            "required": ["circuit_data_path"],
        },
    },
}
