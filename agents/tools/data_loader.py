"""Data loading tools for orchestrator."""
import pickle
import json
from pathlib import Path
from typing import Dict, Any


def load_intermediate_results(results_path: str, format: str = "summary") -> Dict[str, Any]:
    """
    Load and deserialize intermediate results from previous stage.

    Args:
        results_path: Path to pickle file
        format: Return format - "summary" (metadata only) or "full" (complete data)

    Returns:
        Dictionary with loaded data or summary
    """
    try:
        path = Path(results_path)

        if not path.exists():
            return {"error": f"File not found: {results_path}"}

        with open(path, 'rb') as f:
            data = pickle.load(f)

        if format == "summary":
            # Return metadata summary instead of full data
            return _summarize_data(data)
        else:
            # Return full data (may be large)
            return {"data": _serialize_for_json(data)}

    except Exception as e:
        return {"error": f"Failed to load pickle: {str(e)}"}


def _summarize_data(data: Dict) -> Dict[str, Any]:
    """Create summary of data for LLM consumption."""
    summary = {
        "keys": list(data.keys()),
        "types": {k: type(v).__name__ for k, v in data.items()},
    }

    # Add specific summaries for known types
    for key, value in data.items():
        if key == "circuit":
            summary["circuit_info"] = {
                "depth": getattr(value, "depth", lambda: None)(),
                "num_qubits": getattr(value, "num_qubits", None),
            }
        elif key == "phases":
            if hasattr(value, "__len__") and hasattr(value, "__iter__"):
                try:
                    summary["phases_info"] = {
                        "count": len(value),
                        "range": [float(min(value)), float(max(value))],
                    }
                except (ValueError, TypeError):
                    pass
        elif "values" in key and hasattr(value, "__len__"):
            try:
                summary[f"{key}_info"] = {
                    "count": len(value),
                    "range": [float(min(value)), float(max(value))],
                    "mean": float(sum(value) / len(value)),
                }
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    return summary


def _serialize_for_json(obj):
    """Convert objects to JSON-serializable format."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# OpenAI function definition
LOAD_INTERMEDIATE_RESULTS_TOOL = {
    "type": "function",
    "function": {
        "name": "load_intermediate_results",
        "description": "Load intermediate results from a completed stage. Returns summary by default (keys, types, basic stats) or full data if requested.",
        "parameters": {
            "type": "object",
            "properties": {
                "results_path": {
                    "type": "string",
                    "description": "Path to pickle file containing stage results",
                },
                "format": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": "Return format: 'summary' for metadata only (default), 'full' for complete data",
                },
            },
            "required": ["results_path"],
        },
    },
}
