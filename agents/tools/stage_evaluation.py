"""Stage result evaluation tools."""
import pickle
from pathlib import Path
from typing import Dict, Any, Literal
import numpy as np


def evaluate_stage_results(
    stage_name: Literal["map", "optimize", "execute", "post_process"],
    results_path: str,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate quality of stage results and suggest next actions.

    Args:
        stage_name: Name of the stage to evaluate
        results_path: Path to stage output pickle file
        context: Optional context (previous stage results, expectations)

    Returns:
        Dictionary with evaluation:
        {
            "quality_score": float,  # 0-1, higher = better
            "issues": list[str],  # Any problems detected
            "suggestions": list[str],  # Recommended improvements
            "should_retry": bool,  # Whether to retry this stage
            "retry_params": dict,  # Parameters to use if retrying
        }
    """
    context = context or {}

    try:
        # Load stage results
        path = Path(results_path)
        if not path.exists():
            return {
                "error": f"Results file not found: {results_path}",
                "quality_score": 0.0,
                "should_retry": True,
            }

        with open(path, 'rb') as f:
            data = pickle.load(f)

        if stage_name == "map":
            return _evaluate_map_results(data, context)
        elif stage_name == "optimize":
            return _evaluate_optimize_results(data, context)
        elif stage_name == "execute":
            return _evaluate_execute_results(data, context)
        elif stage_name == "post_process":
            return _evaluate_post_process_results(data, context)
        else:
            return {"error": f"Unknown stage: {stage_name}"}

    except Exception as e:
        return {
            "error": f"Failed to evaluate stage results: {str(e)}",
            "quality_score": 0.0,
            "should_retry": False,
        }


def _evaluate_map_results(data: Dict, context: Dict) -> Dict[str, Any]:
    """Evaluate map stage output."""
    issues = []
    suggestions = []

    # Check circuit exists
    if "circuit" not in data:
        issues.append("Missing circuit in output")
        return {
            "quality_score": 0.0,
            "issues": issues,
            "suggestions": ["Re-run map stage"],
            "should_retry": True,
            "retry_params": {},
        }

    # Check observables
    if "observable1" not in data or "observable2" not in data:
        issues.append("Missing observables")

    # Check phase sampling
    phases = data.get("phases", [])
    if hasattr(phases, '__len__'):
        if len(phases) < 8:
            issues.append(f"Very sparse phase sampling: {len(phases)} points")
            suggestions.append("Consider increasing phase count to 16-32")
        elif len(phases) > 64:
            issues.append(f"Excessive phase sampling: {len(phases)} points")
            suggestions.append("Consider reducing to 16-32 for efficiency")

    # Calculate quality score
    quality_score = 1.0
    if issues:
        quality_score -= 0.2 * len(issues)

    return {
        "quality_score": max(quality_score, 0.0),
        "issues": issues,
        "suggestions": suggestions,
        "should_retry": len(issues) > 0,
        "retry_params": {},
    }


def _evaluate_optimize_results(data: Dict, context: Dict) -> Dict[str, Any]:
    """Evaluate optimize stage output."""
    issues = []
    suggestions = []

    original_depth = data.get("original_depth", 0)
    optimized_depth = data.get("optimized_depth", 0)

    # Check if optimization made circuit worse
    if optimized_depth > original_depth:
        issues.append(f"Optimization increased depth: {original_depth} → {optimized_depth}")
        suggestions.append("Try lower optimization level")

    # Check if optimization had no effect on simple circuit
    if original_depth == optimized_depth and original_depth < 10:
        suggestions.append("Circuit already minimal - optimization_level=0 sufficient")

    quality_score = 1.0 if not issues else 0.7

    return {
        "quality_score": quality_score,
        "issues": issues,
        "suggestions": suggestions,
        "should_retry": "increased depth" in str(issues),
        "retry_params": {"optimization_level": 0} if issues else {},
    }


def _evaluate_execute_results(data: Dict, context: Dict) -> Dict[str, Any]:
    """Evaluate execute stage output."""
    issues = []
    suggestions = []

    obs1_values = data.get("observable1_values", [])
    obs2_values = data.get("observable2_values", [])

    if len(obs1_values) == 0 or len(obs2_values) == 0:
        issues.append("No measurement results found")
        return {
            "quality_score": 0.0,
            "issues": issues,
            "suggestions": ["Re-run execute stage"],
            "should_retry": True,
            "retry_params": {},
        }

    # Calculate CHSH correlation
    chsh_values = np.abs(np.array(obs1_values) + np.array(obs2_values))
    max_chsh = float(np.max(chsh_values))

    # Evaluate CHSH violation
    classical_bound = 2.0
    quantum_bound = 2.828

    quality_score = 1.0

    if max_chsh < classical_bound:
        issues.append(f"No CHSH violation detected: {max_chsh:.3f} < {classical_bound}")
        suggestions.append("Check circuit construction or increase shots")
        quality_score = 0.5
    elif max_chsh < classical_bound + 0.3:
        issues.append(f"Weak violation: {max_chsh:.3f}")
        suggestions.append("Consider increasing shots for better statistics")
        quality_score = 0.7
    elif max_chsh > quantum_bound + 0.1:
        issues.append(f"Violation exceeds quantum bound: {max_chsh:.3f} > {quantum_bound:.3f}")
        suggestions.append("Likely simulation error or incorrect calculation")
        quality_score = 0.6

    return {
        "quality_score": quality_score,
        "issues": issues,
        "suggestions": suggestions,
        "should_retry": max_chsh < classical_bound,
        "retry_params": {"shots": 2048} if max_chsh < classical_bound else {},
        "max_chsh_value": max_chsh,
    }


def _evaluate_post_process_results(data: Dict, context: Dict) -> Dict[str, Any]:
    """Evaluate post-process stage output."""
    issues = []
    suggestions = []

    if "plot_path" not in data:
        issues.append("No plot generated")

    if "summary" not in data:
        issues.append("No summary statistics")

    quality_score = 1.0 - 0.3 * len(issues)

    return {
        "quality_score": max(quality_score, 0.0),
        "issues": issues,
        "suggestions": suggestions,
        "should_retry": False,
        "retry_params": {},
    }


# OpenAI function definition
EVALUATE_STAGE_RESULTS_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_stage_results",
        "description": "Evaluate the quality of a completed stage's results. Returns quality score, detected issues, improvement suggestions, and whether the stage should be retried.",
        "parameters": {
            "type": "object",
            "properties": {
                "stage_name": {
                    "type": "string",
                    "enum": ["map", "optimize", "execute", "post_process"],
                    "description": "Name of the stage to evaluate",
                },
                "results_path": {
                    "type": "string",
                    "description": "Path to the stage's output pickle file",
                },
            },
            "required": ["stage_name", "results_path"],
        },
    },
}
