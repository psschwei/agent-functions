"""Parameter recommendation tools."""
from typing import Dict, Any, Literal


def recommend_parameters(
    stage_name: Literal["map", "optimize", "execute", "post_process"],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recommend optimal parameters for an upcoming stage.

    Args:
        stage_name: Name of the stage needing parameters
        context: Context including previous stage results, pattern type, etc.

    Returns:
        Dictionary with parameter recommendations:
        {
            "parameters": dict,  # Recommended parameter values
            "reasoning": str,  # Why these parameters were chosen
            "alternatives": list[dict],  # Alternative parameter sets
            "confidence": float,  # 0-1, confidence in recommendation
        }
    """
    try:
        if stage_name == "map":
            return _recommend_map_parameters(context)
        elif stage_name == "optimize":
            return _recommend_optimize_parameters(context)
        elif stage_name == "execute":
            return _recommend_execute_parameters(context)
        elif stage_name == "post_process":
            return _recommend_post_process_parameters(context)
        else:
            return {"error": f"Unknown stage: {stage_name}"}

    except Exception as e:
        return {"error": f"Failed to recommend parameters: {str(e)}"}


def _recommend_map_parameters(context: Dict) -> Dict[str, Any]:
    """Recommend parameters for map stage."""
    pattern = context.get("pattern_name", "chsh")

    if pattern == "chsh":
        # Standard CHSH parameters
        params = {
            "phase_count": 16,
            "phase_range": [0, 6.283185307179586],  # 0 to 2π
            "observables": ["ZZ", "XX"],
        }
        reasoning = "Standard CHSH test: 16 phases over 2π with ZZ and XX observables"
        confidence = 0.9

        alternatives = [
            {
                "phase_count": 32,
                "description": "Higher resolution for detailed analysis",
                "trade_off": "2x execution time",
            },
            {
                "phase_count": 8,
                "description": "Quick exploratory run",
                "trade_off": "Lower precision",
            },
        ]
    else:
        params = {}
        reasoning = f"Unknown pattern: {pattern}"
        confidence = 0.0
        alternatives = []

    return {
        "parameters": params,
        "reasoning": reasoning,
        "alternatives": alternatives,
        "confidence": confidence,
    }


def _recommend_optimize_parameters(context: Dict) -> Dict[str, Any]:
    """Recommend parameters for optimize stage."""
    circuit_analysis = context.get("circuit_analysis", {})
    recommended_level = circuit_analysis.get("recommended_opt_level", 1)

    params = {
        "optimization_level": recommended_level,
    }

    reasoning = circuit_analysis.get("reasoning", "Standard optimization level")

    return {
        "parameters": params,
        "reasoning": reasoning,
        "alternatives": [
            {"optimization_level": 0, "description": "No optimization"},
            {"optimization_level": 3, "description": "Maximum optimization"},
        ],
        "confidence": 0.8,
    }


def _recommend_execute_parameters(context: Dict) -> Dict[str, Any]:
    """Recommend parameters for execute stage."""
    circuit_depth = context.get("circuit_depth", 10)

    # More shots for deeper circuits (more noise)
    if circuit_depth < 10:
        shots = 1024
        reasoning = "Shallow circuit - standard shot count"
    elif circuit_depth < 50:
        shots = 2048
        reasoning = "Moderate depth - increased shots for noise resilience"
    else:
        shots = 4096
        reasoning = "Deep circuit - high shot count for statistical accuracy"

    params = {
        "shots": shots,
        "execution_strategy": "batched",
    }

    return {
        "parameters": params,
        "reasoning": reasoning,
        "alternatives": [
            {"shots": shots // 2, "description": "Faster execution"},
            {"shots": shots * 2, "description": "Better statistics"},
        ],
        "confidence": 0.7,
    }


def _recommend_post_process_parameters(context: Dict) -> Dict[str, Any]:
    """Recommend parameters for post-process stage."""
    execute_results = context.get("execute_results", {})
    max_chsh = execute_results.get("max_chsh_value", 0.0)

    # Adapt visualization based on results
    if max_chsh > 2.5:
        plot_types = ["observables_vs_phase", "chsh_correlation", "detailed_stats"]
        reasoning = "Strong violation - detailed analysis warranted"
    elif max_chsh > 2.0:
        plot_types = ["observables_vs_phase", "chsh_correlation"]
        reasoning = "Moderate violation - standard plots"
    else:
        plot_types = ["observables_vs_phase", "diagnostic"]
        reasoning = "No violation - diagnostic analysis needed"

    params = {
        "plot_types": plot_types,
        "save_format": "png",
        "dpi": 150,
    }

    return {
        "parameters": params,
        "reasoning": reasoning,
        "alternatives": [],
        "confidence": 0.8,
    }


# OpenAI function definition
RECOMMEND_PARAMETERS_TOOL = {
    "type": "function",
    "function": {
        "name": "recommend_parameters",
        "description": "Recommend optimal parameters for an upcoming workflow stage based on context and previous results.",
        "parameters": {
            "type": "object",
            "properties": {
                "stage_name": {
                    "type": "string",
                    "enum": ["map", "optimize", "execute", "post_process"],
                    "description": "Name of the stage needing parameters",
                },
                "context": {
                    "type": "object",
                    "description": "Context object with pattern name, previous results, etc.",
                },
            },
            "required": ["stage_name", "context"],
        },
    },
}
