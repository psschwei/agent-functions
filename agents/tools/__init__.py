"""Orchestrator tools for LLM-based decision making."""
from .circuit_analysis import (
    analyze_circuit_complexity,
    ANALYZE_CIRCUIT_COMPLEXITY_TOOL,
)
from .stage_evaluation import (
    evaluate_stage_results,
    EVALUATE_STAGE_RESULTS_TOOL,
)
from .parameter_recommendation import (
    recommend_parameters,
    RECOMMEND_PARAMETERS_TOOL,
)
from .data_loader import (
    load_intermediate_results,
    LOAD_INTERMEDIATE_RESULTS_TOOL,
)

# Tool function registry
TOOL_FUNCTIONS = {
    "analyze_circuit_complexity": analyze_circuit_complexity,
    "evaluate_stage_results": evaluate_stage_results,
    "recommend_parameters": recommend_parameters,
    "load_intermediate_results": load_intermediate_results,
}

# OpenAI tool definitions
TOOL_DEFINITIONS = [
    ANALYZE_CIRCUIT_COMPLEXITY_TOOL,
    EVALUATE_STAGE_RESULTS_TOOL,
    RECOMMEND_PARAMETERS_TOOL,
    LOAD_INTERMEDIATE_RESULTS_TOOL,
]

__all__ = [
    "TOOL_FUNCTIONS",
    "TOOL_DEFINITIONS",
    "analyze_circuit_complexity",
    "evaluate_stage_results",
    "recommend_parameters",
    "load_intermediate_results",
]
