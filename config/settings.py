"""Configuration settings for agent-functions project."""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PATTERNS_DIR = PROJECT_ROOT / "patterns"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Ray cluster configuration
RAY_CONFIG = {
    "num_cpus": 4,
    "ignore_reinit_error": True,
    "include_dashboard": False,
}

# Qiskit configuration
QISKIT_CONFIG = {
    "shots": 1024,
    "seed_simulator": 42,
}

# LLM configuration for orchestrator
LLM_CONFIG = {
    "model": "gpt-4",  # Model name for OpenAI-compatible endpoint
    "base_url": None,  # Set to litellm proxy URL, e.g., "http://localhost:4000"
    "api_key": None,  # Falls back to OPENAI_API_KEY env var
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 30,  # Request timeout in seconds
    "max_retries": 3,
}

# Orchestrator reasoning configuration
ORCHESTRATOR_CONFIG = {
    "enable_llm": False,  # Feature flag for LLM reasoning
    "enable_logging": True,  # Log all LLM decisions
    "fallback_on_error": True,  # Use defaults if LLM fails
    "max_tool_calls": 10,  # Prevent infinite loops
    "decision_log_path": DATA_DIR / "orchestrator_decisions.jsonl",
    # Phase 3: Feedback loops and iteration
    "enable_retries": True,  # Allow stage retries on poor results
    "enable_iteration": True,  # Allow multi-iteration optimization
    "max_stage_retries": 2,  # Maximum retries per stage
    "max_workflow_iterations": 3,  # Maximum workflow iterations
    "retry_on_quality_threshold": True,  # Retry if quality below threshold
}

# Mellea configuration for adaptive agents
MELLEA_CONFIG = {
    "enabled": False,  # Feature flag to enable Mellea agents
    "model_backend": "ollama",  # Backend: ollama, watsonx, huggingface, openai
    "max_retries": 2,  # Maximum adaptive retries per stage
    "stages": ["map"],  # Which stages to use Mellea for (map, optimize, post_process)
    "model_name": "llama2",  # Model name for the backend
    "temperature": 0.7,  # Temperature for LLM generation
    "evaluation_enabled": True,  # Enable result quality evaluation
    "adjustment_enabled": True,  # Enable parameter adjustment suggestions
}

# Pattern-specific settings
CHSH_CONFIG = {
    "map_output": DATA_DIR / "chsh_map_result.pkl",
    "optimize_output": DATA_DIR / "chsh_optimize_result.pkl",
    "execute_output": DATA_DIR / "chsh_execute_result.pkl",
    "post_process_output": DATA_DIR / "chsh_final.png",
    "post_process_summary": DATA_DIR / "chsh_summary.json",
}

# LangGraph configuration
LANGGRAPH_CONFIG = {
    "checkpointer_path": PROJECT_ROOT / "checkpoints.db",
    "recursion_limit": 100,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
