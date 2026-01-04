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
