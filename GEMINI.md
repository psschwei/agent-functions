# Project Overview

This project is a multi-agent system for orchestrating quantum and classical workloads using LangGraph and Qiskit Patterns. It implements the CHSH inequality test as a proof-of-concept Qiskit pattern with four stages: Map, Optimize, Execute, and Post-Process.

## Key Technologies

- **Orchestration:** LangGraph
- **Quantum Computing:** Qiskit
- **Distributed Computing:** Ray
- **Programming Language:** Python

## Architecture

The system is composed of three main agents:

- **Orchestrator Agent:** A LangGraph supervisor that coordinates the workflow.
- **Classical Agent:** Executes classical stages (map, optimize, post-process) on a Ray cluster or in-process.
- **Quantum Agent:** Executes quantum circuits on the Qiskit AerSimulator.

The workflow is defined as a `StateGraph` in `workflows/pattern_graph.py`, with a linear sequence of stages: `map`, `optimize`, `execute`, and `post_process`.

The project supports two ways of defining patterns:

1.  **Decorator-Based (Recommended):** All four stages are defined in a single Python file using the `@map_stage`, `@optimize_stage`, `@execute_stage`, and `@post_process_stage` decorators. This is the modern approach and is demonstrated in `patterns/chsh/pattern.py`.
2.  **Script-Based (Legacy):** Each stage is a separate Python script. This is supported for backward compatibility.

The system automatically detects which type of pattern is being used and executes it accordingly.

# Building and Running

## Installation

To set up the project, you need Python 3.11+ and `uv` (or `pip`).

```bash
# Install dependencies
uv sync
```

## Running the CHSH Pattern

To run the CHSH inequality test, use the following command:

```bash
python main.py --pattern chsh
```

### Command-Line Options

- `--visualize`: Display a Mermaid diagram of the workflow in the console.
- `--save-diagram <PATH>`: Save the workflow diagram to a file.
- `--no-ray`: Run without initializing a Ray cluster (useful for debugging).

# Development Conventions

## Creating a New Pattern

To create a new pattern using the recommended decorator-based approach:

1.  Create a new directory for your pattern: `patterns/my_pattern/`
2.  Create a `pattern.py` file inside the new directory: `patterns/my_pattern/pattern.py`
3.  In `pattern.py`, define the four stages using the decorators:

```python
from patterns.decorators import (
    map_stage,
    optimize_stage,
    execute_stage,
    post_process_stage,
    PatternContext,
)

@map_stage()
def my_map_stage(ctx: PatternContext) -> dict:
    # ... your implementation ...
    return {"result": "data"}

@optimize_stage()
def my_optimize_stage(ctx: PatternContext) -> dict:
    # ... your implementation ...
    return {"result": "data"}

@execute_stage()
def my_execute_stage(ctx: PatternContext) -> dict:
    # ... your implementation ...
    return {"result": "data"}

@post_process_stage()
def my_post_process_stage(ctx: PatternContext) -> dict:
    # ... your implementation ...
    return {"result": "data"}
```

4.  Add your new pattern to the `choices` list in the `main.py` argument parser.
5.  Run your pattern: `python main.py --pattern my_pattern`

## Testing

The project has a comprehensive test suite with 59 unit tests. To run the tests:

```bash
# Run all tests
uv run pytest tests/

# Run pattern tests only
uv run pytest tests/patterns/ -v

# Run with coverage
uv run pytest tests/ --cov=patterns
```
