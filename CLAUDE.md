# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent system using LangGraph to orchestrate classical and quantum workloads for Qiskit patterns. The CHSH inequality test serves as the proof-of-concept implementation.

## Development Commands

### Setup
```bash
uv sync                    # Install dependencies (preferred)
pip install -e .           # Alternative installation method
```

### Running the System
```bash
# Run complete CHSH pattern
python main.py --pattern chsh

# Debug without Ray cluster
python main.py --pattern chsh --no-ray

# Visualize workflow
python main.py --pattern chsh --visualize

# Save workflow diagram
python main.py --pattern chsh --save-diagram workflow.md
```

### Running Individual Pattern Stages
Each stage can be executed independently for debugging:
```bash
python patterns/chsh/map.py --output data/map.pkl
python patterns/chsh/optimize.py --input data/map.pkl --output data/optimize.pkl
python patterns/chsh/execute.py --input data/optimize.pkl --output data/execute.pkl
python patterns/chsh/post_process.py --input data/execute.pkl --output data/result.png --summary data/summary.json
```

### Linting and Formatting
```bash
uv run black .             # Format code
uv run ruff check .        # Lint code
```

## Architecture Overview

### Core Multi-Agent Pattern

The system uses a **supervisor pattern** with LangGraph coordinating three types of agents:

1. **Orchestrator** (`agents/orchestrator.py`): Main supervisor using LangGraph's StateGraph
2. **ClassicalAgent** (`agents/classical_agent.py`): Executes classical stages on Ray cluster
3. **QuantumAgent** (`agents/quantum_agent.py`): Executes quantum circuits on Qiskit AerSimulator

### Workflow Execution Model

```
[Orchestrator initializes StateGraph]
         ↓
[Map Stage] → [Optimize Stage] → [Execute Stage] → [Post-Process Stage]
    ↓              ↓                    ↓                   ↓
Classical      Classical            Quantum             Classical
 Agent          Agent               Agent               Agent
    ↓              ↓                    ↓                   ↓
[Ray Task]     [Ray Task]         [Qiskit Aer]        [Ray Task]
```

Each stage is defined as a **node function** in `workflows/pattern_graph.py` that:
- Updates the `PatternState` (current stage, status)
- Instantiates the appropriate agent
- Executes the stage script via executor
- Updates state with results (file paths, timings, errors)

### State Management

The `PatternState` TypedDict in `workflows/pattern_graph.py` is the **single source of truth** for workflow execution:
- `current_stage`: Which stage is executing
- `stage_status`: Status dict for all stages (pending/running/complete/failed)
- `stage_timings`: Duration of each stage
- `{stage}_output`: File paths to intermediate results
- `errors`: List of error messages
- `messages`: LangGraph message history

State flows through the graph and is **updated by each node function**, not by agents directly.

### Data Flow Architecture

**File-based communication** between stages using pickle serialization:
- Each stage writes output to `data/{pattern}_{stage}_result.pkl`
- Next stage reads from previous stage's output file
- Configuration in `config/settings.py` defines all file paths
- Agents receive script paths and I/O paths, execute via subprocess

This design allows:
- **Independent testing** of each stage script
- **Resume from failure** by re-running from any stage
- **Inspection of intermediate results** by loading pickle files

### Execution Backends

Two executor modules abstract the execution environments:

**Ray Executor** (`executors/ray_executor.py`):
- `execute_python_script`: Ray remote function that runs scripts as subprocess
- `run_script_on_ray`: Submits script to Ray cluster and waits for result
- Used by ClassicalAgent for map, optimize, post-process stages

**Qiskit Executor** (`executors/qiskit_executor.py`):
- `run_estimator`: Execute circuits with observables using AerEstimator
- `run_sampler`: Execute circuits and return measurement counts
- `execute_circuit`: Direct circuit execution on AerSimulator
- Uses `qiskit_aer.primitives` (not deprecated `qiskit.primitives.BackendEstimator`)

### Pattern Implementation Structure

Patterns live in `patterns/{pattern_name}/` with four stages:

1. **map.py**: Creates parameterized circuits and observables
   - Takes: command-line args for output path
   - Produces: pickle with circuit, observables, parameters

2. **optimize.py**: Transpiles circuits for target backend
   - Takes: map.py output pickle
   - Produces: pickle with optimized circuit

3. **execute.py**: Runs circuits on quantum simulator
   - Takes: optimize.py output pickle
   - Produces: pickle with measurement results/expectation values

4. **post_process.py**: Analyzes results and generates outputs
   - Takes: execute.py output pickle
   - Produces: visualization (PNG) and summary (JSON)

Each script is **standalone** with argparse CLI and can be run independently.

### Configuration System

All settings are centralized in `config/settings.py`:
- `RAY_CONFIG`: Ray cluster settings (CPU count, dashboard)
- `QISKIT_CONFIG`: Simulation parameters (shots, seed)
- `{PATTERN}_CONFIG`: Pattern-specific file paths
- `LANGGRAPH_CONFIG`: Workflow settings (checkpointer, recursion limit)

Import from config module: `from config import PATTERNS_DIR, CHSH_CONFIG`

## Adding New Patterns

To add a new pattern (e.g., "vqe"):

1. Create `patterns/vqe/` directory with four stage scripts (map.py, optimize.py, execute.py, post_process.py)
2. Add `VQE_CONFIG` to `config/settings.py` with file paths
3. Update `workflows/pattern_graph.py` to use pattern-agnostic paths or add VQE-specific nodes
4. Update `main.py` argument parser to include new pattern choice

The existing workflow will handle orchestration if stages follow the same I/O contract (pickle files with consistent keys).

## Qiskit Version Compatibility

**Important**: Use `qiskit_aer.primitives` (Estimator, Sampler), not `qiskit.primitives.BackendEstimator`:
- `from qiskit_aer.primitives import Estimator as AerEstimator`
- Call as: `estimator.run([circuit], [observable], [param_values])`
- Result access: `result.values[0]` (not `result[0].data.evs`)

This avoids deprecation warnings and API breaking changes in Qiskit 1.0+.

## Python Version

Project targets Python 3.11+ but is tested with Python 3.13. The `requires-python = ">=3.11,<3.14"` constraint in pyproject.toml ensures compatibility.
