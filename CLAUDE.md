# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- **GitHub Organization**: psschwei
- **Repository Name**: agent-functions
- **URL**: https://github.com/psschwei/agent-functions

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

### Git Commits

All commits must be signed and GPG signed:
```bash
git commit -sS -m "commit message"
```

The `-s` flag adds a Signed-off-by line, and `-S` GPG signs the commit.

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

Patterns can be implemented in two ways:

#### Decorator-Based Patterns (Recommended)

Define all stages in a single `patterns/{pattern_name}/pattern.py` file using decorators:

```python
from patterns.decorators import map_stage, optimize_stage, execute_stage, post_process_stage, PatternContext

@map_stage()
def create_circuits(ctx: PatternContext) -> dict:
    """Create parameterized circuits and observables."""
    ctx.log("Creating circuits...")
    # Implementation
    return {"circuit": qc, "observables": obs, ...}

@optimize_stage()
def transpile_circuits(ctx: PatternContext) -> dict:
    """Transpile circuits for target backend."""
    map_data = ctx.load_input()  # Load from previous stage
    # Implementation
    return {"circuit": optimized_qc, ...}

@execute_stage()
def run_simulation(ctx: PatternContext) -> dict:
    """Execute circuits on quantum simulator."""
    opt_data = ctx.load_input()
    # Implementation
    return {"results": results, ...}

@post_process_stage()
def analyze_results(ctx: PatternContext) -> dict:
    """Analyze results and create visualizations."""
    exec_data = ctx.load_input()
    # Implementation
    return {"plot_path": str(plot_path), "summary": summary}
```

**Benefits:**
- Single file instead of four separate scripts
- Less boilerplate (no argparse, automatic I/O)
- Better IDE support (jump to definition, autocomplete)
- Type safety with PatternContext
- Automatic output saving

**PatternContext API:**
- `ctx.load_input()`: Load pickle from previous stage
- `ctx.save_output(data)`: Save pickle (automatic via decorator)
- `ctx.get_input_path()`: Get path to previous stage's output
- `ctx.get_output_path()`: Get path for this stage's output
- `ctx.log(message, level)`: Log message to orchestrator
- `ctx.get_config(key)`: Access pattern-specific config

#### Script-Based Patterns (Legacy)

Patterns live in `patterns/{pattern_name}/` with four separate scripts:

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

**Note:** The workflow automatically detects and uses decorated patterns when available, falling back to scripts if not found.

### Configuration System

All settings are centralized in `config/settings.py`:
- `RAY_CONFIG`: Ray cluster settings (CPU count, dashboard)
- `QISKIT_CONFIG`: Simulation parameters (shots, seed)
- `{PATTERN}_CONFIG`: Pattern-specific file paths
- `LANGGRAPH_CONFIG`: Workflow settings (checkpointer, recursion limit)

Import from config module: `from config import PATTERNS_DIR, CHSH_CONFIG`

## Adding New Patterns

### Decorator-Based Pattern (Recommended)

To add a new pattern (e.g., "vqe") using decorators:

1. Create `patterns/vqe/` directory
2. Create `patterns/vqe/pattern.py` with decorated stage functions:
   ```python
   from patterns.decorators import map_stage, optimize_stage, execute_stage, post_process_stage, PatternContext

   @map_stage()
   def create_vqe_circuit(ctx: PatternContext) -> dict:
       # Implementation
       return {"circuit": qc, ...}

   @optimize_stage()
   def optimize_vqe(ctx: PatternContext) -> dict:
       # Implementation
       return {"circuit": optimized_qc, ...}

   @execute_stage()
   def run_vqe(ctx: PatternContext) -> dict:
       # Implementation
       return {"results": results, ...}

   @post_process_stage()
   def analyze_vqe(ctx: PatternContext) -> dict:
       # Implementation
       return {"plot_path": str(plot_path), "summary": summary}
   ```
3. Update `main.py` argument parser to include "vqe" in pattern choices
4. Run with `python main.py --pattern vqe`

**Note:** No need to modify `config/settings.py` or `workflows/pattern_graph.py` - the workflow automatically discovers and executes decorated patterns using convention-based file paths (`data/{pattern}_{stage}_result.pkl`).

### Script-Based Pattern (Legacy)

To add a new pattern using scripts:

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
