# Agent Functions

Multi-agent system for orchestrating quantum and classical workloads using LangGraph and Qiskit Patterns.

## Overview

This project demonstrates a multi-agent architecture using LangGraph to coordinate classical and quantum computation workloads. It implements the CHSH inequality test as a proof-of-concept Qiskit pattern with four stages: Map, Optimize, Execute, and Post-Process.

### Key Features

- **Decorator-Based Patterns**: Define complete patterns in a single Python file using `@map_stage`, `@optimize_stage`, `@execute_stage`, and `@post_process_stage` decorators
- **Multi-Agent Orchestration**: LangGraph supervisor coordinating classical and quantum agents
- **Agentic Mode**: LLM-powered orchestrator with autonomous decision-making (`--agentic` flag)
- **Mellea Integration**: Adaptive execution with LLM-based result evaluation and parameter optimization
- **Dual-Mode Support**: Automatic fallback between decorator-based and script-based patterns
- **Type-Safe Context**: PatternContext provides typed I/O, logging, and config access

### Architecture

- **Orchestrator Agent**: LangGraph supervisor coordinating the workflow (basic or agentic)
- **Classical Agent**: Executes classical stages (map, optimize, post-process) on Ray cluster or in-process
- **Mellea Classical Agent**: Enhanced classical agent with adaptive execution and LLM-based evaluation
- **Quantum Agent**: Executes quantum circuits on Qiskit AerSimulator
- **Pattern System**: Decorator-based or script-based pattern implementations
- **State Management**: LangGraph StateGraph with workflow tracking
- **Data Flow**: File-based communication between stages with pickle serialization

## Installation

### Prerequisites

- Python 3.11+
- uv package manager (recommended) or pip

### Setup with uv

```bash
# Install dependencies
uv sync

# Or if you don't have uv, use pip
pip install -e .
```

## Usage

### Run the CHSH Pattern

Execute the complete CHSH inequality test:

```bash
python main.py --pattern chsh
```

### Command-Line Options

```bash
# Display workflow visualization
python main.py --pattern chsh --visualize

# Save workflow diagram to file
python main.py --pattern chsh --save-diagram workflow.md

# Run without Ray cluster (for debugging)
python main.py --pattern chsh --no-ray

# Use Mellea-enhanced classical agent with adaptive execution
python main.py --pattern chsh --mellea

# Use agentic orchestrator with LLM reasoning
python main.py --pattern chsh --agentic

# Combine Mellea with agentic orchestrator for maximum adaptability
python main.py --pattern chsh --mellea --agentic
```

## Project Structure

```
agent-functions/
├── agents/              # Agent implementations
│   ├── classical_agent.py       # Classical workload executor
│   ├── quantum_agent.py         # Quantum workload executor
│   ├── orchestrator.py          # Basic LangGraph supervisor
│   ├── agentic_orchestrator.py  # LLM-powered orchestrator
│   ├── llm_client.py            # LLM client wrapper
│   ├── mellea_classical_agent.py  # Adaptive classical agent
│   └── tools/                   # LLM tools for agentic orchestrator
│       ├── circuit_analysis.py      # Circuit complexity analysis
│       ├── stage_evaluation.py      # Stage result evaluation
│       ├── parameter_recommendation.py  # Parameter suggestions
│       └── data_loader.py           # Intermediate data access
├── config/              # Configuration settings
│   └── settings.py
├── executors/           # Execution backends
│   ├── ray_executor.py      # Ray cluster interface
│   └── qiskit_executor.py   # Qiskit simulator interface
├── patterns/            # Qiskit pattern implementations
│   ├── decorators.py        # Decorator-based pattern system
│   ├── loader.py            # Pattern discovery and loading
│   └── chsh/                # CHSH inequality pattern
│       ├── pattern.py       # Decorator-based implementation (recommended)
│       ├── map.py           # Legacy: Stage 1 script
│       ├── optimize.py      # Legacy: Stage 2 script
│       ├── execute.py       # Legacy: Stage 3 script
│       └── post_process.py  # Legacy: Stage 4 script
├── workflows/           # LangGraph workflow definitions
│   └── pattern_graph.py     # Workflow state and graph
├── tests/               # Test suite
│   └── patterns/            # Pattern system tests
├── utils/               # Utility modules
│   ├── logging.py           # Console logging
│   └── metrics.py           # Timing metrics
├── data/                # Output directory
└── main.py              # Entry point
```

## Pattern Development

This project supports two approaches for implementing Qiskit patterns:

### Decorator-Based Patterns (Recommended)

Define all four stages in a single file using Python decorators:

```python
from patterns.decorators import (
    map_stage, optimize_stage, execute_stage, post_process_stage,
    PatternContext
)

@map_stage()
def create_circuit(ctx: PatternContext) -> dict:
    """Create parameterized circuit and observables."""
    ctx.log("Creating circuit...")
    # Implementation
    return {"circuit": qc, "observables": obs, ...}

@optimize_stage()
def transpile_circuit(ctx: PatternContext) -> dict:
    """Transpile circuit for target backend."""
    map_data = ctx.load_input()  # Automatic I/O
    # Implementation
    return {"circuit": optimized_qc, ...}

@execute_stage()
def run_simulation(ctx: PatternContext) -> dict:
    """Execute on quantum simulator."""
    opt_data = ctx.load_input()
    # Implementation
    return {"results": results, ...}

@post_process_stage()
def analyze_results(ctx: PatternContext) -> dict:
    """Analyze and visualize results."""
    exec_data = ctx.load_input()
    # Implementation
    return {"plot_path": str(path), "summary": summary}
```

**Benefits:**
- Single file instead of four separate scripts
- Less boilerplate (no argparse, automatic I/O)
- Better IDE support (autocomplete, jump to definition)
- Type safety with PatternContext
- Easier to test and maintain

### Script-Based Patterns (Legacy)

Traditional approach with four separate Python scripts (map.py, optimize.py, execute.py, post_process.py). Still supported for backward compatibility.

The workflow automatically detects decorated patterns and falls back to scripts if not found.

## CHSH Pattern

The CHSH (Clauser-Horne-Shimony-Holt) inequality test demonstrates quantum entanglement by showing correlations that violate classical physics bounds.

The CHSH pattern is implemented using the decorator-based approach in `patterns/chsh/pattern.py` (280 lines vs 400+ for script-based).

### Stages

1. **Map**: Creates a parameterized Bell state circuit and defines observables
2. **Optimize**: Transpiles the circuit for the target backend (AerSimulator)
3. **Execute**: Runs the circuit with parameter sweep on quantum simulator
4. **Post-Process**: Analyzes results and generates visualization showing CHSH violation

### Expected Output

- Plot showing expectation values and CHSH correlation
- Demonstration of violation exceeding classical bound of 2.0
- JSON summary with CHSH values and statistics

## Development

### Creating a New Pattern

To create a new pattern using the decorator approach:

1. Create a directory: `patterns/my_pattern/`
2. Create `patterns/my_pattern/pattern.py`:

```python
from patterns.decorators import map_stage, optimize_stage, execute_stage, post_process_stage, PatternContext

@map_stage()
def my_map_stage(ctx: PatternContext) -> dict:
    # Your implementation
    return {"circuit": circuit, ...}

@optimize_stage()
def my_optimize_stage(ctx: PatternContext) -> dict:
    input_data = ctx.load_input()
    # Your implementation
    return {"circuit": optimized_circuit, ...}

@execute_stage()
def my_execute_stage(ctx: PatternContext) -> dict:
    input_data = ctx.load_input()
    # Your implementation
    return {"results": results, ...}

@post_process_stage()
def my_post_process_stage(ctx: PatternContext) -> dict:
    input_data = ctx.load_input()
    # Your implementation
    return {"plot_path": str(plot_path), "summary": summary}
```

3. Add "my_pattern" to `main.py` argument parser choices
4. Run: `python main.py --pattern my_pattern`

### Running Individual Stages (Legacy)

For script-based patterns, each stage can be run independently:

```bash
# Map stage
python patterns/chsh/map.py --output data/map.pkl

# Optimize stage
python patterns/chsh/optimize.py --input data/map.pkl --output data/optimize.pkl

# Execute stage
python patterns/chsh/execute.py --input data/optimize.pkl --output data/execute.pkl

# Post-process stage
python patterns/chsh/post_process.py --input data/execute.pkl --output data/result.png --summary data/summary.json
```

### Testing

Run the complete test suite:

```bash
# Run all tests
uv run pytest tests/

# Run pattern tests only
uv run pytest tests/patterns/ -v

# Run with coverage
uv run pytest tests/ --cov=patterns
```

To verify the installation and run an end-to-end test:

```bash
python main.py --pattern chsh
```

Check the `data/` directory for output files:
- `chsh_post_process_result.png`: Visualization of CHSH results
- `chsh_summary.json`: Numerical summary

## Architecture Details

### LangGraph Workflow

The orchestrator uses LangGraph's StateGraph to manage workflow execution:

```
[Start] → [Map] → [Optimize] → [Execute] → [Post-Process] → [End]
           ↓          ↓            ↓              ↓
      Classical   Classical    Quantum        Classical
        Agent      Agent        Agent          Agent
```

### State Management

The workflow state tracks:
- Current stage and status
- File paths for intermediate results
- Timing metrics for each stage
- Error messages and logs

### Execution Model

- **Classical stages**: Executed on Ray cluster for distributed computing (or in-process for decorated patterns)
- **Quantum stages**: Executed using Qiskit AerSimulator
- **Data flow**: File-based with pickle serialization
- **Error handling**: Fail-fast with detailed error reporting
- **Pattern execution**: Automatic detection of decorated vs. script-based patterns with fallback support

### Agentic Orchestrator

The `--agentic` flag enables an LLM-powered orchestrator that provides autonomous decision-making:

```bash
python main.py --pattern chsh --agentic
```

**Capabilities:**
- **Circuit Analysis**: Recommends optimization strategies based on circuit complexity
- **Result Evaluation**: Assesses stage output quality and completeness
- **Parameter Recommendations**: Suggests optimal parameters for upcoming stages
- **Retry Decisions**: Determines when to retry failed or suboptimal stages
- **Workflow Control**: Decides when to terminate vs. iterate the workflow

**Available Tools:**
- `analyze_circuit_complexity()` - Analyze quantum circuit to recommend optimization strategies
- `evaluate_stage_results()` - Assess quality of stage output
- `recommend_parameters()` - Suggest parameter values for stages
- `load_intermediate_results()` - Access intermediate data for analysis

**Configuration** (in `config/settings.py`):
```python
LLM_CONFIG = {
    "model": "gpt-4",  # Model name for OpenAI-compatible endpoint
    "base_url": None,  # Set to litellm proxy URL
    "temperature": 0.7,
    "max_tokens": 2000,
}

ORCHESTRATOR_CONFIG = {
    "enable_llm": False,  # Enable LLM reasoning
    "enable_retries": True,  # Allow stage retries on poor results
    "max_stage_retries": 2,  # Maximum retries per stage
    "max_workflow_iterations": 3,  # Maximum workflow iterations
}
```

### Mellea Integration

Mellea provides adaptive execution with LLM-based result evaluation and parameter optimization:

**Features:**
- **Multiple Backends**: ollama, watsonx, huggingface, openai
- **Result Quality Evaluation**: LLM-based assessment of stage outputs
- **Adaptive Parameter Adjustment**: Automatic suggestions for improving results
- **Retry Logic**: Up to N adaptive retries with LLM-guided improvements
- **Graceful Fallback**: Automatic fallback to standard agent if Mellea unavailable

**Configuration** (in `config/settings.py`):
```python
MELLEA_CONFIG = {
    "enabled": False,  # Feature flag
    "model_backend": "ollama",  # Backend: ollama, watsonx, huggingface, openai
    "max_retries": 2,  # Maximum adaptive retries per stage
    "stages": ["map"],  # Which stages to use Mellea for
    "evaluation_enabled": True,  # Enable result quality evaluation
    "adjustment_enabled": True,  # Enable parameter adjustment suggestions
}
```

### PatternContext API

For decorator-based patterns, the `PatternContext` provides:

- **I/O Methods**:
  - `ctx.load_input()`: Load pickle from previous stage
  - `ctx.save_output(data)`: Save pickle (automatic via decorator)
  - `ctx.get_input_path()`: Get path to previous stage's output
  - `ctx.get_output_path()`: Get path for this stage's output

- **Logging**:
  - `ctx.log(message, level="INFO")`: Log execution messages
  - `ctx.get_logs()`: Retrieve all logged messages

- **Configuration**:
  - `ctx.get_config(key, default=None)`: Access pattern-specific config
  - `ctx.state`: Direct access to workflow state dictionary

## Features

### Dual-Mode Pattern Support

The system supports both decorator-based and script-based patterns:

- **Automatic detection**: Workflow checks for decorated patterns first
- **Graceful fallback**: Falls back to scripts if decorated pattern not found
- **Backward compatible**: Existing script-based patterns continue to work
- **Migration friendly**: Patterns can be migrated incrementally

### Decorator Benefits

Compared to script-based patterns, decorators provide:

- **30% less code**: Single file vs. four separate scripts
- **No boilerplate**: Automatic argparse, I/O handling, and output saving
- **Better tooling**: IDE autocomplete, jump-to-definition, type checking
- **Type safety**: PatternContext provides typed access to state and methods
- **Easier testing**: Test functions directly without subprocess overhead
- **Clear dependencies**: Explicit stage dependencies in decorator parameters

### Test Coverage

Comprehensive test suite with 59 unit tests:

- **test_decorators.py**: Decorator registration and metadata (17 tests)
- **test_pattern_context.py**: I/O, logging, and config access (20 tests)
- **test_pattern_loader.py**: Pattern discovery and validation (22 tests)
- **100% pass rate**: All tests passing with proper fixtures

## Configuration

Edit `config/settings.py` to customize:

- **RAY_CONFIG**: Ray cluster settings (CPU count, dashboard)
- **QISKIT_CONFIG**: Simulation parameters (shots, seed)
- **LLM_CONFIG**: LLM settings for agentic orchestrator (model, endpoint, temperature)
- **ORCHESTRATOR_CONFIG**: Workflow settings (retries, iterations, quality thresholds)
- **MELLEA_CONFIG**: Adaptive agent settings (backend, stages, evaluation)
- **CHSH_CONFIG**: Pattern-specific file paths
- **LANGGRAPH_CONFIG**: Workflow checkpointing and recursion limits

## License

See LICENSE file for details.
