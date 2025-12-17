# Agent Functions Prototype - Implementation Plan

## Overview
Build a multi-agent system using LangGraph to orchestrate classical and quantum workloads for executing Qiskit Patterns. The prototype will demonstrate the CHSH inequality example with four stages: Map, Optimize, Execute, and Post-Process.

## Architecture Decisions

### Framework: LangGraph Supervisor Pattern
- **Orchestrator Agent**: Main supervisor using LangGraph's supervisor pattern
- **Classical Agent**: Subordinate agent for classical workloads (map, optimize, post-process stages)
- **Quantum Agent**: Subordinate agent for quantum execution (execute stage)
- **State Management**: LangGraph StateGraph with file-based checkpointing
- **Communication**: Shared state dictionary + local filesystem for large data artifacts

### Quantum Execution: Direct Qiskit (Not MCP)
Research shows Qiskit MCP servers don't provide simulator capabilities—they're for transpilation and IBM Quantum hardware. For the prototype, we'll use Qiskit directly with `qiskit_aer.AerSimulator`.

**CHSH Pattern Approach**: Simplified version using AerSimulator only. No real IBM Quantum hardware integration, no ISA transpilation. Focus on workflow orchestration.

### Classical Compute: Local Ray Cluster
Start with local Ray cluster using `ray.init()`. Future versions can extend to cloud Ray clusters.

### Data Flow: File-Based (Standalone Scripts)
- Each stage is a standalone Python script with file I/O
- Scripts read input from pickle/JSON files and write output files
- Shared state tracks file paths and stage completion status
- Scripts can be tested independently without agent framework

### Error Handling: Basic (Fail Fast)
- Simple try/catch blocks in each agent
- Errors logged to state.errors list
- Execution stops on first failure
- Focus on surfacing issues quickly for debugging

### Observability Features
1. **Console Logging**: Print statements showing agent activity and stage transitions
2. **Graph Visualization**: Export LangGraph workflow diagram using `get_graph().draw_mermaid()`
3. **Timing Metrics**: Track execution time for each stage using `time.perf_counter()`

## Project Structure

```
agent-functions/
├── README.md
├── pyproject.toml          # uv-managed dependencies
├── .python-version          # Python version for uv
├── config/
│   └── settings.py           # Configuration (Ray, file paths, etc.)
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py       # Main LangGraph supervisor
│   ├── classical_agent.py    # Classical workload executor
│   └── quantum_agent.py      # Quantum workload executor
├── executors/
│   ├── __init__.py
│   ├── ray_executor.py       # Ray cluster management
│   └── qiskit_executor.py    # Qiskit simulator interface
├── patterns/
│   ├── __init__.py
│   └── chsh/
│       ├── map.py            # Stage 1: Create Bell state circuits
│       ├── optimize.py       # Stage 2: Transpile for hardware
│       ├── execute.py        # Stage 3: Run on simulator
│       └── post_process.py   # Stage 4: Analyze results
├── workflows/
│   ├── __init__.py
│   └── pattern_graph.py      # LangGraph state and workflow definition
├── data/
│   └── .gitkeep              # Workspace for intermediate files
├── utils/
│   ├── __init__.py
│   ├── logging.py            # Console logging utilities
│   └── metrics.py            # Timing/metrics tracking
└── main.py                   # Entry point
```

## Implementation Phases

### Phase 1: Core Infrastructure
**Files to create:**
- `pyproject.toml`: Project metadata and dependencies (langgraph, langchain, qiskit, qiskit-aer, ray, numpy, matplotlib)
- `.python-version`: Python 3.11 (for uv)
- `config/settings.py`: Configuration for data paths, Ray settings
- `executors/ray_executor.py`: Simple Ray cluster init/shutdown
- `executors/qiskit_executor.py`: Wrapper for AerSimulator

**Goal:** Establish project structure and verify Ray/Qiskit work locally.

### Phase 2: Pattern Scripts (CHSH Example)
**Files to create (standalone scripts with file I/O):**

- `patterns/chsh/map.py`: Create Bell state with parameterized rotation
  - Input: Command-line arg for output path
  - Output: Pickle file containing {circuit, observable1, observable2, phases}
  - Implementation: Simplified Bell state, no hardware-specific setup

- `patterns/chsh/optimize.py`: Basic circuit optimization
  - Input: Pickle file from map stage
  - Output: Pickle file with optimized circuit
  - Implementation: Simple transpilation for AerSimulator (no ISA, no real backend)

- `patterns/chsh/execute.py`: Run on AerSimulator
  - Input: Pickle file from optimize stage
  - Output: Pickle file with estimator results
  - Implementation: Use qiskit.primitives.Estimator with AerSimulator backend

- `patterns/chsh/post_process.py`: Analyze and plot CHSH violation
  - Input: Pickle file from execute stage
  - Output: PNG plot + JSON summary with CHSH values
  - Implementation: Plot expectation values showing violation of classical bound

**Goal:** Working end-to-end Qiskit pattern that can be run manually via `python map.py --output data/map.pkl`, etc.

### Phase 3: Agent Executors
**Files to create:**
- `agents/classical_agent.py`:
  - Receives: Script path + input file path
  - Executes: Script on Ray cluster using `ray.remote`
  - Returns: Output file path + execution status

- `agents/quantum_agent.py`:
  - Receives: Execute script path + input file path
  - Executes: Script using qiskit_executor
  - Returns: Results file path + execution status

**Goal:** Agents can execute individual pattern stages.

### Phase 4: LangGraph Orchestrator
**Files to create:**
- `workflows/pattern_graph.py`:
  - Define `PatternState` (TypedDict with stage status, file paths, errors)
  - Create nodes for each agent (classical_map, classical_optimize, quantum_execute, classical_post_process)
  - Define conditional edges based on stage completion
  - Use `StateGraph` with supervisor routing logic

- `agents/orchestrator.py`:
  - Initialize LangGraph supervisor
  - Route tasks to classical vs quantum agents
  - Handle errors and retry logic
  - Track overall pattern execution

**Goal:** Orchestrator coordinates all four stages autonomously.

### Phase 5: Entry Point & Integration
**Files to create:**
- `main.py`:
  - CLI interface (argparse)
  - Initialize Ray cluster
  - Load pattern configuration
  - Run orchestrator with LangGraph
  - Display results and timing metrics
  - Export workflow graph visualization

- `utils/logging.py`:
  - Console logger with stage markers
  - Pretty-print agent activity

- `utils/metrics.py`:
  - Timing decorator for stage execution
  - Metrics summary display

**Goal:** Single command runs entire CHSH pattern with multi-agent orchestration, logging, and visualizations.

## State Schema

```python
from typing import TypedDict, Literal, Optional

class PatternState(TypedDict):
    # Input configuration
    pattern_name: str  # e.g., "chsh"

    # Stage tracking
    current_stage: Literal["map", "optimize", "execute", "post_process", "complete"]

    # File paths for stage I/O
    map_output: Optional[str]
    optimize_output: Optional[str]
    execute_output: Optional[str]
    post_process_output: Optional[str]

    # Execution metadata
    errors: list[str]
    stage_status: dict[str, Literal["pending", "running", "complete", "failed"]]
    stage_timings: dict[str, float]  # Execution time in seconds

    # Agent messages/logs
    messages: list[dict]
```

## Workflow Logic

```
[Start]
   ↓
[Orchestrator decides: map stage → Classical Agent]
   ↓
[Classical Agent runs map.py]
   ↓
[State updated: map_output = "data/chsh_map_result.pkl"]
   ↓
[Orchestrator decides: optimize stage → Classical Agent]
   ↓
[Classical Agent runs optimize.py with map_output as input]
   ↓
[State updated: optimize_output = "data/chsh_optimize_result.pkl"]
   ↓
[Orchestrator decides: execute stage → Quantum Agent]
   ↓
[Quantum Agent runs execute.py with optimize_output as input]
   ↓
[State updated: execute_output = "data/chsh_execute_result.pkl"]
   ↓
[Orchestrator decides: post_process stage → Classical Agent]
   ↓
[Classical Agent runs post_process.py with execute_output as input]
   ↓
[State updated: post_process_output = "data/chsh_final.png"]
   ↓
[Complete]
```

## Key Implementation Details

### Ray Integration
```python
# In ray_executor.py
import ray

def init_ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4)  # Local cluster
    return ray

@ray.remote
def execute_python_script(script_path: str, input_path: str, output_path: str):
    # Load input data
    # Execute script logic
    # Save output data
    return {"status": "success", "output_path": output_path}
```

### Qiskit Simulator Wrapper
```python
# In qiskit_executor.py
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator

def run_estimator(circuit, observables, parameters):
    simulator = AerSimulator()
    estimator = Estimator()
    # Bind parameters and run
    job = estimator.run([(circuit, observables, parameters)])
    return job.result()
```

### LangGraph Supervisor Setup
```python
# In orchestrator.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def create_pattern_workflow():
    workflow = StateGraph(PatternState)

    # Add nodes
    workflow.add_node("map_stage", classical_agent_node)
    workflow.add_node("optimize_stage", classical_agent_node)
    workflow.add_node("execute_stage", quantum_agent_node)
    workflow.add_node("post_process_stage", classical_agent_node)

    # Add edges
    workflow.set_entry_point("map_stage")
    workflow.add_edge("map_stage", "optimize_stage")
    workflow.add_edge("optimize_stage", "execute_stage")
    workflow.add_edge("execute_stage", "post_process_stage")
    workflow.add_edge("post_process_stage", END)

    # Compile with checkpointer
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    return workflow.compile(checkpointer=checkpointer)
```

## Testing Strategy

1. **Manual Script Tests**: Run each CHSH script standalone with sample inputs
   ```bash
   python patterns/chsh/map.py --output data/map.pkl
   python patterns/chsh/optimize.py --input data/map.pkl --output data/optimize.pkl
   # etc.
   ```

2. **Agent Executor Tests**: Test classical/quantum agents with simple scripts

3. **End-to-End Test**: Run full CHSH pattern through orchestrator
   ```bash
   python main.py --pattern chsh
   ```

4. **Verification Checks**:
   - CHSH inequality violation in output plot (values exceed ±2)
   - Graph visualization generated correctly
   - Timing metrics displayed in console
   - All intermediate files created in data/

## Success Criteria

- ✅ All four CHSH pattern stages execute successfully
- ✅ Orchestrator correctly delegates classical vs quantum workloads
- ✅ Ray cluster executes classical Python scripts
- ✅ Qiskit simulator produces valid expectation values
- ✅ Final plot shows CHSH inequality violation (values exceed ±2)
- ✅ LangGraph state checkpointing works (can resume if interrupted)
- ✅ Console logging shows clear stage progression
- ✅ Workflow graph visualization generated (mermaid diagram or PNG)
- ✅ Timing metrics displayed showing duration of each stage

## Future Enhancements (Not in Prototype)

- Break single monolithic scripts into pattern stages automatically
- Parallel execution of independent stages
- Cloud Ray cluster deployment
- Real IBM Quantum hardware integration via Qiskit MCP
- Dynamic resource allocation based on workload
- Support for additional Qiskit patterns beyond CHSH
