# Agent Functions

Multi-agent system for orchestrating quantum and classical workloads using LangGraph and Qiskit Patterns.

## Overview

This project demonstrates a multi-agent architecture using LangGraph to coordinate classical and quantum computation workloads. It implements the CHSH inequality test as a proof-of-concept Qiskit pattern with four stages: Map, Optimize, Execute, and Post-Process.

### Architecture

- **Orchestrator Agent**: LangGraph supervisor coordinating the workflow
- **Classical Agent**: Executes classical stages (map, optimize, post-process) on Ray cluster
- **Quantum Agent**: Executes quantum circuits on Qiskit AerSimulator
- **State Management**: LangGraph StateGraph with workflow tracking
- **Data Flow**: File-based communication between stages

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
```

## Project Structure

```
agent-functions/
├── agents/              # Agent implementations
│   ├── classical_agent.py   # Classical workload executor
│   ├── quantum_agent.py     # Quantum workload executor
│   └── orchestrator.py      # Main LangGraph supervisor
├── config/              # Configuration settings
│   └── settings.py
├── executors/           # Execution backends
│   ├── ray_executor.py      # Ray cluster interface
│   └── qiskit_executor.py   # Qiskit simulator interface
├── patterns/            # Qiskit pattern implementations
│   └── chsh/                # CHSH inequality pattern
│       ├── map.py           # Stage 1: Create circuits
│       ├── optimize.py      # Stage 2: Transpile circuits
│       ├── execute.py       # Stage 3: Run on simulator
│       └── post_process.py  # Stage 4: Analyze results
├── workflows/           # LangGraph workflow definitions
│   └── pattern_graph.py     # Workflow state and graph
├── utils/               # Utility modules
│   ├── logging.py           # Console logging
│   └── metrics.py           # Timing metrics
├── data/                # Output directory
└── main.py              # Entry point
```

## CHSH Pattern

The CHSH (Clauser-Horne-Shimony-Holt) inequality test demonstrates quantum entanglement by showing correlations that violate classical physics bounds.

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

### Running Individual Stages

Each pattern stage can be run independently:

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

To verify the installation and run a quick test:

```bash
python main.py --pattern chsh
```

Check the `data/` directory for output files:
- `chsh_final.png`: Visualization of CHSH results
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

- **Classical stages**: Executed on Ray cluster for distributed computing
- **Quantum stages**: Executed using Qiskit AerSimulator
- **Data flow**: File-based with pickle serialization
- **Error handling**: Fail-fast with detailed error reporting

## Configuration

Edit `config/settings.py` to customize:
- Ray cluster settings (CPU count, etc.)
- Qiskit simulation parameters (shots, seed)
- File paths for outputs
- Logging configuration

## License

See LICENSE file for details.
