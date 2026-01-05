# Qiskit Pattern Decorators Implementation Plan

## Executive Summary

This document outlines a plan to add Python decorators for Qiskit pattern stages, enabling patterns to be defined in a single file instead of four separate scripts. The orchestrator will discover and execute decorated functions dynamically, simplifying pattern development while maintaining the current execution model.

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Proposed Decorator Design](#2-proposed-decorator-design)
3. [Pattern Discovery Mechanism](#3-pattern-discovery-mechanism)
4. [Execution Model Changes](#4-execution-model-changes)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Backward Compatibility Strategy](#6-backward-compatibility-strategy)
7. [Implementation Phases](#7-implementation-phases)
8. [Trade-offs and Considerations](#8-trade-offs-and-considerations)
9. [Example Usage](#9-example-usage)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Current Architecture Analysis

### 1.1 Current Pattern Structure

Each pattern currently consists of **four separate Python scripts**:

```
patterns/chsh/
├── map.py            # Creates circuits and observables
├── optimize.py       # Transpiles circuits
├── execute.py        # Runs quantum simulation
└── post_process.py   # Analyzes results and visualizes
```

### 1.2 Current Execution Flow

1. **Orchestrator** (`agents/orchestrator.py`) initializes LangGraph workflow
2. **Workflow nodes** (`workflows/pattern_graph.py`) define stage execution:
   - Each node instantiates an agent (ClassicalAgent or QuantumAgent)
   - Agent receives script path and I/O paths
   - Agent executes script as subprocess via executor
3. **Executors** (`executors/ray_executor.py`, `executors/qiskit_executor.py`) run scripts
4. **State management** via `PatternState` TypedDict tracks progress

### 1.3 Current Limitations

- **File proliferation**: Each pattern requires 4 separate files
- **Navigation overhead**: Understanding a pattern requires jumping between files
- **Duplication**: Import statements, argparse boilerplate repeated in each file
- **Discoverability**: New pattern authors must learn the 4-stage structure
- **Rigid structure**: All patterns must follow exact stage naming

### 1.4 Current Strengths (To Preserve)

✅ **Independent testing**: Each stage script can run standalone
✅ **Resume from failure**: Can restart from any stage
✅ **Inspectable intermediates**: Pickle files between stages
✅ **Clear separation**: Classical vs quantum stages well-defined
✅ **Ray/Qiskit abstraction**: Executors handle distributed/quantum execution

---

## 2. Proposed Decorator Design

### 2.1 Core Decorator API

Create a new module `patterns/decorators.py` with stage decorators:

```python
from patterns.decorators import pattern_stage, PatternContext

@pattern_stage("map", agent_type="classical")
def map_stage(ctx: PatternContext) -> dict:
    """Create circuits and observables."""
    # Implementation
    return {"circuit": ..., "observables": ...}

@pattern_stage("optimize", agent_type="classical")
def optimize_stage(ctx: PatternContext) -> dict:
    """Transpile circuits."""
    circuit = ctx.load_input()  # Load from previous stage
    # Implementation
    return {"circuit": optimized_circuit}

@pattern_stage("execute", agent_type="quantum")
def execute_stage(ctx: PatternContext) -> dict:
    """Run quantum simulation."""
    # Implementation
    return {"results": ...}

@pattern_stage("post_process", agent_type="classical")
def post_process_stage(ctx: PatternContext) -> dict:
    """Analyze and visualize results."""
    # Implementation
    return {"plot_path": ..., "summary": ...}
```

### 2.2 Decorator Parameters

The `@pattern_stage` decorator accepts:

- **stage_name** (str, required): Stage identifier ("map", "optimize", "execute", "post_process")
- **agent_type** (str, required): "classical" or "quantum" (or stage-specific agent types - see below)
- **dependencies** (list[str], optional): Explicit stage dependencies (default: linear pipeline)
- **timeout** (int, optional): Execution timeout in seconds
- **retry** (int, optional): Number of retry attempts on failure

#### Alternative Agent Type Design

Instead of just two agent types (`classical` and `quantum`), the system could support **stage-specific agent types** for finer-grained specialization:

**Current approach (2 agent types)**:
```python
@pattern_stage("map", agent_type="classical")
@pattern_stage("optimize", agent_type="classical")
@pattern_stage("execute", agent_type="quantum")
@pattern_stage("post_process", agent_type="classical")
```

**Enhanced approach (4+ agent types)**:
```python
@pattern_stage("map", agent_type="map")
@pattern_stage("optimize", agent_type="optimize")
@pattern_stage("execute", agent_type="execute")
@pattern_stage("post_process", agent_type="post_process")
```

This would enable:
- **MapAgent**: Specialized for circuit creation, parameter space definition
- **OptimizeAgent**: Specialized for transpilation, circuit optimization, backend selection
- **ExecuteAgent**: Specialized for quantum execution, result collection
- **PostProcessAgent**: Specialized for analysis, visualization, metric computation

**Benefits of stage-specific agents**:
- ✅ Better separation of concerns (each agent has single responsibility)
- ✅ Easier to extend (add new optimization strategies to OptimizeAgent only)
- ✅ Custom resource allocation (e.g., OptimizeAgent gets more CPU, ExecuteAgent gets QPU access)
- ✅ Independent evolution (update ExecuteAgent without touching MapAgent)
- ✅ Better telemetry (track performance per agent type)

**Trade-offs**:
- ❌ More agent classes to maintain (4 instead of 2)
- ❌ Higher initial complexity
- ❌ Potential over-engineering for simple patterns

**Implementation path**:
- Phase 1: Start with 2 agent types (classical/quantum)
- Phase 2+: Allow custom agent type strings, with fallback mapping:
  - `agent_type="map"` → falls back to ClassicalAgent if MapAgent doesn't exist
  - `agent_type="optimize"` → falls back to ClassicalAgent
  - `agent_type="execute"` → falls back to QuantumAgent
  - `agent_type="post_process"` → falls back to ClassicalAgent

This preserves backward compatibility while enabling gradual migration to specialized agents.

### 2.3 PatternContext API

The context object provides:

```python
class PatternContext:
    """Context passed to each pattern stage function."""

    # Stage metadata
    stage_name: str
    pattern_name: str

    # Input/output management
    def load_input(self) -> dict:
        """Load output from previous stage."""

    def save_output(self, data: dict) -> Path:
        """Save stage output to pickle file."""

    def get_input_path(self) -> Path:
        """Get path to previous stage's output."""

    def get_output_path(self) -> Path:
        """Get path for this stage's output."""

    # Configuration access
    def get_config(self, key: str) -> Any:
        """Access pattern-specific config from settings.py."""

    # State access (read-only)
    state: PatternState  # Access to current workflow state

    # Logging
    def log(self, message: str, level: str = "INFO"):
        """Log message to orchestrator."""
```

### 2.4 Decorator Implementation Strategy

The decorator will:

1. **Register functions** in a global registry keyed by pattern name and stage
2. **Validate function signature** (must accept `ctx: PatternContext`)
3. **Store metadata** (stage name, agent type, dependencies)
4. **Wrap function** to handle I/O, timing, error handling automatically

---

## 3. Pattern Discovery Mechanism

### 3.1 Pattern File Structure

Decorated patterns will use this structure:

```
patterns/
├── chsh/
│   ├── __init__.py          # Import pattern module
│   ├── pattern.py           # NEW: Single file with decorated functions
│   ├── map.py               # OLD: Keep for backward compatibility (optional)
│   ├── optimize.py
│   ├── execute.py
│   └── post_process.py
```

Alternative (breaking change):
```
patterns/
├── chsh.py                  # Single file per pattern
├── vqe.py
└── legacy/                  # Move old multi-file patterns here
    └── chsh/
```

### 3.2 Discovery Algorithm

1. **Import pattern module**: `import patterns.chsh.pattern` or `import patterns.chsh`
2. **Scan module for decorated functions**: Check `__pattern_stages__` attribute
3. **Build stage registry**: Map stage names to callable functions
4. **Validate completeness**: Ensure all required stages are defined
5. **Validate dependencies**: Check dependency graph is acyclic

### 3.3 Registry Structure

```python
# Global pattern registry
_PATTERN_REGISTRY: Dict[str, Dict[str, StageMetadata]] = {
    "chsh": {
        "map": StageMetadata(
            func=map_stage,
            agent_type="classical",
            dependencies=[],
            timeout=None,
        ),
        "optimize": StageMetadata(
            func=optimize_stage,
            agent_type="classical",
            dependencies=["map"],
            timeout=None,
        ),
        # ...
    }
}
```

### 3.4 Pattern Loader Module

Create `patterns/loader.py`:

```python
class PatternLoader:
    """Discovers and loads decorated patterns."""

    def load_pattern(self, pattern_name: str) -> Dict[str, StageMetadata]:
        """Load all stages for a pattern."""

    def validate_pattern(self, pattern_name: str) -> bool:
        """Validate pattern has all required stages."""

    def get_stage_function(self, pattern_name: str, stage_name: str) -> Callable:
        """Get the callable for a specific stage."""
```

---

## 4. Execution Model Changes

### 4.1 Current Execution (Script-Based)

```
Orchestrator → Workflow Node → Agent → Executor → Subprocess(script.py)
```

### 4.2 Proposed Execution (Decorator-Based)

Two execution modes:

**Mode 1: Direct Function Invocation** (Simpler, recommended)
```
Orchestrator → Workflow Node → Agent → Function Call (in-process)
```

**Mode 2: Serialized Function Execution** (Preserves Ray distribution)
```
Orchestrator → Workflow Node → Agent → Ray Remote (serialize function) → Subprocess
```

### 4.3 Workflow Node Changes

Modify `workflows/pattern_graph.py`:

**Current**:
```python
def map_stage_node(state: PatternState) -> PatternState:
    script_path = PATTERNS_DIR / pattern_name / "map.py"
    result = agent.run_map_stage(script_path, output_path)
```

**Proposed**:
```python
def map_stage_node(state: PatternState) -> PatternState:
    # Try decorated function first, fall back to script
    pattern_loader = PatternLoader()

    if pattern_loader.has_decorated_stage(pattern_name, "map"):
        stage_func = pattern_loader.get_stage_function(pattern_name, "map")
        ctx = PatternContext(state, "map", pattern_name)
        result = agent.run_decorated_stage(stage_func, ctx)
    else:
        # Fall back to script-based execution
        script_path = PATTERNS_DIR / pattern_name / "map.py"
        result = agent.run_map_stage(script_path, output_path)
```

### 4.4 Agent Changes

Add methods to `ClassicalAgent` and `QuantumAgent`:

```python
class ClassicalAgent:
    def run_decorated_stage(
        self,
        stage_func: Callable[[PatternContext], dict],
        ctx: PatternContext
    ) -> dict:
        """Execute a decorated stage function."""
        # Timing
        # Error handling
        # Logging
        # Call function with context
        # Save output automatically
```

---

## 5. Data Flow Architecture

### 5.1 Current Data Flow (File-Based)

```
map.py → data/chsh_map_result.pkl → optimize.py → data/chsh_optimize_result.pkl → ...
```

### 5.2 Proposed Data Flow (Decorator-Based)

**Option A: Maintain File-Based I/O** (Recommended for Phase 1)
- Decorated functions still use `ctx.load_input()` and `ctx.save_output()`
- PatternContext handles pickle serialization internally
- Preserves ability to inspect intermediate results
- Supports resume-from-failure

**Option B: In-Memory State Passing** (Future optimization)
- Pass results directly via PatternState
- Only serialize to disk at checkpoints
- Faster for small patterns
- Requires LangGraph memory management

### 5.3 File Path Management

PatternContext will determine paths automatically:

```python
class PatternContext:
    def get_output_path(self) -> Path:
        """Auto-generate output path: data/{pattern}_{stage}_result.pkl"""
        return DATA_DIR / f"{self.pattern_name}_{self.stage_name}_result.pkl"
```

No need for pattern-specific config in `settings.py` if using convention.

---

## 6. Backward Compatibility Strategy

### 6.1 Dual-Mode Support

Support **both script-based and decorator-based** patterns:

1. Orchestrator checks if pattern has decorated stages
2. If decorated stages exist, use function invocation
3. If not, fall back to script-based execution
4. Allow **hybrid patterns** (some stages decorated, some scripts)

### 6.2 Migration Path

**Phase 1**: Add decorator support, keep existing scripts
- No breaking changes
- Patterns can opt-in to decorators
- Both modes tested in parallel

**Phase 2**: Migrate CHSH pattern to decorators
- Create `patterns/chsh/pattern.py`
- Keep old scripts for comparison
- Update tests to cover both

**Phase 3**: Deprecation (optional)
- Mark script-based approach as deprecated
- Add warnings when using scripts
- Provide migration guide

**Phase 4**: Remove scripts (optional, breaking)
- Delete deprecated script files
- Simplify orchestrator (remove fallback logic)

### 6.3 Configuration Migration

**Current**: Each pattern has hardcoded config in `settings.py`
```python
CHSH_CONFIG = {
    "map_output": DATA_DIR / "chsh_map_result.pkl",
    "optimize_output": DATA_DIR / "chsh_optimize_result.pkl",
    # ...
}
```

**Proposed**: Use convention-based paths, allow overrides
```python
# Default convention: data/{pattern}_{stage}_result.pkl
# Override only if needed
PATTERN_OVERRIDES = {
    "chsh": {
        "post_process_output": DATA_DIR / "custom_chsh_plot.png"
    }
}
```

---

## 7. Implementation Phases

### Phase 1: Core Decorator Infrastructure (Week 1)

**Goal**: Create decorator system without breaking existing code

**Tasks**:
1. Create `patterns/decorators.py` with `@pattern_stage` decorator
2. Implement `PatternContext` class with I/O helpers
3. Create `patterns/loader.py` for pattern discovery
4. Add global registry for decorated stages
5. Write unit tests for decorator registration

**Deliverables**:
- Working decorator that registers functions
- PatternContext with pickle I/O
- Pattern loader that can discover decorated functions
- Test suite (90%+ coverage)

### Phase 2: Orchestrator Integration (Week 2)

**Goal**: Enable orchestrator to execute decorated functions

**Tasks**:
1. Modify `workflows/pattern_graph.py` to support dual-mode execution
2. Add `run_decorated_stage()` to ClassicalAgent and QuantumAgent
3. Implement fallback logic (decorated → script)
4. Update `PatternState` to track execution mode
5. Add logging for decorator-based execution

**Deliverables**:
- Orchestrator can execute decorated stages
- Backward compatibility maintained
- Integration tests

### Phase 3: Example Pattern Migration (Week 3)

**Goal**: Migrate CHSH pattern to decorators as proof-of-concept

**Tasks**:
1. Create `patterns/chsh/pattern.py` with decorated stages
2. Port logic from map.py, optimize.py, execute.py, post_process.py
3. Add pattern-specific tests
4. Verify output matches original implementation
5. Document decorated pattern structure

**Deliverables**:
- `patterns/chsh/pattern.py` fully functional
- Old scripts still work (backward compatibility)
- Documentation in CLAUDE.md

### Phase 4: Advanced Features (Week 4)

**Goal**: Add advanced decorator features

**Tasks**:
1. Implement dependency specification (non-linear pipelines)
2. Add timeout and retry support
3. Create visualization of decorated stage graph
4. Add validation for decorator usage
5. Error handling improvements

**Deliverables**:
- Advanced decorator features working
- Better error messages
- Enhanced documentation

### Phase 5: Documentation & Migration Guide (Week 5)

**Goal**: Enable users to create new patterns with decorators

**Tasks**:
1. Write comprehensive decorator API documentation
2. Create migration guide for existing patterns
3. Add examples for common patterns (VQE, QAOA)
4. Update main.py CLI documentation
5. Add troubleshooting guide

**Deliverables**:
- Complete API reference
- Migration guide
- Pattern development tutorial
- CLI help updated

---

## 8. Trade-offs and Considerations

### 8.1 Advantages of Decorator Approach

✅ **Single-file patterns**: All stages in one place, easier to understand
✅ **Less boilerplate**: No argparse, no main(), automatic I/O handling
✅ **Better discoverability**: Decorators make structure explicit
✅ **Type safety**: PatternContext provides typed interface
✅ **Flexibility**: Can still use scripts for complex cases
✅ **Easier testing**: Import and call functions directly
✅ **Better IDE support**: Jump to definition, autocomplete

### 8.2 Disadvantages and Risks

❌ **Execution model complexity**: Two ways to run stages (scripts vs functions)
❌ **Serialization issues**: Functions harder to serialize than scripts for Ray
❌ **Import-time side effects**: Decorators run at import, could cause issues
❌ **Debugging differences**: Stack traces different for in-process vs subprocess
❌ **Migration effort**: Converting existing patterns takes time
❌ **Larger files**: Single file with 4 stages can be ~500 lines

### 8.3 Technical Challenges

**Challenge 1: Ray Distribution**
- **Problem**: Ray prefers script execution, functions need pickling
- **Solution**: Use cloudpickle for function serialization, or execute in-process for simple cases

**Challenge 2: Dependency Management**
- **Problem**: Decorators evaluated at import time, order matters
- **Solution**: Defer validation to discovery phase, not decoration phase

**Challenge 3: Error Handling**
- **Problem**: Exceptions in decorated functions vs subprocess errors differ
- **Solution**: Wrap decorated functions with try/except, normalize error format

**Challenge 4: Testing Isolation**
- **Problem**: Decorated functions share global registry
- **Solution**: Add `PatternRegistry.clear()` for test isolation

### 8.4 Performance Considerations

**In-Process Execution**:
- ✅ Faster: No subprocess overhead
- ✅ Better debugging: Full stack traces
- ❌ Memory: All stages in same process

**Subprocess Execution** (current):
- ✅ Isolation: Failures don't crash orchestrator
- ✅ Resource cleanup: Process exit frees memory
- ❌ Slower: Subprocess spawn overhead

**Recommendation**:
- Default to in-process for classical stages (faster)
- Use subprocess for quantum stages (isolation)
- Make it configurable per-stage

---

## 9. Example Usage

### 9.1 Complete Decorated Pattern Example

```python
# patterns/chsh/pattern.py
"""CHSH inequality test pattern using decorators."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
import matplotlib.pyplot as plt

from patterns.decorators import pattern_stage, PatternContext


@pattern_stage("map", agent_type="classical")
def map_stage(ctx: PatternContext) -> dict:
    """Create Bell circuit and CHSH observables."""
    ctx.log("Creating parameterized Bell circuit...")

    # Create parameter
    theta = Parameter('θ')

    # Create Bell circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(theta, 1)

    # Create observables
    obs1 = SparsePauliOp(["ZZ"], coeffs=[1.0])
    obs2 = SparsePauliOp(["XX"], coeffs=[1.0])

    # Phase sweep
    phases = np.linspace(0, 2 * np.pi, 16)

    ctx.log(f"Created circuit with {len(phases)} phase values")

    return {
        "circuit": qc,
        "observable1": obs1,
        "observable2": obs2,
        "phases": phases,
        "parameter": theta,
    }


@pattern_stage("optimize", agent_type="classical")
def optimize_stage(ctx: PatternContext) -> dict:
    """Transpile circuit for AerSimulator."""
    ctx.log("Loading map stage output...")
    map_data = ctx.load_input()

    circuit = map_data["circuit"]
    ctx.log(f"Original circuit depth: {circuit.depth()}")

    # Transpile
    backend = AerSimulator()
    optimized = transpile(circuit, backend=backend, optimization_level=1)

    ctx.log(f"Optimized circuit depth: {optimized.depth()}")

    return {
        "circuit": optimized,
        "observable1": map_data["observable1"],
        "observable2": map_data["observable2"],
        "phases": map_data["phases"],
        "parameter": map_data["parameter"],
        "original_depth": circuit.depth(),
        "optimized_depth": optimized.depth(),
    }


@pattern_stage("execute", agent_type="quantum")
def execute_stage(ctx: PatternContext) -> dict:
    """Execute circuits on quantum simulator."""
    ctx.log("Loading optimize stage output...")
    opt_data = ctx.load_input()

    circuit = opt_data["circuit"]
    obs1 = opt_data["observable1"]
    phases = opt_data["phases"]
    parameter = opt_data["parameter"]

    # Run estimator
    estimator = AerEstimator()

    ctx.log(f"Running {len(phases)} circuits...")

    # Bind parameters and run
    circuits = [circuit.assign_parameters({parameter: phase}) for phase in phases]
    observables = [obs1] * len(circuits)

    job = estimator.run(circuits, observables)
    result = job.result()

    expectation_values = result.values

    ctx.log(f"Execution complete. Got {len(expectation_values)} results")

    return {
        "expectation_values": expectation_values,
        "phases": phases,
    }


@pattern_stage("post_process", agent_type="classical")
def post_process_stage(ctx: PatternContext) -> dict:
    """Analyze results and create visualization."""
    ctx.log("Loading execute stage output...")
    exec_data = ctx.load_input()

    phases = exec_data["phases"]
    values = exec_data["expectation_values"]

    # Create plot
    ctx.log("Creating visualization...")
    plt.figure(figsize=(10, 6))
    plt.plot(phases, values, 'o-', label='Expectation Value')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Expectation Value')
    plt.title('CHSH Inequality Test Results')
    plt.grid(True)
    plt.legend()

    # Save plot
    plot_path = ctx.get_output_path().with_suffix('.png')
    plt.savefig(plot_path)
    plt.close()

    # Create summary
    max_val = max(values)
    min_val = min(values)
    chsh_value = max_val - min_val

    summary = {
        "max_expectation": float(max_val),
        "min_expectation": float(min_val),
        "chsh_value": float(chsh_value),
        "chsh_threshold": 2.0,
        "violates_classical_bound": chsh_value > 2.0,
    }

    ctx.log(f"CHSH value: {chsh_value:.3f} (classical bound: 2.0)")

    return {
        "plot_path": str(plot_path),
        "summary": summary,
    }
```

### 9.2 Usage Comparison

**Current (Script-Based)**:
```bash
# Must run each stage separately
python patterns/chsh/map.py --output data/map.pkl
python patterns/chsh/optimize.py --input data/map.pkl --output data/opt.pkl
python patterns/chsh/execute.py --input data/opt.pkl --output data/exec.pkl
python patterns/chsh/post_process.py --input data/exec.pkl --output plot.png

# Or run via orchestrator
python main.py --pattern chsh
```

**Proposed (Decorator-Based)**:
```bash
# Same orchestrator command
python main.py --pattern chsh

# But now can also import and test individual stages
python -c "
from patterns.chsh.pattern import map_stage
from patterns.decorators import PatternContext
ctx = PatternContext(state, 'map', 'chsh')
result = map_stage(ctx)
print(result)
"
```

### 9.3 Pattern Registration

The decorated pattern is automatically registered at import:

```python
# patterns/__init__.py
from patterns.chsh.pattern import *  # Auto-registers stages

# Or explicit registration
from patterns.loader import PatternLoader
loader = PatternLoader()
loader.register_pattern("chsh", "patterns.chsh.pattern")
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Test Decorator Registration**:
```python
def test_pattern_stage_decorator():
    @pattern_stage("test", agent_type="classical")
    def test_func(ctx: PatternContext) -> dict:
        return {"result": 42}

    assert test_func._pattern_metadata.stage_name == "test"
    assert test_func._pattern_metadata.agent_type == "classical"
```

**Test PatternContext**:
```python
def test_pattern_context_io(tmp_path):
    ctx = PatternContext(state, "test", "test_pattern")
    ctx.save_output({"value": 123})

    loaded = ctx.load_input()
    assert loaded["value"] == 123
```

**Test Pattern Loader**:
```python
def test_pattern_loader_discovery():
    loader = PatternLoader()
    stages = loader.load_pattern("chsh")

    assert "map" in stages
    assert "optimize" in stages
    assert "execute" in stages
    assert "post_process" in stages
```

### 10.2 Integration Tests

**Test Decorated Pattern Execution**:
```python
def test_decorated_chsh_execution():
    orchestrator = Orchestrator(pattern_name="chsh")
    result = orchestrator.run_pattern()

    assert result["status"] == "success"
    assert result["state"]["current_stage"] == "complete"
    assert all(s == "complete" for s in result["state"]["stage_status"].values())
```

**Test Backward Compatibility**:
```python
def test_script_based_execution_still_works():
    # Temporarily hide decorated pattern
    with mock_pattern_unavailable("chsh"):
        orchestrator = Orchestrator(pattern_name="chsh")
        result = orchestrator.run_pattern()

        assert result["status"] == "success"
```

### 10.3 Regression Tests

**Compare Script vs Decorator Outputs**:
```python
def test_decorator_output_matches_script():
    # Run script-based
    script_result = run_script_based_pattern("chsh")

    # Run decorator-based
    decorator_result = run_decorated_pattern("chsh")

    # Compare final outputs
    assert_outputs_equivalent(script_result, decorator_result)
```

### 10.4 Performance Tests

```python
def test_decorator_execution_performance():
    import time

    start = time.perf_counter()
    result = run_decorated_pattern("chsh")
    decorator_time = time.perf_counter() - start

    start = time.perf_counter()
    result = run_script_based_pattern("chsh")
    script_time = time.perf_counter() - start

    # Decorator-based should be faster (no subprocess overhead)
    assert decorator_time < script_time
```

---

## 11. Open Questions

### 11.1 Design Decisions Needed

1. **Agent type granularity**: Use 2 agent types (classical/quantum) or 4+ stage-specific agents (MapAgent, OptimizeAgent, ExecuteAgent, PostProcessAgent)? The latter provides better separation of concerns but adds complexity. See [Alternative Agent Type Design](#alternative-agent-type-design).

2. **File organization**: Single `pattern.py` vs `{pattern_name}.py` in patterns/ root? Should decorated patterns live alongside scripts or in a separate directory structure?

3. **Execution mode**: In-process by default, or always subprocess for consistency? In-process is faster but subprocess provides better isolation and resource cleanup.

4. **Config approach**: Convention-based paths (`data/{pattern}_{stage}_result.pkl`) or keep explicit config in `settings.py`? Convention reduces boilerplate but explicit config provides more control.

5. **Migration timeline**: When to deprecate script-based approach? Maintain indefinitely, or set a deprecation timeline once decorator approach is proven?

6. **Pattern validation**: Strict (must have all 4 stages) or flexible (allow patterns with subset of stages)? Flexibility enables simpler patterns but may complicate orchestration logic.

### 11.2 Future Enhancements

1. **Stage-specific agent types**: Instead of just `classical` and `quantum` agents, support specialized agents for each stage type (MapAgent, OptimizeAgent, ExecuteAgent, PostProcessAgent). This enables better separation of concerns, custom resource allocation per stage, and independent evolution of agent capabilities. See [Alternative Agent Type Design](#alternative-agent-type-design) for detailed proposal.

2. **Non-linear workflows**: Support branching/parallel stages beyond the current linear pipeline (map → optimize → execute → post_process). Enable DAG-based workflows where stages can run in parallel or have conditional paths.

3. **Conditional stages**: Skip stages based on runtime conditions (e.g., skip optimization if circuit is already optimal, skip certain measurements based on intermediate results).

4. **Stage composition**: Reuse stages across patterns. For example, multiple patterns could share the same optimization or post-processing logic while having different map/execute stages.

5. **Dynamic stage generation**: Generate stages programmatically based on configuration or runtime parameters. Useful for patterns that need variable numbers of optimization passes or adaptive workflows.

6. **Interactive development**: Live reload of decorated patterns during development. Watch pattern files and automatically re-register stages when code changes, enabling rapid iteration without restarting the orchestrator.

---

## 12. Success Criteria

### 12.1 Minimum Viable Product (MVP)

- ✅ Decorator system registers and discovers pattern stages
- ✅ Orchestrator can execute decorated patterns end-to-end
- ✅ Backward compatibility: Script-based patterns still work
- ✅ CHSH pattern migrated to decorators successfully
- ✅ Test coverage ≥ 85%
- ✅ Documentation complete

### 12.2 Stretch Goals

- 🎯 All patterns migrated to decorators
- 🎯 Advanced features (dependencies, retries, timeouts)
- 🎯 10x faster pattern development (measured by lines of boilerplate)
- 🎯 Performance improvement (in-process execution)
- 🎯 CLI enhancements (run individual decorated stages)

---

## 13. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Ray serialization fails for decorated functions | High | Medium | Use cloudpickle, fall back to subprocess |
| Performance regression | Medium | Low | Benchmark early, optimize if needed |
| Breaking changes force migration | High | Low | Maintain backward compatibility indefinitely |
| Complex patterns don't fit decorator model | Medium | Medium | Keep script-based option available |
| Import-time errors break system | High | Low | Lazy loading, robust error handling |

---

## 14. Conclusion

The decorator-based pattern system offers significant ergonomic improvements for pattern development while maintaining backward compatibility with the existing script-based approach. The phased implementation plan minimizes risk and allows for iterative refinement based on real-world usage.

**Recommended Next Steps**:

1. Review this plan with stakeholders
2. Prototype decorator infrastructure (Phase 1)
3. Validate with CHSH migration (Phase 3)
4. Gather feedback and iterate
5. Expand to remaining patterns

This approach balances innovation with stability, providing a clear path toward simpler pattern development without disrupting existing workflows.
