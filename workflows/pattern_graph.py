"""
LangGraph workflow definition for Qiskit pattern execution.

Defines the state schema and workflow graph for orchestrating
classical and quantum agents through pattern stages.
"""
from typing import TypedDict, Literal, Optional, Annotated
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agents import ClassicalAgent, QuantumAgent
from config import PATTERNS_DIR, CHSH_CONFIG
from patterns.loader import PatternLoader
from patterns.decorators import PatternContext


class PatternState(TypedDict):
    """
    State schema for pattern execution workflow.

    Tracks the current stage, file paths for intermediate results,
    execution metadata, and any errors that occur.
    """
    # Input configuration
    pattern_name: str  # e.g., "chsh"
    enable_llm: bool  # Enable LLM-powered decision-making in stages

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

    # Phase 3: Retry and iteration tracking
    stage_retry_count: dict[str, int]  # Number of retries per stage
    workflow_iteration: int  # Current workflow iteration number
    iteration_history: list[dict]  # History of previous iterations
    should_iterate: bool  # Whether to run another iteration

    # Agent messages/logs
    messages: Annotated[list, add_messages]


def map_stage_node(state: PatternState) -> PatternState:
    """
    Execute the map stage using ClassicalAgent.

    Creates the initial circuits and observables for the pattern.
    Supports both decorated and script-based patterns.
    """
    print("\n" + "=" * 60)
    print("MAP STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-Map")

    # Update state
    state["current_stage"] = "map"
    state["stage_status"]["map"] = "running"

    # Try decorated pattern first
    loader = PatternLoader()
    try:
        loader.load_pattern(pattern_name)
        if loader.has_decorated_stage(pattern_name, "map"):
            print(f"[Workflow] Using decorated map stage for pattern '{pattern_name}'")

            # Execute decorated stage
            ctx = PatternContext(state, "map", pattern_name, enable_llm=state.get("enable_llm", False))
            stage_func = loader.get_stage_function(pattern_name, "map")
            result = agent.run_decorated_stage(stage_func, ctx)
        else:
            raise ValueError("No decorated stage found")
    except (ImportError, ValueError, KeyError):
        # Fall back to script-based execution
        print(f"[Workflow] Using script-based map stage for pattern '{pattern_name}'")

        script_path = PATTERNS_DIR / pattern_name / "map.py"
        output_path = Path(CHSH_CONFIG["map_output"])
        result = agent.run_map_stage(script_path, output_path)

    # Update state with results
    if result["status"] == "success":
        state["map_output"] = result["output_path"]
        state["stage_status"]["map"] = "complete"
        state["stage_timings"]["map"] = result["duration"]
    else:
        state["stage_status"]["map"] = "failed"
        state["errors"].append(f"Map stage failed: {result.get('error', 'Unknown error')}")

    return state


def optimize_stage_node(state: PatternState) -> PatternState:
    """
    Execute the optimize stage using ClassicalAgent.

    Transpiles/optimizes circuits for the target backend.
    Supports both decorated and script-based patterns.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZE STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-Optimize")

    # Update state
    state["current_stage"] = "optimize"
    state["stage_status"]["optimize"] = "running"

    # Try decorated pattern first
    loader = PatternLoader()
    try:
        loader.load_pattern(pattern_name)
        if loader.has_decorated_stage(pattern_name, "optimize"):
            print(f"[Workflow] Using decorated optimize stage for pattern '{pattern_name}'")

            # Execute decorated stage
            ctx = PatternContext(state, "optimize", pattern_name, enable_llm=state.get("enable_llm", False))
            stage_func = loader.get_stage_function(pattern_name, "optimize")
            result = agent.run_decorated_stage(stage_func, ctx)
        else:
            raise ValueError("No decorated stage found")
    except (ImportError, ValueError, KeyError):
        # Fall back to script-based execution
        print(f"[Workflow] Using script-based optimize stage for pattern '{pattern_name}'")

        script_path = PATTERNS_DIR / pattern_name / "optimize.py"
        input_path = Path(state["map_output"])
        output_path = Path(CHSH_CONFIG["optimize_output"])
        result = agent.run_optimize_stage(script_path, input_path, output_path)

    # Update state with results
    if result["status"] == "success":
        state["optimize_output"] = result["output_path"]
        state["stage_status"]["optimize"] = "complete"
        state["stage_timings"]["optimize"] = result["duration"]
    else:
        state["stage_status"]["optimize"] = "failed"
        state["errors"].append(f"Optimize stage failed: {result.get('error', 'Unknown error')}")

    return state


def execute_stage_node(state: PatternState) -> PatternState:
    """
    Execute the quantum execution stage using QuantumAgent.

    Runs the circuits on the quantum simulator.
    Supports both decorated and script-based patterns.
    """
    print("\n" + "=" * 60)
    print("EXECUTE STAGE (Quantum)")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = QuantumAgent(name="QuantumAgent-Execute")

    # Update state
    state["current_stage"] = "execute"
    state["stage_status"]["execute"] = "running"

    # Try decorated pattern first
    loader = PatternLoader()
    try:
        loader.load_pattern(pattern_name)
        if loader.has_decorated_stage(pattern_name, "execute"):
            print(f"[Workflow] Using decorated execute stage for pattern '{pattern_name}'")

            # Execute decorated stage
            ctx = PatternContext(state, "execute", pattern_name, enable_llm=state.get("enable_llm", False))
            stage_func = loader.get_stage_function(pattern_name, "execute")
            result = agent.run_decorated_stage(stage_func, ctx)
        else:
            raise ValueError("No decorated stage found")
    except (ImportError, ValueError, KeyError):
        # Fall back to script-based execution
        print(f"[Workflow] Using script-based execute stage for pattern '{pattern_name}'")

        script_path = PATTERNS_DIR / pattern_name / "execute.py"
        input_path = Path(state["optimize_output"])
        output_path = Path(CHSH_CONFIG["execute_output"])
        result = agent.run_execute_stage(script_path, input_path, output_path)

    # Update state with results
    if result["status"] == "success":
        state["execute_output"] = result["output_path"]
        state["stage_status"]["execute"] = "complete"
        state["stage_timings"]["execute"] = result["duration"]
    else:
        state["stage_status"]["execute"] = "failed"
        state["errors"].append(f"Execute stage failed: {result.get('error', 'Unknown error')}")

    return state


def post_process_stage_node(state: PatternState) -> PatternState:
    """
    Execute the post-process stage using ClassicalAgent.

    Analyzes results and creates visualizations.
    Supports both decorated and script-based patterns.
    """
    print("\n" + "=" * 60)
    print("POST-PROCESS STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-PostProcess")

    # Update state
    state["current_stage"] = "post_process"
    state["stage_status"]["post_process"] = "running"

    # Try decorated pattern first
    loader = PatternLoader()
    try:
        loader.load_pattern(pattern_name)
        if loader.has_decorated_stage(pattern_name, "post_process"):
            print(f"[Workflow] Using decorated post_process stage for pattern '{pattern_name}'")

            # Execute decorated stage
            ctx = PatternContext(state, "post_process", pattern_name, enable_llm=state.get("enable_llm", False))
            stage_func = loader.get_stage_function(pattern_name, "post_process")
            result = agent.run_decorated_stage(stage_func, ctx)
        else:
            raise ValueError("No decorated stage found")
    except (ImportError, ValueError, KeyError):
        # Fall back to script-based execution
        print(f"[Workflow] Using script-based post_process stage for pattern '{pattern_name}'")

        script_path = PATTERNS_DIR / pattern_name / "post_process.py"
        input_path = Path(state["execute_output"])
        output_path = Path(CHSH_CONFIG["post_process_output"])
        summary_path = Path(CHSH_CONFIG["post_process_summary"])
        result = agent.run_post_process_stage(
            script_path, input_path, output_path, summary_path
        )

    # Update state with results
    if result["status"] == "success":
        state["post_process_output"] = result["output_path"]
        state["stage_status"]["post_process"] = "complete"
        state["stage_timings"]["post_process"] = result["duration"]
        state["current_stage"] = "complete"
    else:
        state["stage_status"]["post_process"] = "failed"
        state["errors"].append(f"Post-process stage failed: {result.get('error', 'Unknown error')}")

    return state


def decision_before_optimize(state: PatternState) -> PatternState:
    """
    Decision node: Should we proceed to optimize stage?

    Orchestrator analyzes map results and decides optimization strategy.
    """
    print("\n" + "=" * 60)
    print("DECISION: Analyzing Map Results")
    print("=" * 60)

    # Import here to avoid circular dependency
    from agents.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(pattern_name=state["pattern_name"])

    # Reason about optimize stage parameters
    decision = orchestrator.reason_after_stage(
        stage_name="map",
        state=state,
        result={"status": "success", "output_path": state["map_output"]},
    )

    # If LLM recommends retry, mark in state (though Phase 1 doesn't implement retry yet)
    if decision.get("should_retry"):
        state["errors"].append(f"Map stage evaluation suggests retry: {decision.get('reasoning')}")

    # Get parameters for optimize stage
    params = orchestrator.reason_before_stage("optimize", state)

    # Store recommended parameters in state for optimize node to use
    if "_llm_recommended_params" not in state:
        state["_llm_recommended_params"] = {}
    state["_llm_recommended_params"] = params.get("parameters", {})
    state["messages"].append({
        "role": "assistant",
        "content": f"Optimize stage parameters: {params.get('reasoning')}",
    })

    return state


def decision_before_execute(state: PatternState) -> PatternState:
    """Decision node before execute stage."""
    print("\n" + "=" * 60)
    print("DECISION: Analyzing Optimize Results")
    print("=" * 60)

    from agents.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(pattern_name=state["pattern_name"])

    # Evaluate optimize results
    decision = orchestrator.reason_after_stage(
        stage_name="optimize",
        state=state,
        result={"status": "success", "output_path": state["optimize_output"]},
    )

    # Get execute parameters
    params = orchestrator.reason_before_stage("execute", state)
    if "_llm_recommended_params" not in state:
        state["_llm_recommended_params"] = {}
    state["_llm_recommended_params"] = params.get("parameters", {})
    state["messages"].append({
        "role": "assistant",
        "content": f"Execute stage parameters: {params.get('reasoning')}",
    })

    return state


def decision_before_post_process(state: PatternState) -> PatternState:
    """Decision node before post-process stage."""
    print("\n" + "=" * 60)
    print("DECISION: Analyzing Execute Results")
    print("=" * 60)

    from agents.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(pattern_name=state["pattern_name"])

    # Evaluate execute results
    decision = orchestrator.reason_after_stage(
        stage_name="execute",
        state=state,
        result={"status": "success", "output_path": state["execute_output"]},
    )

    # Get post-process parameters
    params = orchestrator.reason_before_stage("post_process", state)
    if "_llm_recommended_params" not in state:
        state["_llm_recommended_params"] = {}
    state["_llm_recommended_params"] = params.get("parameters", {})
    state["messages"].append({
        "role": "assistant",
        "content": f"Post-process stage parameters: {params.get('reasoning')}",
    })

    return state


def post_workflow_reflection(state: PatternState) -> PatternState:
    """
    Reflection node: Analyze complete workflow and decide whether to iterate.

    Orchestrator reflects on the entire workflow execution and determines if
    another iteration with different parameters would improve results.
    """
    print("\n" + "=" * 60)
    print("REFLECTION: Analyzing Complete Workflow")
    print("=" * 60)

    from config import ORCHESTRATOR_CONFIG

    # Check if iteration is enabled
    if not ORCHESTRATOR_CONFIG.get("enable_iteration", True):
        print("  Iteration disabled in config")
        state["should_iterate"] = False
        return state

    # Check if we've reached max iterations
    max_iterations = ORCHESTRATOR_CONFIG.get("max_workflow_iterations", 3)
    current_iteration = state.get("workflow_iteration", 1)

    if current_iteration >= max_iterations:
        print(f"  Max iterations reached ({current_iteration}/{max_iterations})")
        state["should_iterate"] = False
        return state

    # Import here to avoid circular dependency
    from agents.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(pattern_name=state["pattern_name"])

    # Analyze the complete workflow
    reflection = orchestrator.reflect_on_workflow(state)

    # Store iteration results in history
    iteration_summary = {
        "iteration": current_iteration,
        "stage_timings": state["stage_timings"].copy(),
        "errors": state["errors"].copy(),
        "reflection": reflection,
    }
    state["iteration_history"].append(iteration_summary)

    # Decide whether to iterate
    should_iterate = reflection.get("should_iterate", False)
    state["should_iterate"] = should_iterate

    if should_iterate:
        print(f"  🔁 Iteration recommended: {reflection.get('reasoning', 'No reason provided')}")
        print(f"  Suggested changes: {reflection.get('recommended_changes', {})}")
    else:
        print(f"  ✓ Workflow complete - no iteration needed")
        print(f"  Reasoning: {reflection.get('reasoning', 'Results are satisfactory')}")

    return state


def reset_for_iteration(state: PatternState) -> PatternState:
    """
    Reset state for another workflow iteration.

    Increments iteration counter, resets stage status, and applies
    recommended parameter changes from reflection.
    """
    print("\n" + "=" * 60)
    print(f"ITERATION: Starting Iteration #{state['workflow_iteration'] + 1}")
    print("=" * 60)

    # Increment iteration counter
    state["workflow_iteration"] = state["workflow_iteration"] + 1

    # Reset stage status
    state["stage_status"] = {
        "map": "pending",
        "optimize": "pending",
        "execute": "pending",
        "post_process": "pending",
    }

    # Reset retry counts
    state["stage_retry_count"] = {
        "map": 0,
        "optimize": 0,
        "execute": 0,
        "post_process": 0,
    }

    # Clear previous errors
    state["errors"] = []

    # Reset to map stage
    state["current_stage"] = "map"

    print(f"  Iteration #{state['workflow_iteration']} initialized")
    print(f"  Previous iterations: {len(state['iteration_history'])}")

    return state


def route_after_reflection(state: PatternState) -> str:
    """
    Route workflow after reflection: iterate or end.

    Returns:
        "iterate" if should_iterate is True, "end" otherwise
    """
    if state.get("should_iterate", False):
        return "iterate"
    return "end"


def create_pattern_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for pattern execution.

    Now includes decision nodes between stages for LLM reasoning,
    post-workflow reflection, and iteration loops (Phase 3).

    Returns:
        Compiled StateGraph workflow
    """
    # Create workflow
    workflow = StateGraph(PatternState)

    # Add nodes for each stage
    workflow.add_node("map_stage", map_stage_node)
    workflow.add_node("decision_before_optimize", decision_before_optimize)
    workflow.add_node("optimize_stage", optimize_stage_node)
    workflow.add_node("decision_before_execute", decision_before_execute)
    workflow.add_node("execute_stage", execute_stage_node)
    workflow.add_node("decision_before_post_process", decision_before_post_process)
    workflow.add_node("post_process_stage", post_process_stage_node)

    # Phase 3: Add reflection and iteration nodes
    workflow.add_node("post_workflow_reflection", post_workflow_reflection)
    workflow.add_node("reset_for_iteration", reset_for_iteration)

    # Define workflow edges with decision nodes
    workflow.add_edge(START, "map_stage")
    workflow.add_edge("map_stage", "decision_before_optimize")
    workflow.add_edge("decision_before_optimize", "optimize_stage")
    workflow.add_edge("optimize_stage", "decision_before_execute")
    workflow.add_edge("decision_before_execute", "execute_stage")
    workflow.add_edge("execute_stage", "decision_before_post_process")
    workflow.add_edge("decision_before_post_process", "post_process_stage")

    # Phase 3: After post_process, go to reflection instead of END
    workflow.add_edge("post_process_stage", "post_workflow_reflection")

    # Phase 3: Conditional edge after reflection - iterate or end
    workflow.add_conditional_edges(
        "post_workflow_reflection",
        route_after_reflection,
        {
            "iterate": "reset_for_iteration",
            "end": END,
        }
    )

    # Phase 3: After reset, loop back to map_stage
    workflow.add_edge("reset_for_iteration", "map_stage")

    # Compile the workflow
    return workflow.compile()


def create_initial_state(pattern_name: str = "chsh", enable_llm: bool = False) -> PatternState:
    """
    Create the initial state for pattern execution.

    Args:
        pattern_name: Name of the pattern to execute
        enable_llm: Enable LLM-powered decision-making in stages

    Returns:
        Initial PatternState
    """
    return PatternState(
        pattern_name=pattern_name,
        enable_llm=enable_llm,
        current_stage="map",
        map_output=None,
        optimize_output=None,
        execute_output=None,
        post_process_output=None,
        errors=[],
        stage_status={
            "map": "pending",
            "optimize": "pending",
            "execute": "pending",
            "post_process": "pending",
        },
        stage_timings={},
        # Phase 3: Initialize retry and iteration tracking
        stage_retry_count={
            "map": 0,
            "optimize": 0,
            "execute": 0,
            "post_process": 0,
        },
        workflow_iteration=1,
        iteration_history=[],
        should_iterate=False,
        messages=[],
    )
