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


class PatternState(TypedDict):
    """
    State schema for pattern execution workflow.

    Tracks the current stage, file paths for intermediate results,
    execution metadata, and any errors that occur.
    """
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
    messages: Annotated[list, add_messages]


def map_stage_node(state: PatternState) -> PatternState:
    """
    Execute the map stage using ClassicalAgent.

    Creates the initial circuits and observables for the pattern.
    """
    print("\n" + "=" * 60)
    print("MAP STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-Map")

    # Get script path
    script_path = PATTERNS_DIR / pattern_name / "map.py"
    output_path = Path(CHSH_CONFIG["map_output"])

    # Update state
    state["current_stage"] = "map"
    state["stage_status"]["map"] = "running"

    # Execute
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
    """
    print("\n" + "=" * 60)
    print("OPTIMIZE STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-Optimize")

    # Get script path
    script_path = PATTERNS_DIR / pattern_name / "optimize.py"
    input_path = Path(state["map_output"])
    output_path = Path(CHSH_CONFIG["optimize_output"])

    # Update state
    state["current_stage"] = "optimize"
    state["stage_status"]["optimize"] = "running"

    # Execute
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
    """
    print("\n" + "=" * 60)
    print("EXECUTE STAGE (Quantum)")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = QuantumAgent(name="QuantumAgent-Execute")

    # Get script path
    script_path = PATTERNS_DIR / pattern_name / "execute.py"
    input_path = Path(state["optimize_output"])
    output_path = Path(CHSH_CONFIG["execute_output"])

    # Update state
    state["current_stage"] = "execute"
    state["stage_status"]["execute"] = "running"

    # Execute
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
    """
    print("\n" + "=" * 60)
    print("POST-PROCESS STAGE")
    print("=" * 60)

    pattern_name = state["pattern_name"]
    agent = ClassicalAgent(name="ClassicalAgent-PostProcess")

    # Get script path
    script_path = PATTERNS_DIR / pattern_name / "post_process.py"
    input_path = Path(state["execute_output"])
    output_path = Path(CHSH_CONFIG["post_process_output"])
    summary_path = Path(CHSH_CONFIG["post_process_summary"])

    # Update state
    state["current_stage"] = "post_process"
    state["stage_status"]["post_process"] = "running"

    # Execute
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


def create_pattern_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for pattern execution.

    Returns:
        Compiled StateGraph workflow
    """
    # Create workflow
    workflow = StateGraph(PatternState)

    # Add nodes for each stage
    workflow.add_node("map_stage", map_stage_node)
    workflow.add_node("optimize_stage", optimize_stage_node)
    workflow.add_node("execute_stage", execute_stage_node)
    workflow.add_node("post_process_stage", post_process_stage_node)

    # Define workflow edges (linear pipeline)
    workflow.add_edge(START, "map_stage")
    workflow.add_edge("map_stage", "optimize_stage")
    workflow.add_edge("optimize_stage", "execute_stage")
    workflow.add_edge("execute_stage", "post_process_stage")
    workflow.add_edge("post_process_stage", END)

    # Compile the workflow
    return workflow.compile()


def create_initial_state(pattern_name: str = "chsh") -> PatternState:
    """
    Create the initial state for pattern execution.

    Args:
        pattern_name: Name of the pattern to execute

    Returns:
        Initial PatternState
    """
    return PatternState(
        pattern_name=pattern_name,
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
        messages=[],
    )
