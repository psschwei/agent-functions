"""
Orchestrator Agent

Main supervisor that coordinates classical and quantum agents
through the pattern execution workflow using LangGraph.
"""
from typing import Dict, Any
import time

from workflows.pattern_graph import create_pattern_workflow, create_initial_state


class Orchestrator:
    """
    Orchestrator agent for managing pattern execution workflow.

    Uses LangGraph to coordinate classical and quantum agents
    through the stages of a Qiskit pattern.
    """

    def __init__(self, pattern_name: str = "chsh"):
        """
        Initialize the orchestrator.

        Args:
            pattern_name: Name of the pattern to execute
        """
        self.pattern_name = pattern_name
        self.workflow = None
        self.state = None

    def initialize_workflow(self):
        """Initialize the LangGraph workflow."""
        print("\n" + "=" * 60)
        print("ORCHESTRATOR: Initializing Workflow")
        print("=" * 60)
        print(f"Pattern: {self.pattern_name}")

        # Create workflow
        self.workflow = create_pattern_workflow()

        # Create initial state
        self.state = create_initial_state(self.pattern_name)

        print("✓ Workflow initialized")
        print(f"  Stages: {list(self.state['stage_status'].keys())}")

    def run_pattern(self) -> Dict[str, Any]:
        """
        Execute the complete pattern workflow.

        Returns:
            Final state after workflow execution
        """
        if self.workflow is None:
            self.initialize_workflow()

        print("\n" + "=" * 60)
        print("ORCHESTRATOR: Starting Pattern Execution")
        print("=" * 60)

        start_time = time.perf_counter()

        try:
            # Run the workflow
            final_state = None
            for state in self.workflow.stream(self.state):
                final_state = state

            end_time = time.perf_counter()
            total_duration = end_time - start_time

            print("\n" + "=" * 60)
            print("ORCHESTRATOR: Pattern Execution Complete")
            print("=" * 60)

            # Extract the actual state from the stream output
            # LangGraph stream returns dict with node names as keys
            if final_state:
                # Get the last node's state
                last_node_key = list(final_state.keys())[-1]
                actual_state = final_state[last_node_key]
            else:
                actual_state = self.state

            # Print summary
            print(f"\nExecution Summary:")
            print(f"  Total Duration: {total_duration:.2f}s")
            print(f"  Final Stage: {actual_state.get('current_stage', 'unknown')}")

            print(f"\nStage Status:")
            for stage, status in actual_state.get("stage_status", {}).items():
                duration = actual_state.get("stage_timings", {}).get(stage, 0)
                status_symbol = "✓" if status == "complete" else "✗" if status == "failed" else "○"
                print(f"  {status_symbol} {stage}: {status} ({duration:.2f}s)")

            if actual_state.get("errors"):
                print(f"\nErrors:")
                for error in actual_state["errors"]:
                    print(f"  - {error}")

            if actual_state.get("post_process_output"):
                print(f"\nOutputs:")
                print(f"  Plot: {actual_state['post_process_output']}")

            return {
                "status": "success" if not actual_state.get("errors") else "failed",
                "state": actual_state,
                "total_duration": total_duration,
            }

        except Exception as e:
            end_time = time.perf_counter()
            total_duration = end_time - start_time

            print("\n" + "=" * 60)
            print("ORCHESTRATOR: Pattern Execution Failed")
            print("=" * 60)
            print(f"Error: {str(e)}")

            return {
                "status": "failed",
                "error": str(e),
                "total_duration": total_duration,
            }

    def get_workflow_visualization(self) -> str:
        """
        Get a Mermaid diagram visualization of the workflow.

        Returns:
            Mermaid diagram string
        """
        if self.workflow is None:
            self.initialize_workflow()

        try:
            # Get the Mermaid diagram
            mermaid_diagram = self.workflow.get_graph().draw_mermaid()
            return mermaid_diagram
        except Exception as e:
            return f"Error generating visualization: {str(e)}"
