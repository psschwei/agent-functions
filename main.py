#!/usr/bin/env python3
"""
Agent Functions - Main Entry Point

Multi-agent system for orchestrating quantum and classical workloads
using LangGraph and Qiskit.
"""
import argparse
import sys
from pathlib import Path

from agents import Orchestrator, AgenticOrchestrator
from executors import init_ray_cluster, shutdown_ray_cluster
from utils import print_banner, print_section, format_duration


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute Qiskit patterns using multi-agent orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the CHSH pattern
  python main.py --pattern chsh

  # Run with Mellea-enhanced adaptive execution
  python main.py --pattern chsh --mellea

  # Run with agentic orchestrator (LLM reasoning)
  python main.py --pattern chsh --agentic

  # Combine Mellea with agentic orchestrator
  python main.py --pattern chsh --mellea --agentic

  # Run with workflow visualization
  python main.py --pattern chsh --visualize

  # Save workflow diagram
  python main.py --pattern chsh --save-diagram workflow.md
        """
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="chsh",
        choices=["chsh"],
        help="Name of the Qiskit pattern to execute (default: chsh)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display workflow visualization"
    )

    parser.add_argument(
        "--save-diagram",
        type=str,
        metavar="PATH",
        help="Save workflow diagram to file (Mermaid format)"
    )

    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="Skip Ray cluster initialization (for debugging)"
    )

    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use agentic orchestrator with LLM reasoning (default: basic orchestrator)"
    )

    parser.add_argument(
        "--mellea",
        action="store_true",
        help="Use Mellea-enhanced classical agent with adaptive execution (requires mellea package)"
    )

    return parser.parse_args()


def display_welcome():
    """Display welcome banner."""
    print_banner("Agent Functions - Quantum Pattern Orchestration", width=70)
    print("\nA multi-agent system using LangGraph to orchestrate")
    print("classical and quantum workloads for Qiskit patterns.")
    print()


def display_results(result: dict):
    """
    Display execution results.

    Args:
        result: Result dictionary from orchestrator
    """
    print_section("Execution Results")

    status = result.get("status", "unknown")
    total_duration = result.get("total_duration", 0)

    if status == "success":
        state = result.get("state", {})

        print(f"\n  Status: SUCCESS")
        print(f"  Total Duration: {format_duration(total_duration)}")

        # Display stage timings
        if state.get("stage_timings"):
            print(f"\n  Stage Breakdown:")
            for stage, duration in state["stage_timings"].items():
                print(f"    - {stage}: {format_duration(duration)}")

        # Display outputs
        if state.get("post_process_output"):
            print(f"\n  Output Files:")
            print(f"    - Plot: {state['post_process_output']}")

            # Check for summary file
            summary_path = Path(state['post_process_output']).parent / "chsh_summary.json"
            if summary_path.exists():
                print(f"    - Summary: {summary_path}")

    else:
        print(f"\n  Status: FAILED")
        print(f"  Total Duration: {format_duration(total_duration)}")

        if "error" in result:
            print(f"  Error: {result['error']}")


def save_workflow_diagram(orchestrator: Orchestrator, output_path: str):
    """
    Save workflow diagram to file.

    Args:
        orchestrator: Orchestrator instance
        output_path: Path to save the diagram
    """
    try:
        diagram = orchestrator.get_workflow_visualization()

        with open(output_path, 'w') as f:
            f.write("# Workflow Diagram\n\n")
            f.write("```mermaid\n")
            f.write(diagram)
            f.write("\n```\n")

        print(f"\n  Workflow diagram saved to: {output_path}")

    except Exception as e:
        print(f"\n  Error saving diagram: {str(e)}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Display welcome
    display_welcome()

    # Initialize Ray cluster
    if not args.no_ray:
        print_section("Initializing Ray Cluster")
        try:
            init_ray_cluster()
            print("  Ray cluster initialized successfully")
        except Exception as e:
            print(f"  Error initializing Ray: {str(e)}")
            print("  Continuing without Ray...")

    try:
        # Enable Mellea if flag is set
        if args.mellea:
            from config import MELLEA_CONFIG
            MELLEA_CONFIG["enabled"] = True
            print_section("Mellea Configuration")
            print(f"  Mellea enabled for stages: {MELLEA_CONFIG.get('stages', [])}")
            print(f"  Backend: {MELLEA_CONFIG.get('model_backend', 'ollama')}")
            print(f"  Max retries: {MELLEA_CONFIG.get('max_retries', 2)}")
        
        # Create orchestrator
        orchestrator_type = "Agentic" if args.agentic else "Basic"
        print_section(f"Creating {orchestrator_type} Orchestrator for '{args.pattern}' Pattern")

        if args.agentic:
            orchestrator = AgenticOrchestrator(pattern_name=args.pattern)
        else:
            orchestrator = Orchestrator(pattern_name=args.pattern)

        print(f"  {orchestrator_type} Orchestrator created")

        # Display workflow visualization if requested
        if args.visualize:
            print_section("Workflow Visualization")
            try:
                diagram = orchestrator.get_workflow_visualization()
                print("\n```mermaid")
                print(diagram)
                print("```\n")
            except Exception as e:
                print(f"  Error generating visualization: {str(e)}")

        # Save workflow diagram if requested
        if args.save_diagram:
            print_section("Saving Workflow Diagram")
            save_workflow_diagram(orchestrator, args.save_diagram)

        # Run the pattern
        print_section("Executing Pattern Workflow")
        result = orchestrator.run_pattern()

        # Display results
        display_results(result)

        # Exit with appropriate code
        exit_code = 0 if result.get("status") == "success" else 1

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        exit_code = 130

    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        # Shutdown Ray cluster
        if not args.no_ray:
            print_section("Shutting Down Ray Cluster")
            try:
                shutdown_ray_cluster()
                print("  Ray cluster shutdown complete")
            except Exception as e:
                print(f"  Error shutting down Ray: {str(e)}")

        print("\n" + "=" * 70)
        print("Execution Complete")
        print("=" * 70 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
