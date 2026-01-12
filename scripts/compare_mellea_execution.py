#!/usr/bin/env python3
"""
Comparison script for Mellea vs Standard execution.

This script runs a pattern multiple times with and without Mellea,
comparing execution time, success rate, and retry behavior.
"""
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.pattern_graph import create_pattern_workflow, create_initial_state
from config import MELLEA_CONFIG, DATA_DIR


def run_pattern_execution(
    pattern_name: str,
    use_mellea: bool,
    run_number: int
) -> Dict[str, Any]:
    """
    Run a single pattern execution.

    Args:
        pattern_name: Name of the pattern to execute
        use_mellea: Whether to use Mellea-enhanced agents
        run_number: Run number for logging

    Returns:
        Dictionary with execution metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Run #{run_number} - {'Mellea' if use_mellea else 'Standard'} Execution")
    print(f"{'=' * 60}")

    # Temporarily modify config
    original_enabled = MELLEA_CONFIG.get("enabled", False)
    MELLEA_CONFIG["enabled"] = use_mellea

    try:
        # Create workflow and initial state
        workflow = create_pattern_workflow()
        initial_state = create_initial_state(pattern_name=pattern_name, enable_llm=False)

        # Execute workflow
        start_time = time.perf_counter()
        
        final_state = None
        for state in workflow.stream(initial_state):
            final_state = state
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time

        # Extract metrics
        stage_timings = final_state.get("stage_timings", {})
        errors = final_state.get("errors", [])
        stage_status = final_state.get("stage_status", {})
        
        # Check if all stages completed successfully
        all_success = all(
            status == "complete" 
            for status in stage_status.values()
        )

        # Count retries (if available in state)
        retry_count = sum(final_state.get("stage_retry_count", {}).values())

        result = {
            "run_number": run_number,
            "use_mellea": use_mellea,
            "success": all_success,
            "total_duration": total_duration,
            "stage_timings": stage_timings,
            "stage_status": stage_status,
            "errors": errors,
            "retry_count": retry_count,
        }

        print(f"\n{'✓' if all_success else '✗'} Execution {'succeeded' if all_success else 'failed'}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Stage timings: {json.dumps(stage_timings, indent=2)}")
        if retry_count > 0:
            print(f"  Retries: {retry_count}")
        if errors:
            print(f"  Errors: {errors}")

        return result

    except Exception as e:
        print(f"\n✗ Exception during execution: {e}")
        return {
            "run_number": run_number,
            "use_mellea": use_mellea,
            "success": False,
            "total_duration": 0,
            "stage_timings": {},
            "stage_status": {},
            "errors": [str(e)],
            "retry_count": 0,
        }

    finally:
        # Restore original config
        MELLEA_CONFIG["enabled"] = original_enabled


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from multiple runs.

    Args:
        results: List of execution results

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {}

    successful_runs = [r for r in results if r["success"]]
    failed_runs = [r for r in results if not r["success"]]

    stats = {
        "total_runs": len(results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs) / len(results) if results else 0,
    }

    if successful_runs:
        durations = [r["total_duration"] for r in successful_runs]
        stats["avg_duration"] = sum(durations) / len(durations)
        stats["min_duration"] = min(durations)
        stats["max_duration"] = max(durations)

        # Average stage timings
        stage_timings = {}
        for result in successful_runs:
            for stage, timing in result["stage_timings"].items():
                if stage not in stage_timings:
                    stage_timings[stage] = []
                stage_timings[stage].append(timing)

        stats["avg_stage_timings"] = {
            stage: sum(timings) / len(timings)
            for stage, timings in stage_timings.items()
        }

        # Total retries
        total_retries = sum(r["retry_count"] for r in successful_runs)
        stats["total_retries"] = total_retries
        stats["avg_retries"] = total_retries / len(successful_runs)

    return stats


def print_comparison(standard_stats: Dict[str, Any], mellea_stats: Dict[str, Any]):
    """
    Print comparison between standard and Mellea execution.

    Args:
        standard_stats: Statistics from standard execution
        mellea_stats: Statistics from Mellea execution
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\n📊 Success Rate:")
    print(f"  Standard: {standard_stats.get('success_rate', 0):.1%} "
          f"({standard_stats.get('successful_runs', 0)}/{standard_stats.get('total_runs', 0)})")
    print(f"  Mellea:   {mellea_stats.get('success_rate', 0):.1%} "
          f"({mellea_stats.get('successful_runs', 0)}/{mellea_stats.get('total_runs', 0)})")

    if standard_stats.get("successful_runs", 0) > 0 and mellea_stats.get("successful_runs", 0) > 0:
        print("\n⏱️  Execution Time:")
        std_avg = standard_stats.get("avg_duration", 0)
        mel_avg = mellea_stats.get("avg_duration", 0)
        print(f"  Standard: {std_avg:.2f}s (min: {standard_stats.get('min_duration', 0):.2f}s, "
              f"max: {standard_stats.get('max_duration', 0):.2f}s)")
        print(f"  Mellea:   {mel_avg:.2f}s (min: {mellea_stats.get('min_duration', 0):.2f}s, "
              f"max: {mellea_stats.get('max_duration', 0):.2f}s)")
        
        if std_avg > 0:
            overhead = ((mel_avg - std_avg) / std_avg) * 100
            print(f"  Overhead: {overhead:+.1f}%")

        print("\n📈 Stage Timings (Average):")
        std_stages = standard_stats.get("avg_stage_timings", {})
        mel_stages = mellea_stats.get("avg_stage_timings", {})
        
        for stage in sorted(set(std_stages.keys()) | set(mel_stages.keys())):
            std_time = std_stages.get(stage, 0)
            mel_time = mel_stages.get(stage, 0)
            print(f"  {stage:12s}: Standard {std_time:.2f}s | Mellea {mel_time:.2f}s")

        print("\n🔄 Retries:")
        print(f"  Standard: {standard_stats.get('total_retries', 0)} total "
              f"({standard_stats.get('avg_retries', 0):.1f} avg)")
        print(f"  Mellea:   {mellea_stats.get('total_retries', 0)} total "
              f"({mellea_stats.get('avg_retries', 0):.1f} avg)")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Mellea vs Standard execution performance"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="chsh",
        help="Pattern name to execute (default: chsh)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "mellea_comparison.json",
        help="Output file for detailed results (default: data/mellea_comparison.json)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MELLEA EXECUTION COMPARISON")
    print("=" * 60)
    print(f"Pattern: {args.pattern}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Output file: {args.output}")

    # Run standard executions
    print("\n" + "=" * 60)
    print("STANDARD EXECUTION")
    print("=" * 60)
    standard_results = []
    for i in range(1, args.runs + 1):
        result = run_pattern_execution(args.pattern, use_mellea=False, run_number=i)
        standard_results.append(result)

    # Run Mellea executions
    print("\n" + "=" * 60)
    print("MELLEA EXECUTION")
    print("=" * 60)
    mellea_results = []
    for i in range(1, args.runs + 1):
        result = run_pattern_execution(args.pattern, use_mellea=True, run_number=i)
        mellea_results.append(result)

    # Calculate statistics
    standard_stats = calculate_statistics(standard_results)
    mellea_stats = calculate_statistics(mellea_results)

    # Print comparison
    print_comparison(standard_stats, mellea_stats)

    # Save detailed results
    output_data = {
        "pattern": args.pattern,
        "runs_per_config": args.runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "standard": {
            "results": standard_results,
            "statistics": standard_stats,
        },
        "mellea": {
            "results": mellea_results,
            "statistics": mellea_stats,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Detailed results saved to: {args.output}")

    # Return exit code based on success
    if standard_stats.get("successful_runs", 0) > 0 or mellea_stats.get("successful_runs", 0) > 0:
        return 0
    else:
        print("\n⚠️  All runs failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
