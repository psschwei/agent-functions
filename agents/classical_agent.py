"""
Classical Agent

Executes classical workload stages (map, optimize, post-process) on Ray cluster.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time

from executors.ray_executor import run_script_on_ray
from patterns.decorators import PatternContext


class ClassicalAgent:
    """Agent for executing classical computation stages on Ray cluster."""

    def __init__(self, name: str = "ClassicalAgent"):
        """
        Initialize the classical agent.

        Args:
            name: Agent name for logging
        """
        self.name = name

    def execute_stage(
        self,
        stage_name: str,
        script_path: Path,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a classical stage on the Ray cluster.

        Args:
            stage_name: Name of the stage (e.g., "map", "optimize", "post_process")
            script_path: Path to the Python script to execute
            input_path: Optional input file path
            output_path: Optional output file path
            **kwargs: Additional arguments to pass to the script

        Returns:
            Dictionary with execution status and results
        """
        print(f"\n[{self.name}] Executing {stage_name} stage...")
        print(f"  Script: {script_path}")
        if input_path:
            print(f"  Input: {input_path}")
        if output_path:
            print(f"  Output: {output_path}")

        start_time = time.perf_counter()

        try:
            # Execute script on Ray cluster
            result = run_script_on_ray(
                script_path=script_path,
                input_path=input_path,
                output_path=output_path,
                **kwargs
            )

            end_time = time.perf_counter()
            duration = end_time - start_time

            if result["status"] == "success":
                print(f"[{self.name}] ✓ {stage_name} stage completed in {duration:.2f}s")
                return {
                    "status": "success",
                    "stage": stage_name,
                    "output_path": str(output_path) if output_path else None,
                    "duration": duration,
                    "agent": self.name,
                }
            else:
                print(f"[{self.name}] ✗ {stage_name} stage failed!")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                if result.get('stderr'):
                    print(f"  Stderr: {result['stderr']}")
                return {
                    "status": "failed",
                    "stage": stage_name,
                    "error": result.get("error", "Unknown error"),
                    "stderr": result.get("stderr", ""),
                    "duration": duration,
                    "agent": self.name,
                }

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            print(f"[{self.name}] ✗ Exception in {stage_name} stage: {str(e)}")
            return {
                "status": "failed",
                "stage": stage_name,
                "error": str(e),
                "duration": duration,
                "agent": self.name,
            }

    def run_map_stage(
        self,
        script_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Execute the map stage."""
        return self.execute_stage(
            stage_name="map",
            script_path=script_path,
            output_path=output_path
        )

    def run_optimize_stage(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Execute the optimize stage."""
        return self.execute_stage(
            stage_name="optimize",
            script_path=script_path,
            input_path=input_path,
            output_path=output_path
        )

    def run_post_process_stage(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        summary_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Execute the post-process stage."""
        kwargs = {}
        if summary_path:
            kwargs["summary"] = summary_path

        return self.execute_stage(
            stage_name="post_process",
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            **kwargs
        )

    def run_decorated_stage(
        self,
        stage_func: Callable[[PatternContext], dict],
        ctx: PatternContext,
        stage_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a decorated pattern stage function.

        Args:
            stage_func: The decorated stage function to execute
            ctx: PatternContext for the stage
            stage_name: Optional stage name (defaults to ctx.stage_name)

        Returns:
            Dictionary with execution status and results
        """
        stage_name = stage_name or ctx.stage_name

        print(f"\n[{self.name}] Executing decorated {stage_name} stage...")

        start_time = time.perf_counter()

        try:
            # Execute the decorated function
            result = stage_func(ctx)

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Get output path from context
            output_path = ctx.get_output_path()

            print(f"[{self.name}] ✓ {stage_name} stage completed in {duration:.2f}s")

            return {
                "status": "success",
                "stage": stage_name,
                "output_path": str(output_path),
                "duration": duration,
                "agent": self.name,
                "logs": ctx.get_logs(),
            }

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            print(f"[{self.name}] ✗ Exception in decorated {stage_name} stage: {str(e)}")

            return {
                "status": "failed",
                "stage": stage_name,
                "error": str(e),
                "duration": duration,
                "agent": self.name,
                "logs": ctx.get_logs(),
            }
