"""Ray executor for running classical workloads on a Ray cluster."""
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import ray

from config import RAY_CONFIG


def init_ray_cluster() -> None:
    """Initialize a local Ray cluster if not already initialized."""
    if not ray.is_initialized():
        ray.init(**RAY_CONFIG)
        print(f"Ray cluster initialized with {RAY_CONFIG['num_cpus']} CPUs")
    else:
        print("Ray cluster already initialized")


def shutdown_ray_cluster() -> None:
    """Shutdown the Ray cluster."""
    if ray.is_initialized():
        ray.shutdown()
        print("Ray cluster shutdown complete")


@ray.remote
def execute_python_script(
    script_path: str,
    input_path: str = None,
    output_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a Python script as a Ray remote task.

    Args:
        script_path: Path to the Python script to execute
        input_path: Optional input file path to pass to the script
        output_path: Optional output file path for the script
        **kwargs: Additional arguments to pass to the script

    Returns:
        Dictionary with execution status and output path
    """
    try:
        # Build command
        cmd = [sys.executable, str(script_path)]

        if input_path:
            cmd.extend(["--input", str(input_path)])

        if output_path:
            cmd.extend(["--output", str(output_path)])

        # Add any additional arguments
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        # Execute the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return {
            "status": "success",
            "output_path": output_path,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "failed",
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


def run_script_on_ray(
    script_path: Path,
    input_path: Path = None,
    output_path: Path = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Submit a Python script to the Ray cluster and wait for results.

    Args:
        script_path: Path to the Python script to execute
        input_path: Optional input file path
        output_path: Optional output file path
        **kwargs: Additional script arguments

    Returns:
        Execution result dictionary
    """
    # Submit task to Ray
    future = execute_python_script.remote(
        str(script_path),
        str(input_path) if input_path else None,
        str(output_path) if output_path else None,
        **kwargs
    )

    # Wait for result
    result = ray.get(future)

    return result
