"""
Quantum Agent

Executes quantum workload stages (execute) using Qiskit on AerSimulator.
"""
from pathlib import Path
from typing import Dict, Any
import time
import subprocess
import sys


class QuantumAgent:
    """Agent for executing quantum computation stages using Qiskit."""

    def __init__(self, name: str = "QuantumAgent"):
        """
        Initialize the quantum agent.

        Args:
            name: Agent name for logging
        """
        self.name = name

    def execute_stage(
        self,
        stage_name: str,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a quantum stage using Qiskit.

        Args:
            stage_name: Name of the stage (e.g., "execute")
            script_path: Path to the Python script to execute
            input_path: Input file path from previous stage
            output_path: Output file path for results
            **kwargs: Additional arguments to pass to the script

        Returns:
            Dictionary with execution status and results
        """
        print(f"\n[{self.name}] Executing {stage_name} stage...")
        print(f"  Script: {script_path}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")

        start_time = time.perf_counter()

        try:
            # Build command
            cmd = [sys.executable, str(script_path)]
            cmd.extend(["--input", str(input_path)])
            cmd.extend(["--output", str(output_path)])

            # Add any additional arguments
            for key, value in kwargs.items():
                cmd.extend([f"--{key}", str(value)])

            # Execute the script directly (Qiskit execution doesn't need Ray)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            end_time = time.perf_counter()
            duration = end_time - start_time

            print(f"[{self.name}] ✓ {stage_name} stage completed in {duration:.2f}s")

            return {
                "status": "success",
                "stage": stage_name,
                "output_path": str(output_path),
                "duration": duration,
                "agent": self.name,
                "stdout": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            print(f"[{self.name}] ✗ {stage_name} stage failed!")
            print(f"  Error: {str(e)}")
            if e.stderr:
                print(f"  Stderr: {e.stderr}")

            return {
                "status": "failed",
                "stage": stage_name,
                "error": str(e),
                "stderr": e.stderr,
                "stdout": e.stdout,
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

    def run_execute_stage(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Execute the quantum execution stage."""
        return self.execute_stage(
            stage_name="execute",
            script_path=script_path,
            input_path=input_path,
            output_path=output_path
        )
