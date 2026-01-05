"""
Pattern loader for discovering and loading decorated patterns.

Handles pattern module imports, validation, and stage function retrieval.
"""
import importlib
import importlib.util
from pathlib import Path
from typing import Callable, Optional

from patterns.decorators import (
    StageMetadata,
    get_pattern_stages,
    has_decorated_pattern,
    PatternContext,
)
from config import PATTERNS_DIR


class PatternLoader:
    """Discovers and loads decorated patterns."""

    REQUIRED_STAGES = ["map", "optimize", "execute", "post_process"]

    def __init__(self):
        """Initialize pattern loader."""
        self._loaded_patterns: set[str] = set()

    def load_pattern(self, pattern_name: str) -> dict[str, StageMetadata]:
        """
        Load all stages for a pattern.

        Attempts to import the pattern module which triggers decorator
        registration, then validates that all required stages are present.

        Args:
            pattern_name: Name of the pattern to load

        Returns:
            Dictionary mapping stage names to StageMetadata

        Raises:
            ImportError: If pattern module cannot be imported
            ValueError: If pattern is missing required stages
        """
        # Try to import pattern module to trigger decorator registration
        if pattern_name not in self._loaded_patterns:
            self._import_pattern_module(pattern_name)
            self._loaded_patterns.add(pattern_name)

        # Get registered stages
        try:
            stages = get_pattern_stages(pattern_name)
        except KeyError:
            raise ValueError(
                f"Pattern '{pattern_name}' has no decorated stages. "
                "Ensure the pattern module imports and decorators are applied."
            )

        # Validate pattern completeness
        if not self.validate_pattern(pattern_name):
            missing = set(self.REQUIRED_STAGES) - set(stages.keys())
            raise ValueError(
                f"Pattern '{pattern_name}' is missing required stages: {missing}"
            )

        return stages

    def _import_pattern_module(self, pattern_name: str):
        """
        Import the pattern module to trigger decorator registration.

        Tries multiple import strategies:
        1. Import patterns.{pattern_name}.pattern
        2. Import patterns.{pattern_name}

        Args:
            pattern_name: Name of the pattern

        Raises:
            ImportError: If pattern module cannot be imported
        """
        # Try patterns.{pattern_name}.pattern first
        try:
            module = importlib.import_module(f"patterns.{pattern_name}.pattern")
            return module
        except ModuleNotFoundError:
            pass

        # Try patterns.{pattern_name} (if pattern.py is in __init__.py)
        try:
            module = importlib.import_module(f"patterns.{pattern_name}")
            return module
        except ModuleNotFoundError:
            pass

        # Try loading pattern.py file directly
        pattern_file = PATTERNS_DIR / pattern_name / "pattern.py"
        if pattern_file.exists():
            spec = importlib.util.spec_from_file_location(
                f"patterns.{pattern_name}.pattern",
                pattern_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

        raise ImportError(
            f"Could not import pattern '{pattern_name}'. "
            f"Expected 'patterns/{pattern_name}/pattern.py' or "
            f"'patterns/{pattern_name}/__init__.py' with decorated stages."
        )

    def validate_pattern(self, pattern_name: str) -> bool:
        """
        Validate that a pattern has all required stages.

        Args:
            pattern_name: Name of the pattern

        Returns:
            True if pattern has all required stages, False otherwise
        """
        if not has_decorated_pattern(pattern_name):
            return False

        try:
            stages = get_pattern_stages(pattern_name)
        except KeyError:
            return False

        # Check all required stages are present
        for stage_name in self.REQUIRED_STAGES:
            if stage_name not in stages:
                return False

        return True

    def has_decorated_stage(self, pattern_name: str, stage_name: str) -> bool:
        """
        Check if a specific stage is decorated for a pattern.

        Args:
            pattern_name: Name of the pattern
            stage_name: Name of the stage

        Returns:
            True if the stage is decorated, False otherwise
        """
        if not has_decorated_pattern(pattern_name):
            return False

        try:
            stages = get_pattern_stages(pattern_name)
            return stage_name in stages
        except KeyError:
            return False

    def get_stage_function(
        self,
        pattern_name: str,
        stage_name: str
    ) -> Callable[[PatternContext], dict]:
        """
        Get the callable function for a specific stage.

        Args:
            pattern_name: Name of the pattern
            stage_name: Name of the stage

        Returns:
            Stage function that accepts PatternContext and returns dict

        Raises:
            KeyError: If pattern or stage not found
        """
        stages = get_pattern_stages(pattern_name)
        if stage_name not in stages:
            raise KeyError(
                f"Stage '{stage_name}' not found in pattern '{pattern_name}'"
            )

        return stages[stage_name].func

    def get_stage_metadata(
        self,
        pattern_name: str,
        stage_name: str
    ) -> StageMetadata:
        """
        Get metadata for a specific stage.

        Args:
            pattern_name: Name of the pattern
            stage_name: Name of the stage

        Returns:
            StageMetadata for the stage

        Raises:
            KeyError: If pattern or stage not found
        """
        stages = get_pattern_stages(pattern_name)
        if stage_name not in stages:
            raise KeyError(
                f"Stage '{stage_name}' not found in pattern '{pattern_name}'"
            )

        return stages[stage_name]

    def get_agent_type(self, pattern_name: str, stage_name: str) -> str:
        """
        Get the agent type for a specific stage.

        Args:
            pattern_name: Name of the pattern
            stage_name: Name of the stage

        Returns:
            Agent type ("classical" or "quantum")

        Raises:
            KeyError: If pattern or stage not found
        """
        metadata = self.get_stage_metadata(pattern_name, stage_name)
        return metadata.agent_type

    def list_pattern_stages(self, pattern_name: str) -> list[str]:
        """
        List all stages for a pattern in execution order.

        Args:
            pattern_name: Name of the pattern

        Returns:
            List of stage names in execution order

        Raises:
            KeyError: If pattern not found
        """
        stages = get_pattern_stages(pattern_name)

        # Return in standard execution order
        stage_order = ["map", "optimize", "execute", "post_process"]
        return [s for s in stage_order if s in stages]

    def get_dependency_graph(self, pattern_name: str) -> dict[str, list[str]]:
        """
        Get the dependency graph for a pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Dictionary mapping stage names to their dependencies

        Raises:
            KeyError: If pattern not found
        """
        stages = get_pattern_stages(pattern_name)
        return {
            stage_name: metadata.dependencies
            for stage_name, metadata in stages.items()
        }

    def validate_dependencies(self, pattern_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate that all stage dependencies are satisfied and acyclic.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Tuple of (is_valid, error_message)
            is_valid is True if dependencies are valid, False otherwise
            error_message is None if valid, otherwise describes the issue
        """
        stages = get_pattern_stages(pattern_name)

        # Check all dependencies exist
        for stage_name, metadata in stages.items():
            for dep in metadata.dependencies:
                if dep not in stages:
                    return False, f"Stage '{stage_name}' depends on '{dep}' which doesn't exist"

        # Check for cycles using DFS
        def has_cycle(stage: str, visited: set, rec_stack: set) -> bool:
            visited.add(stage)
            rec_stack.add(stage)

            for dep in stages[stage].dependencies:
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(stage)
            return False

        visited = set()
        for stage_name in stages:
            if stage_name not in visited:
                if has_cycle(stage_name, visited, set()):
                    return False, "Circular dependency detected in stage graph"

        return True, None
