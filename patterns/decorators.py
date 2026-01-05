"""
Decorator-based pattern stage system for Qiskit patterns.

Provides stage-specific decorators (@map_stage, @optimize_stage, etc.) that
enable patterns to be defined in a single file instead of four separate scripts.
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional, Literal
from functools import wraps

from config import DATA_DIR


@dataclass
class StageMetadata:
    """Metadata for a decorated pattern stage."""

    stage_name: Literal["map", "optimize", "execute", "post_process"]
    agent_type: Literal["classical", "quantum"]
    func: Callable
    dependencies: list[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry: int = 0

    def __post_init__(self):
        """Set default dependencies based on stage."""
        if not self.dependencies:
            # Linear pipeline by default
            dependency_map = {
                "map": [],
                "optimize": ["map"],
                "execute": ["optimize"],
                "post_process": ["execute"],
            }
            self.dependencies = dependency_map[self.stage_name]


# Global registry: {pattern_name: {stage_name: StageMetadata}}
_PATTERN_REGISTRY: dict[str, dict[str, StageMetadata]] = {}


class PatternContext:
    """
    Context passed to each pattern stage function.

    Provides helpers for I/O, configuration access, and logging.
    """

    def __init__(
        self,
        state: dict,
        stage_name: str,
        pattern_name: str,
    ):
        """
        Initialize pattern context.

        Args:
            state: Current PatternState dict
            stage_name: Name of the current stage
            pattern_name: Name of the pattern being executed
        """
        self.state = state
        self.stage_name = stage_name
        self.pattern_name = pattern_name
        self._logs: list[str] = []

    def load_input(self) -> dict:
        """
        Load output from the previous stage.

        Returns:
            Dictionary containing previous stage's output data

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If this is the first stage (no input to load)
        """
        input_path = self.get_input_path()
        if input_path is None:
            raise ValueError(f"Stage '{self.stage_name}' is the first stage - no input to load")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'rb') as f:
            return pickle.load(f)

    def save_output(self, data: dict) -> Path:
        """
        Save stage output to pickle file.

        Args:
            data: Dictionary to save as output

        Returns:
            Path where data was saved
        """
        output_path = self.get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        return output_path

    def get_input_path(self) -> Optional[Path]:
        """
        Get path to previous stage's output.

        Returns:
            Path to input file, or None if this is the first stage
        """
        # Determine previous stage
        stage_order = ["map", "optimize", "execute", "post_process"]
        try:
            current_idx = stage_order.index(self.stage_name)
        except ValueError:
            raise ValueError(f"Unknown stage: {self.stage_name}")

        if current_idx == 0:
            return None  # First stage has no input

        previous_stage = stage_order[current_idx - 1]
        return DATA_DIR / f"{self.pattern_name}_{previous_stage}_result.pkl"

    def get_output_path(self) -> Path:
        """
        Get path for this stage's output.

        Returns:
            Path where output should be saved
        """
        return DATA_DIR / f"{self.pattern_name}_{self.stage_name}_result.pkl"

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Access pattern-specific config from state.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.state.get(key, default)

    def log(self, message: str, level: str = "INFO"):
        """
        Log message from stage execution.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        log_entry = f"[{level}] {self.stage_name}: {message}"
        self._logs.append(log_entry)
        print(log_entry)

    def get_logs(self) -> list[str]:
        """Get all logged messages."""
        return self._logs.copy()


def _create_stage_decorator(
    stage_name: Literal["map", "optimize", "execute", "post_process"],
    default_agent_type: Literal["classical", "quantum"],
):
    """
    Factory function to create stage-specific decorators.

    Args:
        stage_name: Name of the stage
        default_agent_type: Default agent type for this stage

    Returns:
        Decorator function
    """
    def decorator(
        timeout: Optional[int] = None,
        retry: int = 0,
        dependencies: Optional[list[str]] = None,
        agent_type: Optional[Literal["classical", "quantum"]] = None,
    ):
        """
        Decorator for pattern stage functions.

        Args:
            timeout: Execution timeout in seconds
            retry: Number of retry attempts on failure
            dependencies: Explicit stage dependencies (default: linear pipeline)
            agent_type: Override default agent type
        """
        def wrapper(func: Callable[[PatternContext], dict]):
            # Determine pattern name from module
            pattern_name = func.__module__.split('.')[1] if '.' in func.__module__ else "unknown"

            # Create metadata
            metadata = StageMetadata(
                stage_name=stage_name,
                agent_type=agent_type or default_agent_type,
                func=func,
                dependencies=dependencies or [],
                timeout=timeout,
                retry=retry,
            )

            # Register in global registry
            if pattern_name not in _PATTERN_REGISTRY:
                _PATTERN_REGISTRY[pattern_name] = {}
            _PATTERN_REGISTRY[pattern_name][stage_name] = metadata

            # Store metadata on function
            func._pattern_metadata = metadata
            func._pattern_name = pattern_name

            @wraps(func)
            def decorated_func(ctx: PatternContext) -> dict:
                """Wrapped function that handles automatic output saving."""
                # Execute the stage function
                result = func(ctx)

                # Automatically save output
                if result is not None:
                    ctx.save_output(result)

                return result

            # Preserve metadata on wrapped function
            decorated_func._pattern_metadata = metadata
            decorated_func._pattern_name = pattern_name

            return decorated_func

        # Support both @decorator and @decorator() syntax
        if callable(timeout):
            # Called as @decorator (timeout is actually the function)
            func = timeout
            timeout = None
            return wrapper(func)
        else:
            # Called as @decorator(...) with parameters
            return wrapper

    return decorator


# Create stage-specific decorators
map_stage = _create_stage_decorator("map", "classical")
optimize_stage = _create_stage_decorator("optimize", "classical")
execute_stage = _create_stage_decorator("execute", "quantum")
post_process_stage = _create_stage_decorator("post_process", "classical")


def get_pattern_registry() -> dict[str, dict[str, StageMetadata]]:
    """
    Get the global pattern registry.

    Returns:
        Dictionary mapping pattern names to stage metadata
    """
    return _PATTERN_REGISTRY.copy()


def clear_pattern_registry():
    """Clear the global pattern registry (useful for testing)."""
    _PATTERN_REGISTRY.clear()


def get_pattern_stages(pattern_name: str) -> dict[str, StageMetadata]:
    """
    Get all registered stages for a pattern.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Dictionary mapping stage names to StageMetadata

    Raises:
        KeyError: If pattern not found in registry
    """
    if pattern_name not in _PATTERN_REGISTRY:
        raise KeyError(f"Pattern '{pattern_name}' not found in registry")
    return _PATTERN_REGISTRY[pattern_name].copy()


def has_decorated_pattern(pattern_name: str) -> bool:
    """
    Check if a pattern has decorated stages registered.

    Args:
        pattern_name: Name of the pattern

    Returns:
        True if pattern has decorated stages, False otherwise
    """
    return pattern_name in _PATTERN_REGISTRY and len(_PATTERN_REGISTRY[pattern_name]) > 0
