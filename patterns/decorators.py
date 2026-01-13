"""
Decorator-based pattern stage system for Qiskit patterns.

Provides stage-specific decorators (@map_stage, @optimize_stage, etc.) that
enable patterns to be defined in a single file instead of four separate scripts.
"""
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional, Literal, List, Dict
from functools import wraps
from datetime import datetime

from config import DATA_DIR, LLM_CONFIG, ORCHESTRATOR_CONFIG
from agents.llm_client import LLMClient


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

    Provides helpers for I/O, configuration access, logging, and LLM decision-making.
    """

    def __init__(
        self,
        state: dict,
        stage_name: str,
        pattern_name: str,
        enable_llm: bool = False,
    ):
        """
        Initialize pattern context.

        Args:
            state: Current PatternState dict
            stage_name: Name of the current stage
            pattern_name: Name of the pattern being executed
            enable_llm: Enable LLM-powered decision-making
        """
        self.state = state
        self.stage_name = stage_name
        self.pattern_name = pattern_name
        self.enable_llm = enable_llm
        self._logs: list[str] = []
        self._decisions: list[dict] = []

        # Initialize LLM client if enabled
        self.llm_client: Optional[LLMClient] = None
        if enable_llm:
            self.llm_client = LLMClient(**LLM_CONFIG)

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

    def decide(
        self,
        question: str,
        options: List[Any],
        context: Optional[Dict[str, Any]] = None,
        default: Optional[Any] = None,
    ) -> Any:
        """
        Make a decision using LLM reasoning.

        Args:
            question: The decision question to ask the LLM
            options: List of possible options to choose from
            context: Additional context to provide to the LLM
            default: Default option if LLM disabled or fails

        Returns:
            The selected option from the options list
        """
        if not self.enable_llm or self.llm_client is None:
            # Use default if provided, otherwise use first option
            selected = default if default is not None else options[0]
            self.log(f"LLM disabled - using default option: {selected}", level="INFO")
            self._log_decision(question, str(selected), "default (LLM disabled)", context)
            return selected

        try:
            # Construct prompt for LLM
            context_str = json.dumps(context, indent=2) if context else "None"
            options_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

            prompt = f"""You are making a decision for the {self.stage_name} stage of the {self.pattern_name} pattern.

Question: {question}

Available options:
{options_str}

Context:
{context_str}

Analyze the question and context, then respond with ONLY the number (1, 2, 3, etc.) corresponding to your chosen option.
Explain your reasoning briefly, then end with "DECISION: <number>".

Example response format:
The circuit has low depth and gate count, so optimization overhead is not needed.
DECISION: 1
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(messages)

            # Parse decision from response
            content = response["content"]
            if content and "DECISION:" in content:
                decision_text = content.split("DECISION:")[-1].strip()
                # Extract first number from decision text
                import re
                match = re.search(r'\d+', decision_text)
                if match:
                    decision_idx = int(match.group()) - 1
                    if 0 <= decision_idx < len(options):
                        selected = options[decision_idx]
                        reasoning = content.split("DECISION:")[0].strip()
                        self.log(f"LLM decision: {selected}", level="INFO")
                        self.log(f"Reasoning: {reasoning}", level="INFO")
                        self._log_decision(question, str(selected), reasoning, context)
                        return selected

            # If parsing failed, use default
            self.log("Failed to parse LLM decision - using default", level="WARNING")
            selected = default if default is not None else options[0]
            self._log_decision(question, str(selected), "parsing failed, using default", context)
            return selected

        except Exception as e:
            self.log(f"LLM decision failed: {e} - using default", level="ERROR")
            selected = default if default is not None else options[0]
            self._log_decision(question, str(selected), f"error: {e}", context)
            return selected

    def analyze_with_llm(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        default_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze data using LLM reasoning.

        Args:
            prompt: The analysis prompt/question for the LLM
            context: Additional context data for analysis
            default_response: Default response if LLM disabled or fails

        Returns:
            Dictionary containing analysis results
        """
        if not self.enable_llm or self.llm_client is None:
            response = default_response or {"analysis": "LLM disabled", "reasoning": "Using defaults"}
            self.log(f"LLM disabled - using default response", level="INFO")
            self._log_decision(f"Analysis: {prompt[:50]}...", str(response), "default (LLM disabled)", context)
            return response

        try:
            # Construct analysis prompt
            context_str = ""
            if context:
                # Safely convert context to string, handling numpy arrays and complex objects
                context_dict = {}
                for key, value in context.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        context_dict[key] = f"<array with shape {value.shape}>"
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        context_dict[key] = value
                    else:
                        context_dict[key] = str(type(value))
                context_str = f"\n\nContext:\n{json.dumps(context_dict, indent=2)}"

            full_prompt = f"""You are analyzing results for the {self.stage_name} stage of the {self.pattern_name} pattern.

{prompt}{context_str}

Provide your analysis as a JSON object with relevant fields. Include a "reasoning" field explaining your analysis.

Example response format:
{{
    "reasoning": "Your detailed reasoning here",
    "recommendation": "Your recommendation",
    "key_insight": "Main insight from analysis"
}}
"""

            messages = [{"role": "user", "content": full_prompt}]
            response = self.llm_client.chat_completion(messages)

            content = response["content"]
            if content:
                # Try to parse JSON from response
                import re
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        self.log(f"LLM analysis complete", level="INFO")
                        if "reasoning" in analysis:
                            self.log(f"Reasoning: {analysis['reasoning']}", level="INFO")
                        self._log_decision(
                            f"Analysis: {prompt[:50]}...",
                            json.dumps(analysis, indent=2),
                            analysis.get("reasoning", ""),
                            context
                        )
                        return analysis
                    except json.JSONDecodeError:
                        pass

                # If JSON parsing failed, return content as reasoning
                analysis = {"reasoning": content, "raw_response": content}
                self._log_decision(f"Analysis: {prompt[:50]}...", content, content, context)
                return analysis

            # If no content, use default
            response_data = default_response or {"analysis": "No response", "reasoning": "LLM returned empty"}
            self._log_decision(f"Analysis: {prompt[:50]}...", str(response_data), "no content", context)
            return response_data

        except Exception as e:
            self.log(f"LLM analysis failed: {e} - using default", level="ERROR")
            response_data = default_response or {"error": str(e), "reasoning": "Analysis failed"}
            self._log_decision(f"Analysis: {prompt[:50]}...", str(response_data), f"error: {e}", context)
            return response_data

    def _log_decision(
        self,
        question: str,
        decision: str,
        reasoning: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an LLM decision for audit trail.

        Args:
            question: The decision question
            decision: The decision made
            reasoning: The reasoning behind the decision
            context: Additional context
        """
        # Sanitize context for JSON serialization
        sanitized_context = None
        if context:
            sanitized_context = {}
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    sanitized_context[key] = value
                elif hasattr(value, 'tolist'):  # numpy array
                    sanitized_context[key] = f"<array with shape {value.shape}>"
                else:
                    sanitized_context[key] = str(type(value).__name__)

        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "pattern": self.pattern_name,
            "stage": self.stage_name,
            "question": question,
            "decision": decision,
            "reasoning": reasoning,
            "context": sanitized_context,
        }
        self._decisions.append(decision_entry)

        # Also log to file if enabled
        if ORCHESTRATOR_CONFIG.get("enable_logging", True):
            log_path = DATA_DIR / f"stage_decisions_{self.pattern_name}.jsonl"
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(decision_entry) + '\n')
            except (TypeError, ValueError) as e:
                # If serialization still fails, log without context
                decision_entry_safe = decision_entry.copy()
                decision_entry_safe["context"] = f"<serialization error: {e}>"
                with open(log_path, 'a') as f:
                    f.write(json.dumps(decision_entry_safe) + '\n')

    def get_decisions(self) -> List[Dict]:
        """Get all logged decisions."""
        return self._decisions.copy()


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

            # Update metadata to point to the decorated function (with auto-save)
            # so that loader.get_stage_function() returns the wrapper
            metadata.func = decorated_func

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
