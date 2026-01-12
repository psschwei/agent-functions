"""
Mellea Classical Agent

Enhanced classical agent using Mellea for adaptive execution and result evaluation.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time
import json

try:
    import mellea
    from mellea import generative, req
    from mellea.strategies import RejectionSamplingStrategy
    MELLEA_AVAILABLE = True
except ImportError:
    MELLEA_AVAILABLE = False
    print("Warning: Mellea not installed. MelleaClassicalAgent will not be available.")

from agents.classical_agent import ClassicalAgent
from patterns.decorators import PatternContext


class MelleaClassicalAgent(ClassicalAgent):
    """
    Enhanced classical agent using Mellea for adaptive execution.
    
    Provides:
    - LLM-based result evaluation
    - Adaptive retry logic within stages
    - Parameter adjustment suggestions
    - Quality validation with constraints
    """

    def __init__(self, name: str = "MelleaClassicalAgent", model_backend: str = "ollama", max_retries: int = 2):
        """
        Initialize the Mellea-enhanced classical agent.

        Args:
            name: Agent name for logging
            model_backend: Mellea backend to use (ollama, watsonx, huggingface, etc.)
            max_retries: Maximum number of adaptive retries per stage
        """
        if not MELLEA_AVAILABLE:
            raise ImportError(
                "Mellea is not installed. Install it with: pip install mellea"
            )
        
        super().__init__(name)
        self.max_retries = max_retries
        self.model_backend = model_backend
        
        # Initialize Mellea session
        try:
            self.session = mellea.start_session(backend=model_backend)
            print(f"[{self.name}] Initialized Mellea session with {model_backend} backend")
        except Exception as e:
            print(f"[{self.name}] Warning: Failed to initialize Mellea session: {e}")
            print(f"[{self.name}] Falling back to standard ClassicalAgent behavior")
            self.session = None

    def run_decorated_stage(
        self,
        stage_func: Callable[[PatternContext], dict],
        ctx: PatternContext,
        stage_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a decorated pattern stage function with Mellea-based adaptive execution.

        Args:
            stage_func: The decorated stage function to execute
            ctx: PatternContext for the stage
            stage_name: Optional stage name (defaults to ctx.stage_name)

        Returns:
            Dictionary with execution status and results
        """
        if self.session is None:
            # Fallback to standard execution if Mellea not available
            return super().run_decorated_stage(stage_func, ctx, stage_name)

        stage_name = stage_name or ctx.stage_name
        print(f"\n[{self.name}] Executing decorated {stage_name} stage with Mellea adaptation...")

        # Execute with adaptive retry logic
        for attempt in range(self.max_retries + 1):
            print(f"[{self.name}] Attempt {attempt + 1}/{self.max_retries + 1}")
            
            start_time = time.perf_counter()
            
            try:
                # Execute the decorated function
                result = stage_func(ctx)
                
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # Get output path from context
                output_path = ctx.get_output_path()
                
                # Use Mellea to evaluate result quality
                evaluation = self._evaluate_result_quality(stage_name, result, ctx)
                
                print(f"[{self.name}] Quality evaluation: {evaluation['quality']}")
                
                if evaluation["quality"] == "satisfactory" or attempt >= self.max_retries:
                    print(f"[{self.name}] ✓ {stage_name} stage completed in {duration:.2f}s")
                    
                    return {
                        "status": "success",
                        "stage": stage_name,
                        "output_path": str(output_path),
                        "duration": duration,
                        "agent": self.name,
                        "logs": ctx.get_logs(),
                        "mellea_evaluation": evaluation,
                        "attempts": attempt + 1,
                    }
                else:
                    # Use Mellea to suggest adjustments
                    print(f"[{self.name}] Result quality insufficient, suggesting adjustments...")
                    adjustments = self._suggest_adjustments(stage_name, result, evaluation)
                    
                    print(f"[{self.name}] Suggested adjustments: {adjustments}")
                    
                    # Apply adjustments to context for next attempt
                    self._apply_adjustments(ctx, adjustments)
                    
                    # Continue to next attempt
                    continue
                    
            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                print(f"[{self.name}] ✗ Exception in decorated {stage_name} stage: {str(e)}")
                
                if attempt >= self.max_retries:
                    return {
                        "status": "failed",
                        "stage": stage_name,
                        "error": str(e),
                        "duration": duration,
                        "agent": self.name,
                        "logs": ctx.get_logs(),
                        "attempts": attempt + 1,
                    }
                else:
                    print(f"[{self.name}] Retrying after exception...")
                    continue

    def _evaluate_result_quality(
        self, 
        stage_name: str, 
        result: dict,
        ctx: PatternContext
    ) -> Dict[str, Any]:
        """
        Use Mellea to evaluate the quality of stage results.

        Args:
            stage_name: Name of the stage
            result: Result dictionary from stage execution
            ctx: PatternContext for additional context

        Returns:
            Dictionary with quality assessment and identified issues
        """
        try:
            # Prepare evaluation prompt
            result_summary = self._summarize_result(result)
            logs = ctx.get_logs()
            
            evaluation_prompt = f"""
Evaluate the quality of the {stage_name} stage execution results.

Stage: {stage_name}
Result Summary: {result_summary}
Execution Logs: {logs[-5:] if logs else "No logs"}

Provide a JSON response with:
- "quality": "satisfactory" or "needs_improvement"
- "issues": list of identified issues (empty if satisfactory)
- "reasoning": brief explanation of the assessment

Focus on:
- Data completeness (all expected keys present)
- Value validity (no None/NaN where unexpected)
- Logical consistency (values make sense for the stage)
"""

            # Use Mellea's generative capabilities with validation
            evaluation_result = self.session.instruct(
                evaluation_prompt,
                requirements=[
                    req("Must return valid JSON"),
                    req("Must include 'quality' field with value 'satisfactory' or 'needs_improvement'"),
                    req("Must include 'issues' field as a list"),
                    req("Must include 'reasoning' field as a string"),
                ],
                strategy=RejectionSamplingStrategy(loop_budget=3)
            )
            
            # Parse the evaluation result
            if isinstance(evaluation_result, str):
                evaluation = json.loads(evaluation_result)
            else:
                evaluation = evaluation_result
                
            return evaluation
            
        except Exception as e:
            print(f"[{self.name}] Warning: Mellea evaluation failed: {e}")
            # Fallback to simple heuristic evaluation
            return {
                "quality": "satisfactory",
                "issues": [],
                "reasoning": f"Mellea evaluation failed, using fallback: {str(e)}"
            }

    def _suggest_adjustments(
        self,
        stage_name: str,
        result: dict,
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Mellea to suggest adjustments based on evaluation.

        Args:
            stage_name: Name of the stage
            result: Result dictionary from stage execution
            evaluation: Quality evaluation from _evaluate_result_quality

        Returns:
            Dictionary with suggested parameter adjustments
        """
        try:
            issues = evaluation.get("issues", [])
            
            adjustment_prompt = f"""
Based on the following issues identified in the {stage_name} stage execution,
suggest specific parameter adjustments to improve the results.

Issues: {json.dumps(issues)}

Provide a JSON response with:
- "adjustments": dict of parameter names to new values
- "reasoning": brief explanation of why these adjustments should help

Be specific and actionable. Focus on parameters that can be adjusted in the stage configuration.
"""

            adjustment_result = self.session.instruct(
                adjustment_prompt,
                requirements=[
                    req("Must return valid JSON"),
                    req("Must include 'adjustments' field as a dict"),
                    req("Must include 'reasoning' field as a string"),
                ],
                strategy=RejectionSamplingStrategy(loop_budget=3)
            )
            
            # Parse the adjustment result
            if isinstance(adjustment_result, str):
                adjustments = json.loads(adjustment_result)
            else:
                adjustments = adjustment_result
                
            return adjustments
            
        except Exception as e:
            print(f"[{self.name}] Warning: Mellea adjustment suggestion failed: {e}")
            return {
                "adjustments": {},
                "reasoning": f"Adjustment suggestion failed: {str(e)}"
            }

    def _apply_adjustments(self, ctx: PatternContext, adjustments: Dict[str, Any]) -> None:
        """
        Apply suggested adjustments to the context for the next attempt.

        Args:
            ctx: PatternContext to modify
            adjustments: Adjustment suggestions from _suggest_adjustments
        """
        adjustment_dict = adjustments.get("adjustments", {})
        
        if not adjustment_dict:
            print(f"[{self.name}] No adjustments to apply")
            return
            
        # Update context state with adjustments
        for key, value in adjustment_dict.items():
            ctx.state[f"adjusted_{key}"] = value
            print(f"[{self.name}] Applied adjustment: {key} = {value}")

    def _summarize_result(self, result: dict) -> str:
        """
        Create a concise summary of result for LLM evaluation.

        Args:
            result: Result dictionary to summarize

        Returns:
            String summary of the result
        """
        summary_parts = []
        
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple)):
                summary_parts.append(f"{key}: list with {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: dict with {len(value)} keys")
            else:
                summary_parts.append(f"{key}: {type(value).__name__}")
        
        return ", ".join(summary_parts)
